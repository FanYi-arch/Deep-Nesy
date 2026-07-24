import os
from typing import Tuple, Optional

import torch
from torch.utils.data import TensorDataset, Dataset as TorchDataset
import numpy as np

from deepproblog.dataset import Dataset
from deepproblog.query import Query

# Default data directory: planner1/data/processed.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PLANNER1_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(PLANNER1_ROOT, 'data', 'processed'))

GOAL_INDEX = {0: 'lane_keep', 1: 'turn_left', 2: 'turn_right', 3: 'stop', 4: 'overtake'}
LATERAL_INDEX = {
    0: 'lane_keep',
    1: 'lane_change_left',
    2: 'lane_change_right',
    3: 'turn_left',
    4: 'turn_right',
    5: 'stop',
}
LONGITUDINAL_INDEX = {0: 'accelerate', 1: 'decelerate', 2: 'cruise', 3: 'stop'}


class DrivingQueryDataset(Dataset):
    """Concrete DeepProbLog dataset wrapping queries for __len__ and to_query."""
    def __init__(self, queries):
        self.queries = queries

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.queries)

    def to_query(self, i: int) -> Query:  # type: ignore[override]
        return self.queries[i]


class MemmapTensorDataset(TorchDataset):
    """Dataset streaming samples from a NumPy memmap (.npy) and normalizing on access."""

    def __init__(self, array: np.memmap, start: int, end: int):
        self.array = array
        self.start = int(start)
        self.end = int(end)

    def __len__(self) -> int:  # type: ignore[override]
        return self.end - self.start

    def __getitem__(self, index: int):  # type: ignore[override]
        pos = self.start + int(index)
        sample = torch.from_numpy(self.array[pos])
        if sample.dtype == torch.uint8:
            sample = sample.float() / 255.0
        else:
            sample = sample.float()
            if sample.max() > 1.0:
                sample = sample / 255.0
        return (sample.contiguous(),)

"""Unsharded dataset loaded from images.pt or memory-mapped images.npy."""


def _resolve_data_dir(data_dir: Optional[str] = None) -> str:
    """Resolve data directory in order of priority:
    1) explicit argument
    2) environment variables: PLANNER1_DATA_DIR, WAYMO_DATA_DIR
    3) default directory inside repo (planner1/data/processed)
    """
    if data_dir:
        candidates = [data_dir]
        if not os.path.isabs(data_dir):
            candidates.append(os.path.abspath(os.path.join(PLANNER1_ROOT, data_dir)))
            candidates.append(os.path.abspath(os.path.join(PROJECT_ROOT, data_dir)))
        for candidate in candidates:
            if os.path.isdir(candidate):
                return os.path.abspath(candidate)
    env_dir = os.environ.get('PLANNER1_DATA_DIR') or os.environ.get('WAYMO_DATA_DIR')
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)
    return DEFAULT_DATA_DIR


def load_waymo_tensors(split: str = 'train', data_dir: Optional[str] = None) -> TorchDataset:
    """Load preprocessed images, preferring an NPY memmap over images.pt."""
    base_dir = _resolve_data_dir(data_dir)
    images_npy = os.path.join(base_dir, 'images.npy')
    images_pt = os.path.join(base_dir, 'images.pt')
    if os.path.exists(images_npy):
        arr = np.load(images_npy, mmap_mode='r')
        n = int(arr.shape[0])
        n_train = int(0.8 * n)
        if split == 'train':
            return MemmapTensorDataset(arr, 0, n_train)
        else:
            return MemmapTensorDataset(arr, n_train, n)
    if os.path.exists(images_pt):
        images = torch.load(images_pt)
        if isinstance(images, torch.Tensor):
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            elif images.max() > 1.0:
                images = images / 255.0
        n = len(images)
        n_train = int(0.8 * n)
        return TensorDataset(images[:n_train]) if split == 'train' else TensorDataset(images[n_train:])
    raise FileNotFoundError(f"Processed images not found in {base_dir}. Run preprocess_waymo_motion.py first.")


INVALID_LABEL = -3

def driving_planning_dataset(
    split: str = 'train',
    data_dir: Optional[str] = None,
    mode: str = 'single',
    limit: Optional[int] = None,
    seed: Optional[int] = 0,
    shuffle: bool = False,
) -> Dataset:
    """Build a DeepProbLog dataset and skip labels unsupported by the mode.

    The image TensorDataset is registered through add_tensor_source('split', ...),
    so query indices must remain aligned with the original split.
    """
    base_dir = _resolve_data_dir(data_dir)
    images_npy = os.path.join(base_dir, 'images.npy')
    images_path = os.path.join(base_dir, 'images.pt')
    if mode == 'single':
        goals_path = os.path.join(base_dir, 'goals.pt')
    elif mode == 'lateral':
        goals_path = os.path.join(base_dir, 'goals_lateral.pt')
    elif mode == 'longitudinal':
        goals_path = os.path.join(base_dir, 'goals_longitudinal.pt')
    elif mode == 'dual':
        goals_lat_path = os.path.join(base_dir, 'goals_lateral.pt')
        goals_lon_path = os.path.join(base_dir, 'goals_longitudinal.pt')
        if not (os.path.exists(goals_lat_path) and os.path.exists(goals_lon_path)):
             raise FileNotFoundError(f"Processed dual tensors not found in {base_dir}.")
        goals_lat = torch.load(goals_lat_path)
        goals_lon = torch.load(goals_lon_path)
        goals = goals_lat # use for length check
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use single|lateral|longitudinal|dual")

    image_exists = os.path.exists(images_npy) or os.path.exists(images_path)
    if mode != 'dual' and not (image_exists and os.path.exists(goals_path)):
        raise FileNotFoundError(
            f"Processed tensors not found for mode={mode} in {base_dir}. Run preprocessing first."
        )

    if mode == 'dual' and not image_exists:
        raise FileNotFoundError(f"Processed images not found in {base_dir}. Run preprocessing first.")

    if mode != 'dual':
        goals = torch.load(goals_path)

    n = len(goals)
    n_train = int(0.8 * n)
    if split == 'train':
        base_indices = list(range(n_train))
    else:
        base_indices = list(range(n_train, n))

    # Optionally shuffle and limit the queries.
    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        perm = torch.randperm(len(base_indices), generator=g).tolist()
        base_indices = [base_indices[i] for i in perm]
    if limit is not None and limit > 0:
        base_indices = base_indices[: min(limit, len(base_indices))]

    from problog.logic import Term, Constant
    queries = []
    
    if mode == 'dual':
        filtered_out = 0
        for i in base_indices:
            lat_idx = int(goals_lat[i].item())
            lon_idx = int(goals_lon[i].item())
            
            if lat_idx not in LATERAL_INDEX or lon_idx not in LONGITUDINAL_INDEX:
                filtered_out += 1
                continue
                
            lat_name = LATERAL_INDEX[lat_idx]
            lon_name = LONGITUDINAL_INDEX[lon_idx]
            
            local_idx = int(i if split == 'train' else (i - n_train))
            tensor_term = Term('tensor', Term(split, Constant(local_idx)))
            # Joint query: driving_plan_dual(tensor, lat, lon)
            q = Query(Term('driving_plan_dual', tensor_term, Term(lat_name), Term(lon_name)))
            queries.append(q)
    else:
        idx_to_name_map = GOAL_INDEX if mode == 'single' else (LATERAL_INDEX if mode == 'lateral' else LONGITUDINAL_INDEX)
        predicate = 'driving_plan' if mode == 'single' else ('driving_plan_lateral' if mode == 'lateral' else 'driving_plan_longitudinal')

        filtered_out = 0
        for i in base_indices:
            goal_idx = int(goals[i].item())
            if goal_idx == INVALID_LABEL or goal_idx not in idx_to_name_map:
                filtered_out += 1
                continue
            goal_name = idx_to_name_map[goal_idx]
            local_idx = int(i if split == 'train' else (i - n_train))
            tensor_term = Term('tensor', Term(split, Constant(local_idx)))
            q = Query(Term(predicate, tensor_term, Term(goal_name)))
            queries.append(q)

    if filtered_out > 0:
        print(f"INFO: Filtered out {filtered_out} samples with invalid labels in {split} set for mode {mode}.")

    return DrivingQueryDataset(queries)


# Convenience function returning (train_tensor_ds, test_tensor_ds).
def get_tensor_sources(data_dir: Optional[str] = None) -> Tuple[TorchDataset, TorchDataset]:
    """Return image-only train and test datasets, preferring memory mapping."""
    base_dir = _resolve_data_dir(data_dir)
    images_npy = os.path.join(base_dir, 'images.npy')
    images_pt = os.path.join(base_dir, 'images.pt')
    if os.path.exists(images_npy):
        arr = np.load(images_npy, mmap_mode='r')
        n = int(arr.shape[0])
        n_train = int(0.8 * n)
        return MemmapTensorDataset(arr, 0, n_train), MemmapTensorDataset(arr, n_train, n)
    if os.path.exists(images_pt):
        images = torch.load(images_pt)
        if isinstance(images, torch.Tensor):
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            elif images.max() > 1.0:
                images = images / 255.0
        n = len(images)
        n_train = int(0.8 * n)
        return TensorDataset(images[:n_train]), TensorDataset(images[n_train:])
    raise FileNotFoundError(f"Processed images not found in {base_dir}. Run preprocess_waymo_motion.py first.")
