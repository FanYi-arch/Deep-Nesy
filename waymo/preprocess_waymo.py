"""
Waymo Perception Frame preprocessing:
- Read camera-enabled Perception TFRecords and extract 84x84 front-camera images.
- Estimate high-level goals from consecutive poses and speeds:
  lane_keep, turn_left, turn_right, stop, and overtake.
- Save images.pt and goals.pt under the requested output directory.

For Waymo Motion Scenario records named like
"uncompressed_scenario_training_20s_*.tfrecord-*", use
planner1/waymo/preprocess_waymo_motion.py to render bird's-eye-view inputs.

Dependencies required only during preprocessing:
- tensorflow
- a waymo-open-dataset build compatible with the TensorFlow version
"""
import os
import io
import math
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image


def extract_yaw_from_pose(pose_np: np.ndarray) -> float:
    """Extract the yaw angle from a 4x4 pose matrix."""
    # Assume the rotation matrix occupies the upper-left 3x3 block.
    r11 = pose_np[0, 0]
    r21 = pose_np[1, 0]
    yaw = math.atan2(r21, r11)
    return yaw


def resize_to_84(img: Image.Image) -> Image.Image:
    return img.resize((84, 84), Image.BILINEAR)


def label_from_motion(prev_pose: Optional[np.ndarray], cur_pose: np.ndarray, dt: float = 0.1) -> str:
    """Estimate speed and a high-level goal from two consecutive poses.

    Waymo data are sampled at approximately 10 Hz.
    - |delta yaw| > 8 degrees -> turn_left/right
    - speed < 0.5 m/s -> stop
    - else lane_keep
    """
    if prev_pose is None:
        return "lane_keep"

    prev_yaw = extract_yaw_from_pose(prev_pose)
    cur_yaw = extract_yaw_from_pose(cur_pose)

    dyaw = (cur_yaw - prev_yaw)
    # wrap to [-pi, pi]
    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi

    # Estimate translational speed.
    dp = cur_pose[:3, 3] - prev_pose[:3, 3]
    speed = float(np.linalg.norm(dp) / dt)  # m/s

    yaw_deg = abs(dyaw) * 180.0 / math.pi
    if speed < 0.5:
        return "stop"
    if yaw_deg > 8.0:
        return "turn_left" if dyaw > 0 else "turn_right"
    return "lane_keep"


def decode_waymo_frame(frame_bytes) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
    """Decode a Waymo Frame and return the front image and 4x4 pose.

    Requires TensorFlow and waymo-open-dataset.
    """
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2 as open_dataset
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies: tensorflow and waymo-open-dataset are required to read TFRecords."
        ) from e

    frame = open_dataset.Frame()
    frame.ParseFromString(bytes(frame_bytes.numpy() if hasattr(frame_bytes, 'numpy') else frame_bytes))

    # frame.pose.transform stores a row-major array with 16 entries.
    pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)

    front_img = None
    # Select the front-facing camera.
    FRONT = open_dataset.CameraName.FRONT
    for img in frame.images:
        if img.name == FRONT:
            front_img = Image.open(io.BytesIO(img.image)).convert('RGB')
            break

    return front_img, pose


def _prepare_out_dir(out_dir: Optional[str]) -> str:
    """Resolve and create output directory safely with helpful errors."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    # Treat a file-like output path (for example, *.pt) as its parent directory.
    if os.path.splitext(out_dir)[1] in {'.pt', '.pth', '.pkl'}:
        out_dir = os.path.dirname(out_dir)
    out_dir = os.path.abspath(out_dir)
    # Reject an existing non-directory output path.
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise RuntimeError(
            f"Output path exists and is a file: {out_dir}. Please provide a directory path via --out."
        )
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create output directory: {out_dir}. "
            f"Check parent directory permissions or choose a different --out path.\nOriginal error: {e}"
        )
    return out_dir


def _expand_records(records: List[str]) -> List[str]:
    paths: List[str] = []
    for p in records:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith('.tfrecord') or '.tfrecord-' in f:
                    paths.append(os.path.join(p, f))
        else:
            paths.append(p)
    return paths


def preprocess_tfrecords(
    tfrecord_paths: List[str],
    out_dir: str = None,
    limit_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract samples from multiple TFRecords and save them as tensors."""
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow is required for preprocessing Waymo data.") from e

    out_dir = _prepare_out_dir(out_dir)
    tfrecord_paths = _expand_records(tfrecord_paths)
    if len(tfrecord_paths) == 0:
        raise RuntimeError(
            "No TFRecord files found in given --records paths. "
            "If you passed a directory, ensure it contains Perception Frame TFRecords."
        )

    images: List[torch.Tensor] = []
    goals: List[int] = []

    goal_to_idx = {"lane_keep": 0, "turn_left": 1, "turn_right": 2, "stop": 3, "overtake": 4}

    prev_pose = None
    count = 0

    for path in tfrecord_paths:
        dataset = tf.data.TFRecordDataset(path, compression_type='')
        for raw in dataset:
            try:
                img, pose = decode_waymo_frame(raw)
                if img is None or pose is None:
                    continue
                img = resize_to_84(img)
                img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

                goal_name = label_from_motion(prev_pose, pose)
                prev_pose = pose

                images.append(img_t)
                goals.append(goal_to_idx[goal_name])

                count += 1
                if limit_frames is not None and count >= limit_frames:
                    break
            except Exception:
                continue
        if limit_frames is not None and count >= limit_frames:
            break

    if len(images) == 0:
        hint = (
            "No frames decoded. Likely passed Waymo Motion Scenario TFRecords (no camera images). "
            "Use planner1/waymo/preprocess_waymo_motion.py for Motion data, or provide Perception 'segment_*_with_camera.tfrecord'."
        )
        raise RuntimeError(hint)

    images_tensor = torch.stack(images)
    goals_tensor = torch.tensor(goals, dtype=torch.long)

    torch.save(images_tensor, os.path.join(out_dir, 'images.pt'))
    torch.save(goals_tensor, os.path.join(out_dir, 'goals.pt'))

    print(f"Saved {len(images)} samples to {out_dir}")
    return images_tensor, goals_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--records', type=str, nargs='+', required=True,
        help='Perception Frame TFRecord files or directories (with camera images). For Motion Scenario, use preprocess_waymo_motion.py.'
    )
    parser.add_argument('--out', type=str, default=None, help='Output directory (default: planner1/data/processed)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of frames to process')
    args = parser.parse_args()

    preprocess_tfrecords(args.records, args.out, args.limit)
