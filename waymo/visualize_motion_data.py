"""
Visualize tensors produced by the Waymo Motion preprocessing pipeline:
- Load images.pt (N, 3, H, W) and goals.pt (N), including 256x256 inputs.
- Report label-distribution statistics.
- Save representative samples to an output directory (samples/ by default).
- Display either individual images or a matplotlib grid.

Example:
python planner1/waymo/visualize_motion_data.py \
  --data-dir /home/auto/dev_code/deepproblog/waymo_data/processed \
  --num-samples 25 \
  --grid-cols 5

Omit --no-show to display an interactive matplotlib window.
"""
import os
import argparse
import math
import json
from typing import List, Optional, Dict

import torch
import matplotlib
from PIL import Image

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


def _load_images(data_dir: str, limit: Optional[int] = None) -> torch.Tensor:
    """Load images.pt, optionally restricting the result to the first limit samples."""
    images_path = os.path.join(data_dir, 'images.pt')
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Missing images.pt in {data_dir}.")
    t = torch.load(images_path, map_location='cpu')
    if limit is not None and isinstance(t, torch.Tensor) and t.ndim == 4:
        return t[:limit]
    return t


def load_tensors(data_dir: str, limit_images: Optional[int] = None) -> Dict[str, Optional[torch.Tensor]]:
    goals_path = os.path.join(data_dir, 'goals.pt')
    images = _load_images(data_dir, limit=limit_images)
    goals = torch.load(goals_path) if os.path.exists(goals_path) else None
    lat_path = os.path.join(data_dir, 'goals_lateral.pt')
    lon_path = os.path.join(data_dir, 'goals_longitudinal.pt')
    goals_lat = torch.load(lat_path) if os.path.exists(lat_path) else None
    goals_lon = torch.load(lon_path) if os.path.exists(lon_path) else None
    return {'images': images, 'goals': goals, 'goals_lateral': goals_lat, 'goals_longitudinal': goals_lon}


def _summarize_1d(goals: torch.Tensor, idx2name: Dict[int, str]) -> dict:
    stats = {}
    total = int(goals.shape[0])
    for idx, name in idx2name.items():
        count = int((goals == idx).sum().item())
        stats[name] = {
            'count': count,
            'ratio': round(count / total, 4) if total > 0 else 0.0,
        }
    stats['total'] = total
    return stats


def summarize_all(goals: Optional[torch.Tensor], goals_lat: Optional[torch.Tensor], goals_lon: Optional[torch.Tensor]) -> dict:
    out = {}
    if goals is not None:
        out['single'] = _summarize_1d(goals, GOAL_INDEX)
    if goals_lat is not None:
        out['lateral'] = _summarize_1d(goals_lat, LATERAL_INDEX)
    if goals_lon is not None:
        out['longitudinal'] = _summarize_1d(goals_lon, LONGITUDINAL_INDEX)
    # Compute joint statistics when both label tensors are available.
    if (goals_lat is not None) and (goals_lon is not None):
        n = min(int(goals_lat.shape[0]), int(goals_lon.shape[0]))
        joint = [[0 for _ in range(len(LONGITUDINAL_INDEX))] for _ in range(len(LATERAL_INDEX))]
        for i in range(n):
            li = int(goals_lat[i].item())
            lj = int(goals_lon[i].item())
            if 0 <= li < len(joint) and 0 <= lj < len(joint[0]):
                joint[li][lj] += 1
        out['joint'] = {
            'lateral_classes': [LATERAL_INDEX[i] for i in range(len(LATERAL_INDEX))],
            'longitudinal_classes': [LONGITUDINAL_INDEX[j] for j in range(len(LONGITUDINAL_INDEX))],
            'count': joint,
        }
    return out


def save_grid(
    images: torch.Tensor,
    goals: Optional[torch.Tensor],
    out_dir: str,
    num_samples: int,
    grid_cols: int,
    no_show: bool,
    max_grid_tiles: int = 200,
    goals_lat: Optional[torch.Tensor] = None,
    goals_lon: Optional[torch.Tensor] = None,
) -> List[str]:
    """Save samples as paginated grids to avoid oversized figures.

    max_grid_tiles limits each page to at most 200 tiles by default.
    Returns the paths of all saved pages.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = min(num_samples, images.shape[0])
    if n <= 0:
        return []

    max_grid_tiles = max(1, int(max_grid_tiles))
    pages = math.ceil(n / max_grid_tiles)
    saved_paths: List[str] = []

    start = 0
    for p in range(pages):
        this_n = min(max_grid_tiles, n - start)
        grid_rows = math.ceil(this_n / grid_cols)
        # Size each page for the 84x84 source images.
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 1.8, grid_rows * 1.8))
        # Normalize the axes object to a flat iterable.
        if grid_rows == 1 and grid_cols == 1:
            axs_iter = [axs]
        else:
            axs_iter = axs.flatten()

        for i in range(grid_rows * grid_cols):
            ax = axs_iter[i]
            idx = start + i
            if i < this_n and idx < images.shape[0]:
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                title = None
                if goals_lat is not None and goals_lon is not None and idx < goals_lat.shape[0] and idx < goals_lon.shape[0]:
                    li = int(goals_lat[idx].item())
                    lj = int(goals_lon[idx].item())
                    title = f"{LATERAL_INDEX.get(li, 'lat?')} | {LONGITUDINAL_INDEX.get(lj, 'lon?')}"
                elif goals is not None and idx < goals.shape[0]:
                    gi = int(goals[idx].item())
                    title = GOAL_INDEX.get(gi, 'unknown')
                if title is not None:
                    ax.set_title(title)
            ax.axis('off')

        fig.tight_layout()
        out_name = f'samples_grid_{n}_p{p+1:02d}.png' if pages > 1 else f'samples_grid_{n}.png'
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=120)
        if not no_show:
            plt.show()
        plt.close(fig)
        saved_paths.append(out_path)
        start += this_n

    return saved_paths


def save_individual(images: torch.Tensor, goals: Optional[torch.Tensor], out_dir: str, num_samples: int,
                    goals_lat: Optional[torch.Tensor] = None,
                    goals_lon: Optional[torch.Tensor] = None):
    os.makedirs(out_dir, exist_ok=True)
    n = min(num_samples, images.shape[0])
    saved = []
    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy()
        # Build the label shown in the image title.
        title = None
        if goals_lat is not None and goals_lon is not None and i < goals_lat.shape[0] and i < goals_lon.shape[0]:
            li = int(goals_lat[i].item())
            lj = int(goals_lon[i].item())
            title = f"{LATERAL_INDEX.get(li, 'lat?')} | {LONGITUDINAL_INDEX.get(lj, 'lon?')}"
        elif goals is not None and i < goals.shape[0]:
            gi = int(goals[i].item())
            title = GOAL_INDEX.get(gi, 'unknown')
        plt.figure(figsize=(1.8, 1.8))
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.axis('off')
        # Sanitize characters that are invalid in Windows file names.
        if title:
            suffix = ''.join('-' if ch in '<>:"/\\|?*' else ch for ch in title).replace(' ', '_')
        else:
            suffix = 'unknown'
        out_path = os.path.join(out_dir, f"sample_{i:04d}_{suffix}.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        saved.append(out_path)
    return saved


def save_gif(images: torch.Tensor, goals: torch.Tensor, out_dir: str, gif_length: int, fps: int = 5) -> str:
    """Save an animated GIF in sample-index order."""
    os.makedirs(out_dir, exist_ok=True)
    n = min(gif_length, images.shape[0])
    frames: List[Image.Image] = []
    for i in range(n):
        # Convert the tensor to an 8-bit NumPy image.
        arr = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype('uint8')
        frame = Image.fromarray(arr)
        # Keep the frame image-only; labels remain available in the tensor files.
        frames.append(frame)
    gif_path = os.path.join(out_dir, f'sequence_{n}.gif')
    if len(frames) > 0:
        duration = int(1000 / max(fps, 1))  # ms per frame
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
    return gif_path


def _choose_backend(no_show: bool, backend_opt: str = "auto"):
    """Select a matplotlib backend before importing pyplot.

    Use an explicitly requested backend first. Otherwise select Agg for
    non-interactive execution and retain automatic selection when a display
    is available.
    """
    if backend_opt and backend_opt.lower() != "auto":
        matplotlib.use(backend_opt)
        return
    if no_show:
        matplotlib.use("Agg")
        return
    # Fall back to Agg on headless systems to avoid initializing TkAgg.
    if (os.environ.get("DISPLAY", "") == "") and (os.environ.get("MPLBACKEND", "") == ""):
        matplotlib.use("Agg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing images.pt and goals.pt')
    parser.add_argument('--num-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--grid-cols', type=int, default=4, help='Number of grid columns')
    parser.add_argument('--max-grid-tiles', type=int, default=200, help='Maximum number of tiles on one grid page')
    parser.add_argument('--out-subdir', type=str, default='samples', help='Subdirectory used for rendered images')
    parser.add_argument('--no-show', action='store_true', help='Do not open a matplotlib window')
    parser.add_argument('--save-individual', action='store_true', help='Save each sample as a separate image')
    parser.add_argument('--save-gif', action='store_true', help='Save an animated GIF in sample order')
    parser.add_argument('--gif-length', type=int, default=40, help='Maximum number of frames in the GIF')
    parser.add_argument('--gif-fps', type=int, default=5, help='GIF frame rate')
    parser.add_argument('--backend', type=str, default='auto', help='Matplotlib backend (auto, Agg, TkAgg, Qt5Agg, etc.)')
    args = parser.parse_args()

    # Select the backend before the first pyplot import.
    _choose_backend(args.no_show, args.backend)
    global plt  # Used by the rendering helpers above.
    import matplotlib.pyplot as plt

    tensors = load_tensors(args.data_dir, limit_images=args.num_samples)
    images = tensors['images']
    goals = tensors['goals']
    goals_lat = tensors['goals_lateral']
    goals_lon = tensors['goals_longitudinal']

    stats = summarize_all(goals, goals_lat, goals_lon)
    print("Label distribution:")
    if 'single' in stats:
        print("  [single]")
        for k, v in stats['single'].items():
            if k == 'total':
                print(f"    total: {v}")
            else:
                print(f"    {k:20s} count={v['count']:6d} ratio={v['ratio']:.4f}")
    if 'lateral' in stats:
        print("  [lateral]")
        for k, v in stats['lateral'].items():
            if k == 'total':
                print(f"    total: {v}")
            else:
                print(f"    {k:20s} count={v['count']:6d} ratio={v['ratio']:.4f}")
    if 'longitudinal' in stats:
        print("  [longitudinal]")
        for k, v in stats['longitudinal'].items():
            if k == 'total':
                print(f"    total: {v}")
            else:
                print(f"    {k:20s} count={v['count']:6d} ratio={v['ratio']:.4f}")
    if 'joint' in stats:
        print("  [joint counts] rows=lateral, cols=longitudinal")
        lat_names = stats['joint']['lateral_classes']
        lon_names = stats['joint']['longitudinal_classes']
        counts = stats['joint']['count']
        # Print a compact table header.
        header = '             ' + ' '.join([f"{n[:8]:>9s}" for n in lon_names])
        print(header)
        for i, row in enumerate(counts):
            print(f"  {lat_names[i][:11]:>11s} " + ' '.join([f"{c:9d}" for c in row]))

    # Save the distribution statistics as JSON.
    stats_path = os.path.join(args.data_dir, 'stats_visualization.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats JSON saved to {stats_path}")

    out_dir = os.path.join(args.data_dir, args.out_subdir)
    grid_paths = save_grid(images, goals, out_dir, args.num_samples, args.grid_cols, args.no_show, args.max_grid_tiles,
                           goals_lat=goals_lat, goals_lon=goals_lon)
    if len(grid_paths) == 1:
        print(f"Grid image saved to {grid_paths[0]}")
    elif len(grid_paths) > 1:
        print(f"Saved {len(grid_paths)} grid pages. First: {grid_paths[0]}  Last: {grid_paths[-1]}")

    if args.save_individual:
        files = save_individual(images, goals, out_dir, args.num_samples, goals_lat=goals_lat, goals_lon=goals_lon)
        print(f"Saved {len(files)} individual sample images to {out_dir}")

    if args.save_gif:
        gif_path = save_gif(images, goals, out_dir, args.gif_length, args.gif_fps)
        print(f"GIF saved to {gif_path} (frames={min(args.gif_length, images.shape[0])}, fps={args.gif_fps})")


if __name__ == '__main__':
    main()
