"""
Preprocess Waymo Motion Scenario TFRecords:
- Parse Scenario protocol buffers.
- Render each time step as an 84x84 SDC-centered bird's-eye-view image.
- Estimate high-level driving labels from SDC motion over a temporal window.
- Save images, single-task labels, and optional dual-task labels as tensors.

The official TensorFlow and waymo-open-dataset packages are supported. A
TensorFlow-free TFRecord reader and WAYMO_PROTO_DIR fallback are also provided
for environments in which the official Waymo package is unavailable.
"""
import os
import math
import argparse
import importlib.util
import struct
import sys
import types
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from PIL import Image, ImageDraw


# -------------------- Basic utilities --------------------

def _iter_tfrecord_records(path: str):
    """Yield serialized examples from an uncompressed TFRecord without TensorFlow."""
    with open(path, 'rb') as f:
        while True:
            length_bytes = f.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise RuntimeError(f"Truncated TFRecord length header in {path}")
            length = struct.unpack('<Q', length_bytes)[0]
            if len(f.read(4)) != 4:
                raise RuntimeError(f"Truncated TFRecord length checksum in {path}")
            payload = f.read(length)
            if len(payload) != length:
                raise RuntimeError(f"Truncated TFRecord payload in {path}")
            if len(f.read(4)) != 4:
                raise RuntimeError(f"Truncated TFRecord payload checksum in {path}")
            yield payload


def _load_scenario_pb2():
    """Load Waymo's Scenario protobuf from the official package or a proto directory."""
    try:
        from waymo_open_dataset.protos import scenario_pb2
        return scenario_pb2
    except Exception:
        pass

    try:
        from metadrive.utils.waymo_utils.protos import scenario_pb2
        return scenario_pb2
    except Exception:
        pass

    proto_dir = os.environ.get('WAYMO_PROTO_DIR')
    if not proto_dir:
        raise RuntimeError(
            "Missing Waymo Scenario protobuf. Install waymo-open-dataset or set "
            "WAYMO_PROTO_DIR to a directory containing map_pb2.py and scenario_pb2.py."
        )
    proto_dir = os.path.abspath(proto_dir)
    package_name = 'metadrive.utils.waymo_utils.protos'
    parent = ''
    for part in package_name.split('.'):
        parent = part if not parent else f'{parent}.{part}'
        if parent not in sys.modules:
            module = types.ModuleType(parent)
            module.__path__ = [proto_dir]
            sys.modules[parent] = module

    loaded = {}
    for name in ('map_pb2', 'scenario_pb2'):
        full_name = f'{package_name}.{name}'
        file_path = os.path.join(proto_dir, f'{name}.py')
        if not os.path.isfile(file_path):
            raise RuntimeError(f"Missing {file_path}")
        spec = importlib.util.spec_from_file_location(full_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load protobuf module: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded['scenario_pb2']

def _prepare_out_dir(out_dir: Optional[str]) -> str:
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    if os.path.splitext(out_dir)[1] in {'.pt', '.pth', '.pkl'}:
        out_dir = os.path.dirname(out_dir)
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise RuntimeError(f"Output path exists and is a file: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _expand_records(records: List[str]) -> List[str]:
    """Recursively expand directories and glob patterns into TFRecord paths."""
    import glob
    paths: List[str] = []
    for p in records:
        p = os.path.abspath(p)
        # Expand glob patterns when present.
        candidates = glob.glob(p) if any(ch in p for ch in ['*', '?', '[']) else [p]
        for gp in candidates:
            gp = os.path.abspath(gp)
            if os.path.isdir(gp):
                for root, _, files in os.walk(gp):
                    for f in files:
                        if f.endswith('.tfrecord') or '.tfrecord-' in f:
                            paths.append(os.path.join(root, f))
            else:
                if gp.endswith('.tfrecord') or '.tfrecord-' in gp:
                    paths.append(gp)
    paths = sorted(set(paths))
    return paths


def _parse_rgb(s: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Parse an R,G,B string, clamp values to [0, 255], and fall back on error."""
    try:
        parts = [int(x.strip()) for x in str(s).split(',')]
        if len(parts) != 3:
            return default
        r = max(0, min(255, parts[0]))
        g = max(0, min(255, parts[1]))
        b = max(0, min(255, parts[2]))
        return (r, g, b)
    except Exception:
        return default


# -------------------- Labels and coordinate transforms --------------------

def _label_from_window(states, t: int, window: int = 10, dt: float = 0.1,
                       no_valid_check: bool = False,
                       turn_thresh_deg: float = 12.0,
                       stop_speed_mps: float = 0.5) -> str:
    """Assign a label from cumulative yaw change and average speed.

    The window covers [t-window+1, t]. Waymo Motion frames are approximately
    0.1 seconds apart, so a window of 10 spans about one second.
    - turn: absolute cumulative yaw exceeds turn_thresh_deg
    - stop: average speed is below stop_speed_mps
    - otherwise: lane_keep
    """
    n = len(states)
    if n == 0:
        return 'lane_keep'
    start = max(0, t - max(1, window) + 1)
    end = min(t, n - 1)
    if end <= start:
        return 'lane_keep'

    cum_dyaw = 0.0
    dist = 0.0
    pairs = 0
    for i in range(start + 1, end + 1):
        a = states[i - 1]
        b = states[i]
        if not no_valid_check and (not getattr(a, 'valid', False) or not getattr(b, 'valid', False)):
            continue
        dyaw = (b.heading - a.heading)
        # wrap to [-pi, pi]
        dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
        cum_dyaw += dyaw
        dist += math.hypot(b.center_x - a.center_x, b.center_y - a.center_y)
        pairs += 1

    if pairs == 0:
        return 'lane_keep'
    yaw_deg = abs(cum_dyaw) * 180.0 / math.pi
    avg_speed = dist / (pairs * max(dt, 1e-3))

    if avg_speed < stop_speed_mps:
        return 'stop'
    if yaw_deg > turn_thresh_deg:
        # Positive cumulative yaw denotes a left turn in this coordinate system.
        return 'turn_left' if cum_dyaw > 0 else 'turn_right'
    return 'lane_keep'


def _world_to_vehicle(x: float, y: float, cx: float, cy: float, heading: float) -> Tuple[float, float]:
    """Transform world coordinates into the SDC frame: x forward, y left."""
    dx = x - cx
    dy = y - cy
    # Translate to the SDC origin, then rotate by the negative heading.
    c = math.cos(-heading)
    s = math.sin(-heading)
    vx = c * dx - s * dy
    vy = s * dx + c * dy
    return vx, vy


def _meters_to_pixels(vx: float, vy: float, m_per_px: float, H: int, W: int) -> Tuple[int, int]:
    """Convert vehicle-frame meters to pixels, with forward pointing upward."""
    cx, cy = W // 2, H // 2
    u = int(round(vx / m_per_px)) + cx
    v = int(round(-vy / m_per_px)) + cy
    return u, v


# -------------------- Rendering --------------------

def _draw_rotated_box(draw: ImageDraw.ImageDraw, u: int, v: int, length_m: float, width_m: float,
                      heading_rel: float, m_per_px: float, color: int = 255):
    """Draw a vehicle as a rotated rectangle centered at pixel (u, v)."""
    Lm = max(0.5, float(length_m))
    Wm = max(0.3, float(width_m))
    hl = 0.5 * Lm
    hw = 0.5 * Wm
    c = math.cos(heading_rel)
    s = math.sin(heading_rel)
    # Rectangle corners in the vehicle frame: x forward and y left.
    corners = [
        (+hl, +hw),
        (+hl, -hw),
        (-hl, -hw),
        (-hl, +hw),
    ]
    poly = []
    for (dx, dy) in corners:
        # Rotate by the heading relative to the SDC.
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        # Convert meters to pixels while accounting for downward image y.
        pu = int(round(rx / m_per_px)) + u
        pv = int(round(-ry / m_per_px)) + v
        poly.append((pu, pv))
    draw.polygon(poly, fill=color)


def _render_bev(sdc_state: dict, agents: List[dict], H: int = 84, W: int = 84, m_per_px: float = 0.5,
                map_polylines: Optional[List[List[Tuple[float, float]]]] = None,
                map_color=(180, 180, 180),
                map_line_width: int = 2,
                map_max_range: float = 60.0,
                colorize: bool = True,
                background_color: Tuple[int, int, int] = (255, 255, 255),
                agent_color=(64, 160, 255),
                sdc_color=(255, 64, 64),
                sdc_history: Optional[List[dict]] = None) -> Image.Image:
    """Render an SDC-centered bird's-eye-view image.

    With colorize enabled, the map, neighboring agents, and SDC use distinct
    colors. Otherwise a backward-compatible grayscale image is produced.
    Optional SDC history states provide motion trails.
    """
    if colorize:
        img = Image.new('RGB', (W, H), background_color)
    else:
        img = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(img)

    cx, cy = float(sdc_state.get('center_x', 0.0)), float(sdc_state.get('center_y', 0.0))
    hdg = float(sdc_state.get('heading', 0.0))

    # Render optional map polylines.
    if map_polylines:
        for pl in map_polylines:
            if not pl or len(pl) < 2:
                continue
            pts = []
            for (mx, my) in pl:
                # Clip map points by their world-coordinate distance from the SDC.
                if (mx - cx) ** 2 + (my - cy) ** 2 > (map_max_range ** 2):
                    continue
                vx, vy = _world_to_vehicle(mx, my, cx, cy, hdg)
                u, v = _meters_to_pixels(vx, vy, m_per_px, H, W)
                pts.append((u, v))
            if len(pts) >= 2:
                draw_color = map_color if colorize else 80
                w = max(1, int(map_line_width))
                draw.line(pts, fill=draw_color, width=w)

    # Render SDC history using a color gradient.
    if sdc_history:
        for h_state in sdc_history:
            if not h_state.get('valid', False):
                continue
            hx, hy = float(h_state.get('center_x', 0.0)), float(h_state.get('center_y', 0.0))
            vx, vy = _world_to_vehicle(hx, hy, cx, cy, hdg)
            u, v = _meters_to_pixels(vx, vy, m_per_px, H, W)
            if 0 <= u < W and 0 <= v < H:
                # Map temporal offsets from recent red/purple to older blue tones.
                dt = h_state.get('dt', 5)
                # Apply a simple red-to-blue interpolation.
                # ratio 0.0 (dt=0) -> 1.0 (dt=24)
                ratio = min(1.0, dt / 24.0) 
                if colorize:
                    # R: 255 -> 100, G: 64 -> 100, B: 64 -> 255
                    r = int(255 * (1 - ratio) + 100 * ratio)
                    g = int(64 * (1 - ratio) + 100 * ratio)
                    b = int(64 * (1 - ratio) + 255 * ratio)
                    hist_col = (r, g, b)
                else:
                    hist_col = 150
                
                # Slightly shrink older history boxes.
                scale = 1.0 - (dt * 0.015) # 0.94 ~ 0.7
                _draw_rotated_box(draw, u, v, h_state.get('length', 4.6) * scale, h_state.get('width', 1.9) * scale,
                                  float(h_state.get('heading', 0.0)) - hdg,
                                  m_per_px, color=hist_col)

    # Render neighboring agents.
    for a in agents:
        if not a.get('valid', False):
            continue
        ax, ay = float(a.get('center_x', 0.0)), float(a.get('center_y', 0.0))
        vx, vy = _world_to_vehicle(ax, ay, cx, cy, hdg)
        u, v = _meters_to_pixels(vx, vy, m_per_px, H, W)
        if 0 <= u < W and 0 <= v < H:
            draw_col = agent_color if colorize else 200
            _draw_rotated_box(draw, u, v, a.get('length', 4.0), a.get('width', 1.8),
                              float(a.get('heading', 0.0)) - hdg,
                              m_per_px, color=draw_col)

    # Render the SDC.
    draw_col = sdc_color if colorize else 255
    _draw_rotated_box(draw, W // 2, H // 2, sdc_state.get('length', 4.6), sdc_state.get('width', 1.9), 0.0, m_per_px, color=draw_col)

    # Return an RGB image for a consistent downstream tensor shape.
    return img if colorize else img.convert('RGB')


# -------------------- Scenario parsing and sample generation --------------------

def _detect_overtake(
    sc,
    sdc_idx: int,
    t: int,
    window: int = 10,
    dt: float = 0.1,
    no_valid_check: bool = False,
    lateral_thr: float = 2.0,
    min_cross_dist: float = 5.0,
    min_rel_speed: float = 1.0,
) -> bool:
    """Detect overtaking from a longitudinal position crossing within a window.

    A candidate agent must move from ahead of the SDC to behind it, remain
    within the lateral same-lane threshold, exceed the minimum crossing
    distance and relative speed, and have valid states unless validation is
    explicitly disabled.
    """
    try:
        sdc = sc.tracks[sdc_idx]
    except Exception:
        return False
    n = len(sdc.states)
    if n == 0:
        return False
    start = max(0, t - max(1, window) + 1)
    end = min(t, n - 1)
    if end <= start:
        return False

    s0 = sdc.states[start]
    s1 = sdc.states[end]
    if (not no_valid_check) and (not getattr(s0, 'valid', False) or not getattr(s1, 'valid', False)):
        return False

    cx0, cy0, hdg0 = float(s0.center_x), float(s0.center_y), float(s0.heading)
    cx1, cy1, hdg1 = float(s1.center_x), float(s1.center_y), float(s1.heading)
    pairs = (end - start)
    elapsed = pairs * max(dt, 1e-3)
    if pairs <= 0:
        return False

    for k, tr in enumerate(sc.tracks):
        if k == sdc_idx:
            continue
        if end >= len(tr.states):
            continue
        a0 = tr.states[start]
        a1 = tr.states[end]
        if (not no_valid_check) and (not getattr(a0, 'valid', False) or not getattr(a1, 'valid', False)):
            continue
        ax0, ay0 = float(a0.center_x), float(a0.center_y)
        ax1, ay1 = float(a1.center_x), float(a1.center_y)

        # Project positions into the SDC frame at each window endpoint.
        long0, lat0 = _world_to_vehicle(ax0, ay0, cx0, cy0, hdg0)
        long1, lat1 = _world_to_vehicle(ax1, ay1, cx1, cy1, hdg1)

        if abs(lat0) > lateral_thr or abs(lat1) > lateral_thr:
            continue

        # Only consider agents that move from ahead to behind the SDC.
        if long0 > 0.0 and long1 < 0.0:
            cross = (long0 - long1)
            rel_speed = cross / elapsed
            if cross >= min_cross_dist and rel_speed >= min_rel_speed:
                return True

    return False

def preprocess_motion(
    records: List[str],
    out_dir: Optional[str] = None,
    limit_samples: Optional[int] = None,
    step_stride: int = 1,
    debug: bool = False,
    no_valid_check: bool = False,
    label_window: int = 10,
    turn_thresh_deg: float = 12.0,
    stop_speed_mps: float = 0.5,
    draw_map: bool = False,
    map_max_range: float = 60.0,
    detect_overtake: bool = True,
    overtake_lateral_thr: float = 2.0,
    overtake_min_cross: float = 5.0,
    overtake_min_rel_speed: float = 1.0,
    colorize: bool = True,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    map_color: Tuple[int, int, int] = (180, 180, 180),
    map_line_width: int = 2,
    dual_label: bool = False,
    lane_change_lat_thresh: float = 1.5,
    accel_thresh: float = 0.25,
    decel_thresh: float = 0.25,
    # approaching/proximity rule thresholds (exposed to CLI)
    rel_speed_thresh: float = 1.0,
    max_behind_dist: float = 30.0,
    lat_min: float = 1.0,
    max_ahead_dist: float = 30.0,
    close_dist: float = 12.0,
    lateral_min: float = 1.0,
    progress: bool = False,
    quiet: bool = False,
) -> Dict[str, torch.Tensor]:
    """Render Waymo Motion scenarios as images and supervision labels.

    When dual_label is enabled, save lateral labels to goals_lateral.pt and
    longitudinal labels to goals_longitudinal.pt. The backward-compatible
    single-task labels remain available in goals.pt. Returns all saved tensors.
    """
    try:
        import tensorflow as tf
    except Exception:
        tf = None
    scenario_pb2 = _load_scenario_pb2()

    out_dir = _prepare_out_dir(out_dir)
    paths = _expand_records(records)
    print(f"Found {len(paths)} TFRecord files for Motion preprocessing.")
    if len(paths) == 0:
        raise RuntimeError("No TFRecord files found in given --records paths.")

    images: List[torch.Tensor] = []
    # Backward-compatible single-task labels.
    goals_single: List[int] = []
    goal_to_idx_single = {"lane_keep": 0, "turn_left": 1, "turn_right": 2, "stop": 3, "overtake": 4}
    # Dual-task label mappings.
    lateral_classes = ["lane_keep", "lane_change_left", "lane_change_right", "turn_left", "turn_right", "stop", "overtake"]
    lateral_to_idx = {k: i for i, k in enumerate(lateral_classes)}
    longitudinal_classes = ["accelerate", "decelerate", "cruise", "stop"]
    longitudinal_to_idx = {k: i for i, k in enumerate(longitudinal_classes)}
    goals_lateral: List[int] = []
    goals_longitudinal: List[int] = []
    # approaching flags
    approaching_left: List[int] = []
    approaching_right: List[int] = []
    # proximity flags (forward sectors)
    close_ahead: List[int] = []
    close_front_left: List[int] = []
    close_front_right: List[int] = []

    # Global diagnostic counters.
    sc_total = 0
    sc_with_sdc = 0
    frames_total = 0
    frames_used = 0
    skip_invalid = 0
    skip_short = 0
    skip_errors = 0

    produced = 0

    # Accumulate samples and save a single images.pt file at the end.

    # Print one summary after each input file instead of per-frame progress.

    # Per-file diagnostics.
    per_file_stats = []

    for p in paths:
        # Suppress start messages and report only after each file completes.
        ds = tf.data.TFRecordDataset(p, compression_type='') if tf is not None else _iter_tfrecord_records(p)
        # File-level counters.
        file_frames_total = 0
        file_frames_used = 0
        file_skip_invalid = 0
        file_skip_short = 0
        file_skip_errors = 0
        for raw in ds:
            try:
                sc = scenario_pb2.Scenario()
                serialized = bytes(raw.numpy()) if hasattr(raw, 'numpy') else raw
                sc.ParseFromString(serialized)
                # print("=======", sc)
                sc_total += 1
                sdc_idx = sc.sdc_track_index
                # print("=======", sc.tracks)
                if sdc_idx < 0 or sdc_idx >= len(sc.tracks):
                    continue
                sc_with_sdc += 1
                sdc = sc.tracks[sdc_idx]
                # print("======:", sdc)
                T = len(sdc.states)
                if T < 2:
                    skip_short += 1
                    file_skip_short += 1
                    continue
                # Parse optional map polylines once per scenario.
                map_polylines = None
                if draw_map and hasattr(sc, 'map_features'):
                    pls: List[List[Tuple[float, float]]] = []
                    for mf in sc.map_features:
                        try:
                            # Lane centerline or polyline variants
                            if hasattr(mf, 'lane') and mf.lane is not None:
                                poly = []
                                # Support both centerline and polyline field names.
                                if hasattr(mf.lane, 'centerline') and mf.lane.centerline:
                                    for pt in mf.lane.centerline:
                                        poly.append((pt.x, pt.y))
                                elif hasattr(mf.lane, 'polyline') and mf.lane.polyline:
                                    for pt in mf.lane.polyline:
                                        poly.append((pt.x, pt.y))
                                if len(poly) >= 2:
                                    pls.append(poly)
                            # Road lines/edges
                            if hasattr(mf, 'road_line') and mf.road_line is not None and hasattr(mf.road_line, 'polyline'):
                                poly = [(pt.x, pt.y) for pt in mf.road_line.polyline]
                                if len(poly) >= 2:
                                    pls.append(poly)
                            if hasattr(mf, 'road_edge') and mf.road_edge is not None and hasattr(mf.road_edge, 'polyline'):
                                poly = [(pt.x, pt.y) for pt in mf.road_edge.polyline]
                                if len(poly) >= 2:
                                    pls.append(poly)
                        except Exception:
                            # Ignore map features that cannot be parsed.
                            continue
                    map_polylines = pls

                # Collect neighboring-agent states at each time step.
                for t in range(1, T):
                    frames_total += 1
                    file_frames_total += 1
                    if t % step_stride != 0:
                        continue
                    cur = sdc.states[t]
                    # print("cur======:", cur.center_x, cur.center_y, cur.heading, cur.valid)
                    prev = sdc.states[t - 1]
                    # print("prev======:", prev)
                    
                    if not no_valid_check:
                        if not (cur.valid and prev.valid):
                            skip_invalid += 1
                            file_skip_invalid += 1
                            # Quiet mode suppresses per-frame output.
                            continue
                    # Assemble the current SDC state.
                    cur_s = {
                        'center_x': cur.center_x,
                        'center_y': cur.center_y,
                        'heading': cur.heading,
                        'length': cur.length,
                        'width': cur.width,
                        'valid': cur.valid,
                    }
                    prev_s = {
                        'center_x': prev.center_x,
                        'center_y': prev.center_y,
                        'heading': prev.heading,
                        'length': prev.length,
                        'width': prev.width,
                        'valid': prev.valid,
                    }
                    # print("prev_s====:", prev_s)
                    # Align neighboring agents to the same time step.
                    agents_t: List[dict] = []
                    for k, tr in enumerate(sc.tracks):
                        if k == sdc_idx:
                            continue
                        if t >= len(tr.states):
                            continue
                        st = tr.states[t]
                        if not st.valid:
                            continue
                        agents_t.append({
                            'center_x': st.center_x,
                            'center_y': st.center_y,
                            'heading': st.heading,
                            'length': st.length,
                            'width': st.width,
                            'valid': st.valid,
                        })

                    # Use temporal motion features for dual-task labels.
                    # 1) Backward-compatible single-task label.
                    goal_name_single = _label_from_window(
                        sdc.states,
                        t,
                        window=label_window,
                        dt=0.1,
                        no_valid_check=no_valid_check,
                        turn_thresh_deg=turn_thresh_deg,
                        stop_speed_mps=stop_speed_mps,
                    )
                    if detect_overtake and goal_name_single == 'lane_keep':
                        if _detect_overtake(
                            sc,
                            sdc_idx,
                            t,
                            window=label_window,
                            dt=0.1,
                            no_valid_check=no_valid_check,
                            lateral_thr=overtake_lateral_thr,
                            min_cross_dist=overtake_min_cross,
                            min_rel_speed=overtake_min_rel_speed,
                        ):
                            goal_name_single = 'overtake'

                    # 2) Lateral and longitudinal labels.
                    if dual_label:
                        start = max(0, t - max(1, label_window) + 1)
                        end = t
                        
                        # Collect speeds over the fitting window.
                        window_speeds = []
                        window_times = []
                        
                        dist_sum = 0.0
                        valid_pairs = 0
                        
                        for i_w in range(start + 1, end + 1):
                            a = sdc.states[i_w - 1]
                            b = sdc.states[i_w]
                            if (not no_valid_check) and (not a.valid or not b.valid):
                                continue
                            d = math.hypot(b.center_x - a.center_x, b.center_y - a.center_y)
                            dist_sum += d
                            sp = d / 0.1
                            
                            window_speeds.append(sp)
                            window_times.append((i_w - start) * 0.1)  # Relative time.
                            
                            valid_pairs += 1
                        
                        avg_speed = dist_sum / (valid_pairs * 0.1) if valid_pairs > 0 else 0.0
                        
                        # Estimate acceleration with a linear fit.
                        if valid_pairs >= 3:
                            # Require at least three samples for fitting.
                            try:
                                slope, intercept = np.polyfit(window_times, window_speeds, 1)
                                accel = slope
                            except Exception:
                                accel = 0.0
                        elif valid_pairs >= 1:
                            # Fall back to an endpoint difference for short windows.
                             accel = (window_speeds[-1] - window_speeds[0]) / max((valid_pairs) * 0.1, 1e-3)
                        else:
                             accel = 0.0

                        # Assign the longitudinal class.
                        if avg_speed < stop_speed_mps:
                            longitudinal_label = 'stop'
                        elif accel > accel_thresh:
                            longitudinal_label = 'accelerate'
                        elif accel < -decel_thresh:
                            longitudinal_label = 'decelerate'
                        else:
                            longitudinal_label = 'cruise'
                        # Assign the lateral class.
                        lateral_label = 'lane_keep'
                        if longitudinal_label == 'stop':
                            lateral_label = 'stop'
                        else:
                            # Compute cumulative yaw and lateral displacement.
                            cum_dyaw = 0.0
                            a0 = sdc.states[start]
                            a1 = sdc.states[end]
                            for i_w in range(start + 1, end + 1):
                                a_prev = sdc.states[i_w - 1]
                                a_cur = sdc.states[i_w]
                                if (not no_valid_check) and (not a_prev.valid or not a_cur.valid):
                                    continue
                                dyaw = (a_cur.heading - a_prev.heading + math.pi) % (2 * math.pi) - math.pi
                                cum_dyaw += dyaw
                            yaw_deg = abs(cum_dyaw) * 180.0 / math.pi
                            # Measure net lateral displacement in the initial SDC frame.
                            dx = a1.center_x - a0.center_x
                            dy = a1.center_y - a0.center_y
                            c = math.cos(-a0.heading)
                            s_ = math.sin(-a0.heading)
                            x_f = c * dx - s_ * dy  # Longitudinal component.
                            y_f = s_ * dx + c * dy  # Lateral component, positive left.
                            if yaw_deg > turn_thresh_deg:
                                lateral_label = 'turn_left' if cum_dyaw > 0 else 'turn_right'
                            else:
                                if abs(y_f) > lane_change_lat_thresh:
                                    lateral_label = 'lane_change_left' if y_f > 0 else 'lane_change_right'
                            if detect_overtake and lateral_label in ('lane_keep', 'lane_change_left', 'lane_change_right'):
                                if _detect_overtake(
                                    sc,
                                    sdc_idx,
                                    t,
                                    window=label_window,
                                    dt=0.1,
                                    no_valid_check=no_valid_check,
                                    lateral_thr=overtake_lateral_thr,
                                    min_cross_dist=overtake_min_cross,
                                    min_rel_speed=overtake_min_rel_speed,
                                ):
                                    lateral_label = 'overtake'
                        # Compute approaching flags for left/right rear using current and previous agent states
                        # Criteria:
                        # - Agent is behind (long < -1 m)
                        # - Lateral offset indicates left (lat > 1.0) or right (lat < -1.0)
                        # - Relative longitudinal approach speed > 1.0 m/s (closing in)
                        # - Consider agents within 30 m behind
                        approach_left_flag = 0
                        approach_right_flag = 0
                        dt = 0.1
                        for k, tr in enumerate(sc.tracks):
                            if k == sdc_idx:
                                continue
                            if t >= len(tr.states) or (t - 1) < 0 or (t - 1) >= len(tr.states):
                                continue
                            a_prev = tr.states[t - 1]
                            a_cur = tr.states[t]
                            if (not no_valid_check) and (not a_prev.valid or not a_cur.valid):
                                continue
                            # world -> vehicle coords at prev and cur using respective sdc poses
                            long_prev, lat_prev = _world_to_vehicle(a_prev.center_x, a_prev.center_y, cur.center_x, cur.center_y, cur.heading)
                            long_cur, lat_cur = _world_to_vehicle(a_cur.center_x, a_cur.center_y, cur.center_x, cur.center_y, cur.heading)
                            # only consider agents behind
                            if long_cur >= -1.0 or abs(lat_cur) < 0.5:
                                continue
                            if abs(long_cur) > max_behind_dist:
                                continue
                            rel_long_speed = (long_prev - long_cur) / dt
                            if rel_long_speed > rel_speed_thresh:
                                # approaching from right rear (lat negative)
                                if lat_cur < -lat_min:
                                    approach_right_flag = 1
                                # approaching from left rear (lat positive)
                                if lat_cur > lat_min:
                                    approach_left_flag = 1
                                # if both set, can break early
                                if approach_left_flag and approach_right_flag:
                                    break

                        # Compute proximity flags for forward sectors
                        ahead_flag = 0
                        front_left_flag = 0
                        front_right_flag = 0
                        # thresholds are provided by function args
                        for k, tr in enumerate(sc.tracks):
                            if k == sdc_idx:
                                continue
                            if t >= len(tr.states):
                                continue
                            a = tr.states[t]
                            if (not no_valid_check) and (not a.valid):
                                continue
                            long_a, lat_a = _world_to_vehicle(a.center_x, a.center_y, cur.center_x, cur.center_y, cur.heading)
                            dist = math.hypot(long_a, lat_a)
                            # ahead sector: in front (long>0) and within close_dist
                            if 0.0 < long_a <= max_ahead_dist and abs(lat_a) <= lateral_min and dist <= close_dist:
                                ahead_flag = 1
                            # front-left: front and left lateral offset
                            if 0.0 < long_a <= max_ahead_dist and lat_a > lateral_min and dist <= close_dist:
                                front_left_flag = 1
                            # front-right: front and right lateral offset
                            if 0.0 < long_a <= max_ahead_dist and lat_a < -lateral_min and dist <= close_dist:
                                front_right_flag = 1
                            if ahead_flag and front_left_flag and front_right_flag:
                                break

                        goals_lateral.append(lateral_to_idx[lateral_label])
                        goals_longitudinal.append(longitudinal_to_idx[longitudinal_label])
                        approaching_left.append(int(approach_left_flag))
                        approaching_right.append(int(approach_right_flag))
                        close_ahead.append(int(ahead_flag))
                        close_front_left.append(int(front_left_flag))
                        close_front_right.append(int(front_right_flag))

                    # Always render the current sample; sharded resume is not used.

                    # Collect history at t-4, t-8, ..., t-20 for motion trails.
                    sdc_history = []
                    for dt_hist in range(4, 21, 4):
                        t_hist = t - dt_hist
                        if t_hist >= 0:
                            st_h = sdc.states[t_hist]
                            if st_h.valid:
                                sdc_history.append({
                                    'center_x': st_h.center_x,
                                    'center_y': st_h.center_y,
                                    'heading': st_h.heading,
                                    'length': st_h.length,
                                    'width': st_h.width,
                                    'valid': st_h.valid,
                                    'dt': dt_hist,
                                })

                    # Render an 84x84 image at 0.5 meters per pixel.
                    img = _render_bev(
                        cur_s,
                        agents_t,
                        H=84,
                        W=84,
                        m_per_px=0.5,
                        map_polylines=map_polylines if draw_map else None,
                        map_color=map_color,
                        map_line_width=map_line_width,
                        map_max_range=map_max_range,
                        colorize=colorize,
                        background_color=bg_color,
                        agent_color=(64,160,255),
                        sdc_color=(255,64,64),
                        sdc_history=sdc_history,
                    )
                    arr = np.array(img)
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    img_t = torch.from_numpy(arr).permute(2, 0, 1).to(torch.uint8)

                    images.append(img_t)
                    goals_single.append(goal_to_idx_single[goal_name_single])
                    frames_used += 1
                    file_frames_used += 1
                    produced += 1

                    if limit_samples is not None and produced >= limit_samples:
                        raise StopIteration
                    # Online sharding is intentionally disabled.
            except StopIteration:
                break
            except Exception as e:
                # Skip malformed samples while retaining diagnostic counts.
                skip_errors += 1
                file_skip_errors += 1
                if debug and not quiet:
                    print(f"[WARN] Skip scenario due to error: {type(e).__name__}: {e}")
                continue
        # Store per-file diagnostics.
        per_file_stats.append({
            'file': p,
            'frames_total': file_frames_total,
            'frames_used': file_frames_used,
            'skip_invalid': file_skip_invalid,
            'skip_short': file_skip_short,
            'skip_errors': file_skip_errors,
        })
        # Print one compact summary after each file completes.
        try:
            print(f"[FILE DONE] {os.path.basename(p)} total={file_frames_total} used={file_frames_used} invalid={file_skip_invalid} short={file_skip_short} errors={file_skip_errors}")
        except Exception:
            pass
        if limit_samples is not None and produced >= limit_samples:
            break

    if debug and not quiet:
        print(
            f"[DEBUG] scenarios={sc_total}, with_sdc={sc_with_sdc}, "
            f"frames_total={frames_total}, frames_used={frames_used}, "
            f"skip_invalid={skip_invalid}, skip_short={skip_short}, skip_errors={skip_errors}"
        )
        if len(paths) > 10:
            print(f"[DEBUG] first files: {paths[:5]} ... last files: {paths[-5:]}")
    if len(images) == 0:
        raise RuntimeError(
            "No samples produced. Possible reasons:\n"
            "- Passed an empty/incorrect --records path (try pointing to .../waymo_data/train and enable --debug)\n"
            "- All frames filtered by 'valid' flag (try --no-valid-check and --stride 1)\n"
            "- Dependency mismatch: ensure 'waymo-open-dataset' matches your TensorFlow version\n"
            "  e.g. pip install waymo-open-dataset-tf-2-12-0 (or tf-2-11-0) to match your TF."
        )

    # Stack all 84x84 images and save one tensor file.
    images_tensor = torch.stack(images)
    torch.save(images_tensor, os.path.join(out_dir, 'images.pt'))
    # Also save NPY for memory-mapped training with a lower memory peak.
    try:
        import numpy as _np
        _np.save(os.path.join(out_dir, 'images.npy'), images_tensor.cpu().numpy())
    except Exception:
        pass
    out_dict: Dict[str, torch.Tensor] = {"images": images_tensor}
    # Save backward-compatible single-task labels.
    goals_tensor = torch.tensor(goals_single, dtype=torch.long)
    torch.save(goals_tensor, os.path.join(out_dir, 'goals.pt'))
    out_dict['goals'] = goals_tensor
    if dual_label:
        lat_t = torch.tensor(goals_lateral, dtype=torch.long)
        lon_t = torch.tensor(goals_longitudinal, dtype=torch.long)
        torch.save(lat_t, os.path.join(out_dir, 'goals_lateral.pt'))
        torch.save(lon_t, os.path.join(out_dir, 'goals_longitudinal.pt'))
        out_dict['goals_lateral'] = lat_t
        out_dict['goals_longitudinal'] = lon_t
        # Save approaching flags (boolean tensors)
        app_l_t = torch.tensor(approaching_left, dtype=torch.uint8)
        app_r_t = torch.tensor(approaching_right, dtype=torch.uint8)
        torch.save(app_l_t, os.path.join(out_dir, 'approaching_left_rear.pt'))
        torch.save(app_r_t, os.path.join(out_dir, 'approaching_right_rear.pt'))
        out_dict['approaching_left_rear'] = app_l_t
        out_dict['approaching_right_rear'] = app_r_t
        # Save proximity flags
        ca_t = torch.tensor(close_ahead, dtype=torch.uint8)
        cfl_t = torch.tensor(close_front_left, dtype=torch.uint8)
        cfr_t = torch.tensor(close_front_right, dtype=torch.uint8)
        torch.save(ca_t, os.path.join(out_dir, 'close_ahead.pt'))
        torch.save(cfl_t, os.path.join(out_dir, 'close_front_left.pt'))
        torch.save(cfr_t, os.path.join(out_dir, 'close_front_right.pt'))
        out_dict['close_ahead'] = ca_t
        out_dict['close_front_left'] = cfl_t
        out_dict['close_front_right'] = cfr_t

    # Save preprocessing diagnostics and reproducibility metadata.
    debug_summary = {
        'scenarios': sc_total,
        'with_sdc': sc_with_sdc,
        'frames_total': frames_total,
        'frames_used': frames_used,
        'skip_invalid': skip_invalid,
        'skip_short': skip_short,
        'skip_errors': skip_errors,
        'per_file': per_file_stats[:50],  # Bound metadata size.
    }
    # Record all thresholds used by this run.
    debug_summary['thresholds'] = {
        'label_window': label_window,
        'turn_thresh_deg': turn_thresh_deg,
        'stop_speed_mps': stop_speed_mps,
        'overtake_lateral_thr': overtake_lateral_thr,
        'overtake_min_cross': overtake_min_cross,
        'overtake_min_rel_speed': overtake_min_rel_speed,
        'lane_change_lat_thresh': lane_change_lat_thresh,
        'accel_thresh': accel_thresh,
        'decel_thresh': decel_thresh,
        'rel_speed_thresh': rel_speed_thresh,
        'max_behind_dist': max_behind_dist,
        'lat_min': lat_min,
        'max_ahead_dist': max_ahead_dist,
        'close_dist': close_dist,
        'lateral_min': lateral_min,
    }
    if not quiet:
        with open(os.path.join(out_dir, 'preprocess_debug_summary.json'), 'w') as f:
            import json as _json
            _json.dump(debug_summary, f, indent=2)

    if not quiet:
        print(f"Saved samples to {out_dir} (dual_label={dual_label})")
    return out_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=str, nargs='+', required=True, help='TFRecord files or directories (Motion Scenario)')
    parser.add_argument('--out', type=str, default=None, help='Output directory (default: planner1/data/processed)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to produce')
    parser.add_argument('--stride', type=int, default=1, help='Temporal stride for sampling states (default 1)')
    parser.add_argument('--debug', action='store_true', help='Print debug counters')
    parser.add_argument('--no-valid-check', action='store_true', help='Ignore per-state valid flag filtering')
    parser.add_argument('--label-window', type=int, default=10, help='Label window size in frames (for example, 10 is about 1 s)')
    parser.add_argument('--turn-threshold-deg', type=float, default=12.0, help='Threshold of cumulative yaw (deg) to classify turn')
    parser.add_argument('--stop-speed', type=float, default=0.5, help='Average speed threshold (m/s) to classify stop')
    parser.add_argument('--draw-map', action='store_true', help='Render map features (lanes/road lines/edges)')
    parser.add_argument('--map-max-range', type=float, default=60.0, help='Map rendering range around SDC (meters)')
    parser.add_argument('--detect-overtake', action='store_true', help='Enable overtaking detection within label window')
    parser.add_argument('--overtake-lateral-thr', type=float, default=2.0, help='Lateral threshold (m) to consider same lane for overtake')
    parser.add_argument('--overtake-min-cross', type=float, default=5.0, help='Minimum longitudinal cross distance (m) to count as overtake')
    parser.add_argument('--overtake-min-rel-speed', type=float, default=1.0, help='Minimum relative longitudinal speed (m/s) to count as overtake')
    parser.add_argument('--background-color', type=str, default='255,255,255', help='Background RGB, e.g. 255,255,255 (default white)')
    parser.add_argument('--map-color', type=str, default='180,180,180', help='Map polyline RGB color')
    parser.add_argument('--map-line-width', type=int, default=2, help='Map polyline width (pixels)')
    parser.add_argument('--dual-label', action='store_true', help='Save separate lateral and longitudinal label tensors')
    parser.add_argument('--lane-change-lat', type=float, default=1.5, help='Net lateral displacement threshold for lane changes (m)')
    parser.add_argument('--accel-thresh', type=float, default=0.8, help='Average acceleration threshold for accelerate (m/s^2)')
    parser.add_argument('--decel-thresh', type=float, default=0.8, help='Absolute average deceleration threshold for decelerate (m/s^2)')
    # Approaching / proximity rule parameters
    parser.add_argument('--approach-rel-speed-thresh', type=float, default=1.0, help='Rear-approach relative longitudinal speed threshold (m/s)')
    parser.add_argument('--approach-max-behind-dist', type=float, default=30.0, help='Maximum rear distance considered by approach detection (m)')
    parser.add_argument('--approach-lat-min', type=float, default=1.0, help='Minimum lateral offset for left/right rear approach (m)')
    parser.add_argument('--proximity-max-ahead-dist', type=float, default=30.0, help='Maximum forward distance considered by proximity detection (m)')
    parser.add_argument('--proximity-close-dist', type=float, default=12.0, help='Euclidean distance threshold for close proximity (m)')
    parser.add_argument('--proximity-lateral-min', type=float, default=1.0, help='Minimum lateral offset for front-left/right sectors (m)')
    # Save a single images.pt file; sharded resume is not supported.
    try:
        # Python 3.9+ supports BooleanOptionalAction for --colorize/--no-colorize
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
        parser.add_argument('--colorize', action=bool_action, default=True, help='Render RGB with distinct colors for ego/agents/map')
        parser.add_argument('--progress', action=bool_action, default=False, help='Reserved compatibility option; per-file summaries are always used')
        parser.add_argument('--quiet', action=bool_action, default=False, help='Suppress detailed diagnostic output')
    except Exception:
        parser.add_argument('--colorize', action='store_true', default=True, help='Render RGB with distinct colors for ego/agents/map (set default True)')
        parser.add_argument('--progress', action='store_true', default=False, help='Reserved compatibility option; per-file summaries are always used')
        parser.add_argument('--quiet', action='store_true', default=False, help='Suppress detailed diagnostic output')
    # Sharded resume options have been removed.
    args = parser.parse_args()

    preprocess_motion(
        args.records,
        out_dir=args.out,
        limit_samples=args.limit,
        step_stride=args.stride,
        debug=args.debug,
        no_valid_check=args.no_valid_check,
        label_window=args.label_window,
        turn_thresh_deg=args.turn_threshold_deg,
        stop_speed_mps=args.stop_speed,
        draw_map=args.draw_map,
        map_max_range=args.map_max_range,
        detect_overtake=args.detect_overtake,
        overtake_lateral_thr=args.overtake_lateral_thr,
        overtake_min_cross=args.overtake_min_cross,
        overtake_min_rel_speed=args.overtake_min_rel_speed,
        bg_color=_parse_rgb(args.background_color, (255,255,255)),
        map_color=_parse_rgb(args.map_color, (180,180,180)),
        map_line_width=args.map_line_width,
        colorize=args.colorize,
        dual_label=args.dual_label,
        lane_change_lat_thresh=args.lane_change_lat,
        accel_thresh=args.accel_thresh,
        decel_thresh=args.decel_thresh,
        rel_speed_thresh=args.approach_rel_speed_thresh,
        max_behind_dist=args.approach_max_behind_dist,
        lat_min=args.approach_lat_min,
        max_ahead_dist=args.proximity_max_ahead_dist,
        close_dist=args.proximity_close_dist,
        lateral_min=args.proximity_lateral_min,
        progress=args.progress,
        quiet=args.quiet,
    )
