# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_timestamp(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return 0.0
    if re.fullmatch(r'\d+(\.\d+)?', s):
        return float(s)
    parts = s.split(':')
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        raise ValueError(f'Invalid timestamp: {value}')
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    raise ValueError(f'Invalid timestamp: {value}')


def format_timestamp(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(round(sec - m * 60))
    if s >= 60:
        m += 1
        s -= 60
    return f'{m}:{s:02d}'


def load_ground_truth(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if 'timestamp' not in df.columns:
        raise ValueError('Ground truth CSV must contain a timestamp column.')
    df = df.copy()
    df['time_sec'] = df['timestamp'].apply(parse_timestamp)
    df = df.sort_values('time_sec').reset_index(drop=True)
    return df


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    raw = str(video_path).strip()
    if not raw:
        raise ValueError('Video path is empty. Select a video file before starting.')
    p = Path(raw)
    if not p.is_file():
        raise FileNotFoundError(f'Video file was not found: {raw}')
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'OpenCV could not open the video file: {p}')
    return cap


def read_frame_at(cap: cv2.VideoCapture, sec: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f'Cannot read frame at {format_timestamp(sec)}')
    return frame


def video_info(video_path: str | Path) -> Dict[str, float]:
    cap = open_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
        duration = frames / fps if fps > 0 else 0.0
        return {'fps': fps, 'frames': frames, 'width': w, 'height': h, 'duration_sec': duration}
    finally:
        cap.release()


def order_quad_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return 4 points ordered TL, TR, BR, BL."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(sums)]      # TL
    rect[2] = pts[np.argmax(sums)]      # BR
    rect[1] = pts[np.argmin(diffs)]     # TR
    rect[3] = pts[np.argmax(diffs)]     # BL
    return rect


def perspective_output_size(points: Sequence[Sequence[float]], min_size: int = 64) -> Tuple[int, int]:
    rect = order_quad_points(points)
    tl, tr, br, bl = rect
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    width = max(int(round(max(width_top, width_bottom))), int(min_size))
    height = max(int(round(max(height_left, height_right))), int(min_size))
    return width, height


def make_perspective_roi(points: Sequence[Sequence[float]], output_size: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    rect = order_quad_points(points)
    if output_size is None:
        output_size = perspective_output_size(rect)
    w, h = int(output_size[0]), int(output_size[1])
    return {
        'mode': 'perspective',
        'points': [[float(x), float(y)] for x, y in rect.tolist()],
        'output_size': [max(2, w), max(2, h)],
    }


def crop_roi(frame: np.ndarray, roi: Any) -> np.ndarray:
    """Crop either a legacy rectangular ROI or a 4-point perspective ROI.

    Perspective ROI format:
      {"mode": "perspective", "points": [[x,y] x4], "output_size": [w,h]}
    """
    if isinstance(roi, dict) and str(roi.get('mode', '')).lower() == 'perspective':
        points = roi.get('points')
        if not points or len(points) != 4:
            raise ValueError('Perspective ROI requires exactly 4 points.')
        src = order_quad_points(points).astype(np.float32)
        size = roi.get('output_size') or perspective_output_size(src)
        w, h = max(2, int(size[0])), max(2, int(size[1]))
        dst = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Backward-compatible rectangular ROI.
    x, y, w, h = [int(v) for v in roi]
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    return frame[y:y+h, x:x+w].copy()


def point_in_box(point: Sequence[float], box: Sequence[float], expand: float = 0.0) -> bool:
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    if expand:
        bw = x2 - x1
        bh = y2 - y1
        x1 -= bw * expand / 2
        x2 += bw * expand / 2
        y1 -= bh * expand / 2
        y2 += bh * expand / 2
    return x1 <= px <= x2 and y1 <= py <= y2


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def clip_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def robust_region(boxes: List[Sequence[float]], fallback_point: Sequence[float], frame_shape: Sequence[int]) -> List[int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if not boxes:
        px, py = [int(v) for v in fallback_point]
        half_w = max(24, int(w * 0.04))
        half_h = max(24, int(h * 0.06))
        return list(clip_box((px-half_w, py-half_h, px+half_w, py+half_h), w, h))
    arr = np.asarray(boxes, dtype=np.float32)
    q = np.median(arr, axis=0)
    x1, y1, x2, y2 = q.tolist()
    bw = max(10.0, x2 - x1)
    bh = max(10.0, y2 - y1)
    # Slightly shrink toward the center to reduce overlap with neighboring slots.
    margin_x = bw * 0.08
    margin_y = bh * 0.08
    return list(clip_box((x1+margin_x, y1+margin_y, x2-margin_x, y2-margin_y), w, h))


def normalized_visual_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 1.0
    target_w = min(160, max(32, a.shape[1]))
    target_h = min(120, max(32, a.shape[0]))
    aa = cv2.resize(a, (target_w, target_h), interpolation=cv2.INTER_AREA)
    bb = cv2.resize(b, (target_w, target_h), interpolation=cv2.INTER_AREA)
    aa = cv2.cvtColor(aa, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bb = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Reduce global brightness sensitivity by centering each crop.
    aa = aa - float(aa.mean())
    bb = bb - float(bb.mean())
    return float(np.mean(np.abs(aa - bb)))


def zip_paths(zip_path: str | Path, files: Iterable[Tuple[str | Path, str]]) -> None:
    zp = Path(zip_path)
    zp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files:
            p = Path(src)
            if p.exists() and p.is_file():
                zf.write(p, arcname)


def safe_copy(src: str | Path, dst: str | Path) -> None:
    s, d = Path(src), Path(dst)
    if s.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
