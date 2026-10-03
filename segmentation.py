# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class SegmentDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    mask_ratio: float
    core_overlap_ratio: float
    point_covered: int

    @property
    def box(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def score(self) -> float:
        return float(self.conf) + 0.35 * float(self.point_covered) + 0.25 * float(self.core_overlap_ratio)


class VehicleSegmenter:
    """Thin Ultralytics instance-segmentation wrapper used only on ambiguous slot crops."""

    def __init__(self, config: Dict):
        self.config = dict(config or {})
        self.model_name = str(self.config.get('model', 'yolov8m-seg.pt'))
        self.imgsz = int(self.config.get('imgsz', 640))
        self.conf = float(self.config.get('conf', 0.05))
        self.vehicle_classes = set(int(x) for x in self.config.get('vehicle_class_ids', [2, 3, 5, 7]))
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)

    @staticmethod
    def _polygon_mask(shape: Tuple[int, int], xy) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if xy is None:
            return mask
        pts = np.asarray(xy, dtype=np.float32)
        if pts.ndim != 2 or len(pts) < 3:
            return mask
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        return mask

    @staticmethod
    def _core_overlap(mask: np.ndarray, point: Tuple[float, float], radius: float) -> Tuple[float, int]:
        if mask.size == 0:
            return 0.0, 0
        h, w = mask.shape[:2]
        px = int(round(point[0])); py = int(round(point[1]))
        px = max(0, min(w - 1, px)); py = max(0, min(h - 1, py))
        point_covered = int(mask[py, px] > 0)
        r = max(4, int(round(radius)))
        core = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(core, (px, py), r, 1, -1)
        denom = int(core.sum())
        overlap = float(((mask > 0) & (core > 0)).sum()) / float(max(1, denom))
        return overlap, point_covered

    def segment_batch(
        self,
        images: Sequence[np.ndarray],
        points: Sequence[Tuple[float, float]],
        core_radii: Sequence[float],
    ) -> List[List[SegmentDetection]]:
        imgs = list(images)
        if not imgs:
            return []
        self._load()
        results = self._model.predict(
            source=imgs,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=sorted(self.vehicle_classes),
            verbose=False,
        )
        out: List[List[SegmentDetection]] = []
        for idx, res in enumerate(results):
            img = imgs[idx]
            h, w = img.shape[:2]
            point = points[idx]
            radius = core_radii[idx]
            boxes = getattr(res, 'boxes', None)
            masks = getattr(res, 'masks', None)
            polygons = list(getattr(masks, 'xy', []) or []) if masks is not None else []
            dets: List[SegmentDetection] = []
            if boxes is not None:
                for bi, b in enumerate(boxes):
                    cls = int(b.cls[0].detach().cpu().item())
                    if cls not in self.vehicle_classes:
                        continue
                    conf = float(b.conf[0].detach().cpu().item())
                    xyxy = b.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                    poly = polygons[bi] if bi < len(polygons) else None
                    mask = self._polygon_mask((h, w), poly)
                    if mask.sum() == 0:
                        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            mask[y1:y2, x1:x2] = 1
                    mask_ratio = float(mask.mean())
                    core_overlap, point_covered = self._core_overlap(mask, point, radius)
                    dets.append(SegmentDetection(
                        float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]),
                        conf, cls, mask_ratio, core_overlap, point_covered,
                    ))
            out.append(sorted(dets, key=lambda d: d.score(), reverse=True))
        while len(out) < len(imgs):
            out.append([])
        return out[:len(imgs)]


def choose_best_segment(dets: Sequence[SegmentDetection], min_core_overlap: float = 0.08) -> SegmentDetection | None:
    if not dets:
        return None
    good = [d for d in dets if d.point_covered or d.core_overlap_ratio >= float(min_core_overlap)]
    return max(good, key=lambda d: d.score()) if good else None

# -----------------------------------------------------------------------------
# v15.2 image-conditioning helpers
# These are intentionally mild. They do not invent detail; they try to preserve
# vehicle structure when monitor re-recording, glare or low contrast hurt the raw crop.
# -----------------------------------------------------------------------------

def enhance_parking_crop(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    src = image.astype(np.uint8, copy=False)
    # Mild chroma/noise smoothing helps periodic monitor texture without erasing edges.
    smooth = cv2.bilateralFilter(src, 5, 28, 28)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32) / 255.0
    # Compress very bright regions instead of hard clipping them.
    lf = np.power(np.clip(lf, 0.0, 1.0), 1.12)
    l2 = np.clip(lf * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.55, tileGridSize=(8, 8))
    l3 = clahe.apply(l2)
    merged = cv2.cvtColor(cv2.merge([l3, a, b]), cv2.COLOR_LAB2BGR)
    # Restore a little local edge contrast after denoise/CLAHE.
    blur = cv2.GaussianBlur(merged, (0, 0), 1.0)
    sharp = cv2.addWeighted(merged, 1.12, blur, -0.12, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    if image is None or image.size == 0:
        return {'highlight_ratio': 0.0, 'dark_ratio': 0.0, 'sharpness_laplacian_var': 0.0, 'stripe_energy_ratio': 0.0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    highlight = float((gray >= 238).mean())
    dark = float((gray <= 22).mean())
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # A cheap monitor-stripe proxy: repeated row/column intensity oscillation energy.
    row = gray.astype(np.float32).mean(axis=1)
    col = gray.astype(np.float32).mean(axis=0)
    def hf_ratio(x: np.ndarray) -> float:
        if x.size < 8:
            return 0.0
        x = x - float(x.mean())
        spec = np.abs(np.fft.rfft(x)) ** 2
        if spec.size <= 3 or float(spec.sum()) <= 1e-9:
            return 0.0
        cut = max(2, int(round(spec.size * 0.22)))
        return float(spec[cut:].sum() / max(1e-9, spec[1:].sum()))
    stripe = max(hf_ratio(row), hf_ratio(col))
    return {'highlight_ratio': highlight, 'dark_ratio': dark, 'sharpness_laplacian_var': sharp, 'stripe_energy_ratio': stripe}



def build_relative_quality_thresholds(rows, cfg: Dict) -> Dict[str, Dict[str, float]]:
    """Build per-CCTV degradation thresholds from the actual video distribution.

    rows: iterable of dicts with cctv/highlight_ratio/stripe_energy_ratio/sharpness_laplacian_var.
    Falls back to the global distribution when a camera has too few samples.
    """
    data = list(rows or [])
    if not data:
        return {}
    hp = float(cfg.get('highlight_percentile', 0.90))
    sp = float(cfg.get('stripe_percentile', 0.90))
    qp = float(cfg.get('sharpness_percentile', 0.10))
    min_n = max(5, int(cfg.get('condition_min_samples', 30)))

    def one(part):
        h = np.asarray([float(x.get('highlight_ratio', 0.0)) for x in part], dtype=np.float64)
        st = np.asarray([float(x.get('stripe_energy_ratio', 0.0)) for x in part], dtype=np.float64)
        sh = np.asarray([float(x.get('sharpness_laplacian_var', 0.0)) for x in part], dtype=np.float64)
        return {
            'N': int(len(part)),
            'highlight_threshold': float(np.quantile(h, hp)),
            'stripe_threshold': float(np.quantile(st, sp)),
            'sharpness_threshold': float(np.quantile(sh, qp)),
            'highlight_median': float(np.median(h)),
            'stripe_median': float(np.median(st)),
            'sharpness_median': float(np.median(sh)),
            'highlight_hi': float(np.quantile(h, min(0.99, max(hp, 0.95)))),
            'stripe_hi': float(np.quantile(st, min(0.99, max(sp, 0.95)))),
            'sharpness_lo': float(np.quantile(sh, max(0.01, min(qp, 0.05)))),
        }

    global_thr = one(data)
    out = {'__GLOBAL__': global_thr}
    cams = sorted({str(x.get('cctv', '')) for x in data if str(x.get('cctv', ''))})
    for cam in cams:
        part = [x for x in data if str(x.get('cctv', '')) == cam]
        out[cam] = one(part) if len(part) >= min_n else dict(global_thr, N=int(len(part)), fallback_global=1)
    return out


def classify_quality_condition(q: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[str, float]:
    """Classify the most extreme degradation relative to one CCTV's own distribution."""
    if not thresholds:
        return 'CLEAN', 0.0
    h = float(q.get('highlight_ratio', 0.0))
    st = float(q.get('stripe_energy_ratio', 0.0))
    sh = float(q.get('sharpness_laplacian_var', 0.0))
    ht = float(thresholds.get('highlight_threshold', 1.0))
    stt = float(thresholds.get('stripe_threshold', 1.0))
    sht = float(thresholds.get('sharpness_threshold', -1.0))
    hhi = max(ht + 1e-9, float(thresholds.get('highlight_hi', ht + 1e-6)))
    shi = max(stt + 1e-9, float(thresholds.get('stripe_hi', stt + 1e-6)))
    slo = min(sht - 1e-9, float(thresholds.get('sharpness_lo', sht - 1e-6)))
    scores = []
    if h >= ht:
        scores.append(('SUN_GLARE', max(0.0, (h - ht) / max(1e-9, hhi - ht))))
    if st >= stt:
        scores.append(('MONITOR_STRIPES', max(0.0, (st - stt) / max(1e-9, shi - stt))))
    if sh <= sht:
        scores.append(('LOW_RES', max(0.0, (sht - sh) / max(1e-9, sht - slo))))
    if not scores:
        return 'CLEAN', 0.0
    # Prefer the strongest relative tail excursion, not a fixed degradation priority.
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0], float(scores[0][1])

# -----------------------------------------------------------------------------
# v15.4 condition-specific recovery helpers
# One generic enhancement was harmful to segmentation in v15.2, especially for
# glare and monitor stripes. These functions deliberately separate the recovery
# path by degradation type. They never synthesize detail; they only resample,
# tone-compress or aggregate recent causal frames.
# -----------------------------------------------------------------------------

def recover_low_res_crop(image: np.ndarray) -> np.ndarray:
    """Mild low-resolution recovery using interpolation + conservative unsharping."""
    if image is None or image.size == 0:
        return image
    src = image.astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    scale = 1.8
    up = cv2.resize(src, (max(2, int(round(w * scale))), max(2, int(round(h * scale)))), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 0.9)
    sharp = cv2.addWeighted(up, 1.08, blur, -0.08, 0)
    # Keep the original crop geometry so the manual-point/core coordinates remain valid.
    out = cv2.resize(sharp, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0, 255).astype(np.uint8)


def recover_sun_glare_crop(image: np.ndarray) -> np.ndarray:
    """Compress highlights without CLAHE, which over-amplified texture in v15.2."""
    if image is None or image.size == 0:
        return image
    src = cv2.bilateralFilter(image.astype(np.uint8, copy=False), 5, 24, 24)
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    x = l.astype(np.float32) / 255.0
    # Gamma > 1 gently pulls bright/mid pixels down while preserving ordering.
    x = np.power(np.clip(x, 0.0, 1.0), 1.22)
    # Soft shoulder for the highest highlights; no local contrast amplification.
    shoulder = x / (x + 0.10 * (1.0 - x) + 1e-6)
    x = 0.72 * x + 0.28 * shoulder
    l2 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def temporal_median_crop(frames: Sequence[np.ndarray]) -> np.ndarray | None:
    """Causal temporal median. Frames should be current + previous frames of one slot."""
    good = [f for f in frames if f is not None and getattr(f, 'size', 0) > 0]
    if not good:
        return None
    h, w = good[-1].shape[:2]
    aligned = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) if f.shape[:2] != (h, w) else f for f in good]
    stack = np.stack(aligned, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def recover_monitor_stripes_crop(image: np.ndarray, temporal_frames: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """Prefer temporal median for monitor stripe/moire; fall back to mild spatial cleanup."""
    base = temporal_median_crop(list(temporal_frames or []))
    if base is None:
        base = image
    if base is None or base.size == 0:
        return base
    # Very mild cleanup only; aggressive CLAHE/sharpening made stripes worse in v15.2.
    med = cv2.medianBlur(base.astype(np.uint8, copy=False), 3)
    return cv2.bilateralFilter(med, 5, 18, 18)


def conditioned_parking_crop(image: np.ndarray, condition: str = 'CLEAN', temporal_frames: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """Condition-aware preprocessing used by v15.4 robustness/SEG experiments."""
    c = str(condition or 'CLEAN').upper()
    if c == 'LOW_RES':
        # Fixed-CCTV low-resolution recovery can profit from causal multi-frame aggregation:
        # reduce frame noise first, then do conservative interpolation/unsharping.
        base = temporal_median_crop(list(temporal_frames or []))
        return recover_low_res_crop(base if base is not None else image)
    if c in ('SUN_GLARE', 'GLARE', 'OVEREXPOSURE'):
        return recover_sun_glare_crop(image)
    if c in ('MONITOR_STRIPES', 'MOIRE', 'STRIPES'):
        return recover_monitor_stripes_crop(image, temporal_frames=temporal_frames)
    return image.copy() if image is not None else image
