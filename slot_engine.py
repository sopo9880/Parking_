# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from detector import Detection, VehicleDetector
from utils import (
    box_center,
    box_iou,
    clip_box,
    crop_roi,
    ensure_dir,
    format_timestamp,
    load_ground_truth,
    load_json,
    normalized_visual_diff,
    open_video,
    point_in_box,
    read_frame_at,
    robust_region,
    save_json,
    zip_paths,
)


# -----------------------------------------------------------------------------
# Slot I/O
# -----------------------------------------------------------------------------

def load_slots(path: str | Path) -> List[Dict]:
    data = load_json(path, {'slots': []}) or {'slots': []}
    return list(data.get('slots', []))


def save_slots(path: str | Path, slots: List[Dict]) -> None:
    save_json(path, {'slots': slots})


def next_global_id(slots: List[Dict]) -> str:
    used = []
    for s in slots:
        g = str(s.get('global_id', ''))
        if g.startswith('G') and g[1:].isdigit():
            used.append(int(g[1:]))
    return f'G{max(used, default=0) + 1:03d}'


def next_local_id(slots: List[Dict], cctv: str) -> str:
    prefix = cctv.upper().replace('CCTV', 'C') + '_S'
    nums = []
    for s in slots:
        lid = str(s.get('local_id', ''))
        if lid.startswith(prefix) and lid[len(prefix):].isdigit():
            nums.append(int(lid[len(prefix):]))
    return f'{prefix}{max(nums, default=0)+1:03d}'


# -----------------------------------------------------------------------------
# v9 constrained one-to-one assignment
# Manual slot points are the immutable geometric prior. Learned anchors may help,
# but they can never redefine slot identity or cross a manual Voronoi boundary.
# -----------------------------------------------------------------------------

def _hungarian_minimize(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Small dependency-free Hungarian solver for rectangular finite cost matrices."""
    a = np.asarray(cost, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        return []
    transposed = False
    if a.shape[0] > a.shape[1]:
        a = a.T
        transposed = True
    n, m = a.shape
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    inf = 1e18
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, m + 1):
        if p[j] > 0:
            r, c = int(p[j] - 1), int(j - 1)
            pairs.append((c, r) if transposed else (r, c))
    return pairs


def _manual_point(slot: Dict) -> Tuple[float, float]:
    p = slot.get('point', [0.0, 0.0])
    return float(p[0]), float(p[1])


def _effective_anchor(slot: Dict) -> Tuple[float, float]:
    if bool(slot.get('anchor_trusted', False)) and slot.get('anchor_center') is not None:
        p = slot['anchor_center']
        return float(p[0]), float(p[1])
    return _manual_point(slot)


def _manual_neighbor_distances(slots: Sequence[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ss = list(slots)
    for s in ss:
        px, py = _manual_point(s)
        dists = []
        for q in ss:
            if q is s:
                continue
            qx, qy = _manual_point(q)
            dists.append(math.hypot(px-qx, py-qy))
        out[str(s['local_id'])] = min(dists) if dists else 100.0
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _manual_gate_radius(neighbor_dist: float, cfg: Dict) -> float:
    return _clamp(
        float(neighbor_dist) * float(cfg.get('manual_gate_neighbor_factor', 0.82)),
        float(cfg.get('manual_gate_min_px', 28.0)),
        float(cfg.get('manual_gate_max_px', 120.0)),
    )


def _pair_cost(
    slot: Dict,
    det: Detection,
    neighbor_dist: float,
    cfg: Dict,
    voronoi_best_dist: float,
    voronoi_allowed: bool,
) -> Tuple[float, Dict]:
    dcx, dcy = box_center(det.box)
    mpx, mpy = _manual_point(slot)
    acx, acy = _effective_anchor(slot)
    manual_dist = math.hypot(mpx-dcx, mpy-dcy)
    anchor_dist = math.hypot(acx-dcx, acy-dcy)
    gate = _manual_gate_radius(neighbor_dist, cfg)
    contains_manual = point_in_box((mpx, mpy), det.box, expand=float(cfg.get('point_containment_expand', 0.10)))
    region = slot.get('region')
    riou = box_iou(region, det.box) if region else 0.0

    # Hard identity protection: a detection is considered only for the manual-point
    # Voronoi cell(s) nearest to its center. A small boundary slack handles ties/noise.
    if not voronoi_allowed:
        return float(cfg.get('invalid_cost', 5.0)), {
            'manual_distance_px': manual_dist,
            'anchor_distance_px': anchor_dist,
            'voronoi_best_distance_px': voronoi_best_dist,
            'voronoi_allowed': 0,
            'region_iou': riou,
            'contains_point': int(contains_manual),
            'match_score': 0.0,
        }

    # The manual point remains a hard spatial prior. Containment can rescue a center
    # displaced by perspective, but the Voronoi gate above still prevents slot hopping.
    if manual_dist > gate and not contains_manual:
        return float(cfg.get('invalid_cost', 5.0)), {
            'manual_distance_px': manual_dist,
            'anchor_distance_px': anchor_dist,
            'voronoi_best_distance_px': voronoi_best_dist,
            'voronoi_allowed': 1,
            'manual_gate_px': gate,
            'region_iou': riou,
            'contains_point': 0,
            'match_score': 0.0,
        }

    expected_wh = slot.get('expected_box_wh') or [max(20.0, det.x2-det.x1), max(20.0, det.y2-det.y1)]
    ew, eh = max(8.0, float(expected_wh[0])), max(8.0, float(expected_wh[1]))
    dw, dh = max(8.0, det.x2-det.x1), max(8.0, det.y2-det.y1)
    size_cost = min(2.0, abs(math.log(dw/ew)) + abs(math.log(dh/eh)))
    md_cost = min(2.0, manual_dist / max(1.0, gate))
    ad_cost = min(2.0, anchor_dist / max(1.0, gate))

    mw = float(cfg.get('manual_distance_weight', 0.82))
    aw = float(cfg.get('anchor_distance_weight', 0.18)) if bool(slot.get('anchor_trusted', False)) else 0.0
    if aw <= 0:
        mw = max(mw, 1.0)
    denom = max(1e-6, mw + aw)
    dist_cost = (mw*md_cost + aw*ad_cost) / denom

    cost = (
        dist_cost
        + float(cfg.get('size_weight', 0.06))*size_cost
        - float(cfg.get('region_iou_bonus', 0.08))*riou
        - float(cfg.get('confidence_bonus', 0.10))*float(det.conf)
        - (0.10 if contains_manual else 0.0)
    )
    cost = max(0.0, float(cost))
    unmatched = float(cfg.get('unmatched_cost', 1.15))
    score = max(0.0, min(1.0, 1.0-cost/max(1e-6, unmatched)))
    return cost, {
        'manual_distance_px': manual_dist,
        'anchor_distance_px': anchor_dist,
        'distance_px': manual_dist,
        'voronoi_best_distance_px': voronoi_best_dist,
        'voronoi_allowed': 1,
        'manual_gate_px': gate,
        'region_iou': riou,
        'contains_point': int(contains_manual),
        'anchor_trusted': int(bool(slot.get('anchor_trusted', False))),
        'match_score': score,
    }


def assign_detections_to_slots(
    dets: Sequence[Detection], slots: Sequence[Dict], matching_cfg: Optional[Dict] = None,
) -> Tuple[Dict[str, Detection], List[Detection], Dict[str, Dict]]:
    """Manual-Voronoi-constrained Hungarian vehicle↔slot assignment.

    Invariants:
      * one physical detection -> at most one local slot
      * one local slot -> at most one detection
      * the manual point defines slot identity; a learned anchor cannot move a detection
        into another manual slot's Voronoi cell
    """
    cfg = matching_cfg or {}
    slots = list(slots)
    dets = list(dets)
    if not slots:
        return {}, list(dets), {}
    if not dets:
        return {}, [], {}

    unmatched_cost = float(cfg.get('unmatched_cost', 1.15))
    invalid_cost = float(cfg.get('invalid_cost', 5.0))
    slack = float(cfg.get('voronoi_boundary_slack_px', 8.0))
    neighbors = _manual_neighbor_distances(slots)
    n_s, n_d = len(slots), len(dets)
    cost = np.full((n_s, n_d+n_s), unmatched_cost, dtype=np.float64)
    meta_matrix: List[List[Dict]] = [[{} for _ in range(n_d)] for __ in range(n_s)]

    # Compute manual-point Voronoi distances once per detection.
    det_manual_dists: List[List[float]] = []
    det_best: List[float] = []
    for d in dets:
        cx, cy = box_center(d.box)
        arr = [math.hypot(_manual_point(s)[0]-cx, _manual_point(s)[1]-cy) for s in slots]
        det_manual_dists.append(arr)
        det_best.append(min(arr) if arr else 1e9)

    for si, s in enumerate(slots):
        nd = neighbors.get(str(s['local_id']), 100.0)
        for di, d in enumerate(dets):
            md = det_manual_dists[di][si]
            allowed = md <= det_best[di] + slack
            c, meta = _pair_cost(s, d, nd, cfg, det_best[di], allowed)
            cost[si, di] = c
            meta['voronoi_margin_px'] = float(md-det_best[di])
            meta_matrix[si][di] = meta
        for k in range(n_s):
            cost[si, n_d+k] = unmatched_cost + (0.0001 if k != si else 0.0)

    pairs = _hungarian_minimize(cost)
    assigned: Dict[str, Detection] = {}
    assignment_meta: Dict[str, Dict] = {}
    used_dets = set()
    min_score = float(cfg.get('min_match_score', 0.03))
    for si, col in pairs:
        if si >= n_s or col >= n_d:
            continue
        c = float(cost[si, col])
        if c >= min(unmatched_cost, invalid_cost):
            continue
        meta = dict(meta_matrix[si][col])
        if float(meta.get('match_score', 0.0)) < min_score:
            continue
        lid = str(slots[si]['local_id'])
        assigned[lid] = dets[col]
        meta['assignment_cost'] = c
        meta['det_index'] = int(col)
        assignment_meta[lid] = meta
        used_dets.add(col)
    unmatched = [d for i, d in enumerate(dets) if i not in used_dets]
    return assigned, unmatched, assignment_meta


# -----------------------------------------------------------------------------
# Learning: constrained anchor center, compact visual regions, diagnostics
# -----------------------------------------------------------------------------

def _compact_visual_region(
    center: Sequence[float], box_samples: Sequence[Sequence[float]],
    fallback_point: Sequence[float], frame_shape: Sequence[int], settings: Dict,
) -> List[int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    cx, cy = float(center[0]), float(center[1])
    if not np.isfinite(cx) or not np.isfinite(cy):
        cx, cy = float(fallback_point[0]), float(fallback_point[1])
    if box_samples:
        boxes = np.asarray(box_samples, dtype=np.float32)
        widths = boxes[:,2]-boxes[:,0]
        heights = boxes[:,3]-boxes[:,1]
        bw, bh = float(np.median(widths)), float(np.median(heights))
        rw = max(18.0, bw*float(settings.get('visual_region_width_scale', 0.38)))
        rh = max(18.0, bh*float(settings.get('visual_region_height_scale', 0.46)))
    else:
        rw = max(26.0, w*0.050)
        rh = max(26.0, h*0.070)
    return list(clip_box((cx-rw/2, cy-rh/2, cx+rw/2, cy+rh/2), w, h))


def _learn_slot_geometry(
    slot: Dict,
    centers: List[Tuple[float,float]],
    boxes: List[List[float]],
    slots_same_cam: Sequence[Dict],
    img_shape,
    settings: Dict,
):
    mpx, mpy = _manual_point(slot)
    manual_neighbors = []
    for other in slots_same_cam:
        if other is slot:
            continue
        opx, opy = _manual_point(other)
        manual_neighbors.append(math.hypot(mpx-opx, mpy-opy))
    nearest = min(manual_neighbors) if manual_neighbors else 100.0

    sample_gate = _clamp(
        nearest*float(settings.get('anchor_sample_gate_neighbor_factor', 0.90)),
        float(settings.get('anchor_sample_gate_min_px', 35.0)),
        float(settings.get('anchor_sample_gate_max_px', 120.0)),
    )
    max_shift = _clamp(
        nearest*float(settings.get('anchor_max_shift_neighbor_factor', 0.40)),
        float(settings.get('anchor_max_shift_min_px', 18.0)),
        float(settings.get('anchor_max_shift_max_px', 55.0)),
    )

    valid_centers: List[Tuple[float,float]] = []
    valid_boxes: List[List[float]] = []
    for i, c in enumerate(centers):
        cx, cy = float(c[0]), float(c[1])
        if math.hypot(cx-mpx, cy-mpy) <= sample_gate:
            valid_centers.append((cx, cy))
            if i < len(boxes):
                valid_boxes.append(boxes[i])

    raw_count = int(len(centers))
    valid_count = int(len(valid_centers))
    min_samples = int(settings.get('min_anchor_samples', 30))
    max_std = float(settings.get('max_anchor_std_px', 30.0))
    trusted = False
    reason = 'manual_fallback_no_samples'
    anchor = np.asarray([mpx, mpy], dtype=np.float32)
    std_xy = np.asarray([0.0, 0.0], dtype=np.float32)
    raw_candidate = anchor.copy()
    clamped = False

    if valid_centers:
        carr = np.asarray(valid_centers, dtype=np.float32)
        raw_candidate = np.median(carr, axis=0)
        std_xy = np.std(carr, axis=0) if len(carr)>1 else np.asarray([0.0,0.0], dtype=np.float32)
        std_mean = float(np.mean(std_xy))
        if valid_count >= min_samples and std_mean <= max_std:
            delta = raw_candidate-np.asarray([mpx,mpy], dtype=np.float32)
            shift = float(np.linalg.norm(delta))
            reject_far = bool(settings.get('anchor_reject_if_exceeds_max_shift', True)) and shift > max_shift
            if reject_far:
                # v15: the administrator's manual point is the slot identity.
                # A learned anchor that wants to leave the allowed neighborhood is rejected,
                # not merely clamped to the boundary. This prevents C3_S011-style drift.
                anchor = np.asarray([mpx,mpy], dtype=np.float32)
                trusted = False
                reason = f'manual_fallback_raw_shift>{max_shift:.1f}'
            else:
                if shift > max_shift and shift > 1e-6:
                    delta *= float(max_shift/shift)
                    clamped = True
                anchor = np.asarray([mpx,mpy], dtype=np.float32)+delta
                trusted = True
                reason = 'trusted_clamped' if clamped else 'trusted'
        elif valid_count < min_samples:
            reason = f'manual_fallback_samples<{min_samples}'
        else:
            reason = f'manual_fallback_std>{max_std:.1f}'

    if valid_boxes:
        barr = np.asarray(valid_boxes, dtype=np.float32)
        widths = barr[:,2]-barr[:,0]
        heights = barr[:,3]-barr[:,1]
        expected_wh = [round(float(np.median(widths)),2), round(float(np.median(heights)),2)]
    elif boxes:
        barr = np.asarray(boxes, dtype=np.float32)
        widths = barr[:,2]-barr[:,0]
        heights = barr[:,3]-barr[:,1]
        expected_wh = [round(float(np.median(widths)),2), round(float(np.median(heights)),2)]
    else:
        h,w = img_shape[:2]
        expected_wh = [round(float(max(24,w*0.08)),2), round(float(max(24,h*0.10)),2)]

    slot['geometry_version'] = 'v12_manual_voronoi'
    slot['manual_point'] = [round(mpx,2), round(mpy,2)]
    slot['anchor_center_raw'] = [round(float(raw_candidate[0]),2), round(float(raw_candidate[1]),2)]
    slot['anchor_center'] = [round(float(anchor[0]),2), round(float(anchor[1]),2)]
    slot['anchor_trusted'] = bool(trusted)
    slot['anchor_reason'] = reason
    slot['anchor_clamped'] = bool(clamped)
    slot['anchor_std_xy_px'] = [round(float(std_xy[0]),2), round(float(std_xy[1]),2)]
    slot['anchor_std_px'] = round(float(np.mean(std_xy)),2)
    slot['anchor_shift_px'] = round(float(math.hypot(float(anchor[0])-mpx, float(anchor[1])-mpy)),2)
    slot['anchor_max_shift_px'] = round(float(max_shift),2)
    slot['anchor_sample_gate_px'] = round(float(sample_gate),2)
    slot['nearest_manual_slot_distance_px'] = round(float(nearest),2)
    slot['expected_box_wh'] = expected_wh
    slot['assignment_radius_px'] = round(float(_manual_gate_radius(nearest, settings)),2)
    region_center = slot['anchor_center'] if trusted else slot['point']
    slot['region'] = _compact_visual_region(region_center, valid_boxes or boxes, slot['point'], img_shape, settings)
    slot['learned_box_samples'] = int(len(boxes))
    slot['learned_center_samples_raw'] = raw_count
    slot['learned_center_samples'] = valid_count


def _reset_learned_geometry(slots: Sequence[Dict]) -> None:
    """Remove older learned geometry so every v12 Learn starts from manual points."""
    learned_keys = {
        'anchor_center','anchor_center_raw','anchor_trusted','anchor_reason','anchor_clamped',
        'anchor_std_xy_px','anchor_std_px','anchor_shift_px','anchor_max_shift_px',
        'anchor_sample_gate_px','nearest_manual_slot_distance_px','nearest_slot_distance_px',
        'expected_box_wh','assignment_radius_px','region','learned_box_samples',
        'learned_center_samples_raw','learned_center_samples','manual_point','template_initial','geometry_version',
    }
    for s in slots:
        for k in learned_keys:
            s.pop(k, None)


def slot_learning_diagnostics(slots: Sequence[Dict]) -> pd.DataFrame:
    rows = []
    for s in slots:
        p = s.get('point', [0,0])
        a = s.get('anchor_center') or p
        rows.append({
            'cctv': s.get('cctv',''),
            'local_id': s.get('local_id',''),
            'global_id': s.get('global_id',''),
            'manual_x': float(p[0]), 'manual_y': float(p[1]),
            'anchor_x': float(a[0]), 'anchor_y': float(a[1]),
            'anchor_trusted': int(bool(s.get('anchor_trusted', False))),
            'anchor_reason': s.get('anchor_reason',''),
            'anchor_shift_px': float(s.get('anchor_shift_px',0.0) or 0.0),
            'anchor_max_shift_px': float(s.get('anchor_max_shift_px',0.0) or 0.0),
            'nearest_manual_slot_distance_px': float(s.get('nearest_manual_slot_distance_px',0.0) or 0.0),
            'learned_center_samples_raw': int(s.get('learned_center_samples_raw',0) or 0),
            'learned_center_samples': int(s.get('learned_center_samples',0) or 0),
            'anchor_std_px': float(s.get('anchor_std_px',0.0) or 0.0),
        })
    return pd.DataFrame(rows)


def compute_slot_overlap_warnings(slots: Sequence[Dict], settings: Dict) -> pd.DataFrame:
    rows = []
    warn_iou = float(settings.get('slot_overlap_warning_iou', 0.25))
    critical_iou = float(settings.get('slot_overlap_critical_iou', 0.60))
    close_px = float(settings.get('slot_close_anchor_warning_px', 24.0))
    shift_ratio = float(settings.get('anchor_shift_warning_ratio', 0.32))
    by_cam = defaultdict(list)
    for s in slots:
        by_cam[s['cctv']].append(s)
    for cctv, ss in by_cam.items():
        for i in range(len(ss)):
            for j in range(i+1,len(ss)):
                a,b = ss[i],ss[j]
                aiou = box_iou(a.get('region',[0,0,0,0]), b.get('region',[0,0,0,0]))
                ap, bp = _effective_anchor(a), _effective_anchor(b)
                amp, bmp = _manual_point(a), _manual_point(b)
                adist = math.hypot(ap[0]-bp[0], ap[1]-bp[1])
                mdist = math.hypot(amp[0]-bmp[0], amp[1]-bmp[1])
                severity = ''
                reasons = []
                if aiou >= critical_iou:
                    severity='CRITICAL'; reasons.append(f'region_iou>={critical_iou:.2f}')
                elif aiou >= warn_iou:
                    severity='WARNING'; reasons.append(f'region_iou>={warn_iou:.2f}')
                if adist <= close_px:
                    if severity != 'CRITICAL': severity='WARNING'
                    reasons.append(f'anchor_distance<={close_px:.0f}px')
                for x, name in ((a,'A'),(b,'B')):
                    nearest = float(x.get('nearest_manual_slot_distance_px',0.0) or 0.0)
                    shift = float(x.get('anchor_shift_px',0.0) or 0.0)
                    if nearest > 0 and shift > nearest*shift_ratio:
                        if severity != 'CRITICAL': severity='WARNING'
                        reasons.append(f'{name}_anchor_shift>{shift_ratio:.2f}*neighbor')
                if severity:
                    rows.append({
                        'cctv':cctv,'slot_a':a['local_id'],'slot_b':b['local_id'],
                        'region_iou':round(float(aiou),4),
                        'manual_distance_px':round(float(mdist),2),
                        'anchor_distance_px':round(float(adist),2),
                        'slot_a_anchor_shift_px':float(a.get('anchor_shift_px',0.0) or 0.0),
                        'slot_b_anchor_shift_px':float(b.get('anchor_shift_px',0.0) or 0.0),
                        'severity':severity,'reason':';'.join(reasons),
                    })
    cols=['cctv','slot_a','slot_b','region_iou','manual_distance_px','anchor_distance_px',
          'slot_a_anchor_shift_px','slot_b_anchor_shift_px','severity','reason']
    return pd.DataFrame(rows, columns=cols)



def _debug_canvas(img: np.ndarray, settings: Optional[Dict] = None) -> Tuple[np.ndarray, float]:
    """Upscale debug figures so labels remain readable in paper/review images."""
    cfg=(settings or {}).get('debug_render',{}) if isinstance(settings,dict) else {}
    min_width=float(cfg.get('min_width_px',2200)); min_scale=float(cfg.get('min_scale',2.0))
    scale=max(min_scale, min_width/max(1.0,float(img.shape[1])))
    scale=min(scale,4.0)
    out=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    return out,float(scale)


def _p(pt, scale: float) -> Tuple[int,int]:
    return (int(round(float(pt[0])*scale)),int(round(float(pt[1])*scale)))


def _label_box(canvas: np.ndarray, text: str, xy: Tuple[int,int], scale: float = 1.0,
               fg=(255,255,255), bg=(15,15,15), font_scale: float = 0.62, thickness: int = 2) -> None:
    x,y=int(xy[0]),int(xy[1]); fs=max(0.45,font_scale*max(1.0,scale/2.0)); th=max(1,int(round(thickness*max(1.0,scale/2.0))))
    (tw,tht),base=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,fs,th)
    pad=max(4,int(round(5*max(1.0,scale/2.0))))
    x=max(0,min(canvas.shape[1]-tw-2*pad,x)); y=max(tht+2*pad,min(canvas.shape[0]-base-pad,y))
    cv2.rectangle(canvas,(x-pad,y-tht-pad),(x+tw+pad,y+base+pad),bg,-1)
    cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),th+3,cv2.LINE_AA)
    cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,fs,fg,th,cv2.LINE_AA)

def _draw_overlap_previews(slots: Sequence[Dict], first_images: Dict[str,np.ndarray], warnings: pd.DataFrame, out_dir: Path, settings: Optional[Dict]=None):
    by_id={s['local_id']:s for s in slots}
    for cctv,img in first_images.items():
        canvas,sc=_debug_canvas(img,settings)
        # Legend first: readable colors, no yellow-on-yellow text.
        _label_box(canvas,'MANUAL=cyan ring | LEARNED=green dot | OVERLAP=yellow/orange line | CRITICAL=red',
                   (24,44),sc,font_scale=0.58)
        for s in [x for x in slots if x['cctv']==cctv]:
            mx,my=map(float,s['point']); ax,ay=_effective_anchor(s)
            x1,y1,x2,y2=map(float,s.get('region',[mx-10,my-10,mx+10,my+10]))
            cv2.rectangle(canvas,_p((x1,y1),sc),_p((x2,y2),sc),(110,110,110),max(1,int(round(sc))))
            cv2.circle(canvas,_p((mx,my),sc),max(7,int(round(6*sc))),(255,210,0),max(2,int(round(2*sc))))
            if math.hypot(ax-mx,ay-my)>1.0:
                cv2.line(canvas,_p((mx,my),sc),_p((ax,ay),sc),(255,255,255),max(1,int(round(sc))),cv2.LINE_AA)
            acolor=(0,220,0) if bool(s.get('anchor_trusted',False)) else (150,150,150)
            cv2.circle(canvas,_p((ax,ay),sc),max(5,int(round(4*sc))),acolor,-1)
            source=str(s.get('source','MANUAL'))
            extra=''
            if source=='CANDIDATE_DUPLICATE_CONFIRMED': extra=f" DUP->{s.get('duplicate_of','?')}"
            _label_box(canvas,f"{s['local_id']} | {'LEARNED' if s.get('anchor_trusted') else 'MANUAL'}{extra}",
                       _p((mx+8,my-8),sc),sc)
        if not warnings.empty:
            anchor_labels=set()
            for _,r in warnings[warnings['cctv']==cctv].iterrows():
                a=by_id.get(r['slot_a']);b=by_id.get(r['slot_b'])
                if not a or not b:continue
                reason=str(r.get('reason',''))
                pair_problem=('region_iou' in reason) or ('anchor_distance' in reason)
                # Anchor-shift-only warnings belong to one slot; do not draw a misleading
                # long yellow line between unrelated slots.
                if 'A_anchor_shift' in reason: anchor_labels.add(str(r['slot_a']))
                if 'B_anchor_shift' in reason: anchor_labels.add(str(r['slot_b']))
                if not pair_problem:
                    continue
                ap=_manual_point(a);bp=_manual_point(b)
                color=(0,0,255) if r['severity']=='CRITICAL' else (0,215,255)
                p1=_p(ap,sc);p2=_p(bp,sc)
                cv2.line(canvas,p1,p2,color,max(3,int(round(2*sc))),cv2.LINE_AA)
                mid=((p1[0]+p2[0])//2,(p1[1]+p2[1])//2)
                _label_box(canvas,f"{r['severity']} {r['slot_a']} <-> {r['slot_b']} IoU={float(r['region_iou']):.2f}",
                           (mid[0]+8,mid[1]-8),sc,fg=(255,255,255),bg=(25,25,25))
            for lid in sorted(anchor_labels):
                sl=by_id.get(lid)
                if not sl: continue
                mp=_manual_point(sl)
                _label_box(canvas,f'ANCHOR SHIFT WARNING | {lid}',_p((mp[0]+12,mp[1]+24),sc),sc,fg=(255,255,255),bg=(0,70,170))
        cv2.imwrite(str(out_dir/f'{cctv}_slot_overlap_preview.jpg'),canvas,[int(cv2.IMWRITE_JPEG_QUALITY),95])



def _draw_voronoi_previews(slots: Sequence[Dict], first_images: Dict[str,np.ndarray], out_dir: Path, settings: Optional[Dict]=None) -> None:
    """Render readable manual-point Voronoi boundaries for audit/debugging."""
    by_cam=defaultdict(list)
    for s in slots: by_cam[s['cctv']].append(s)
    for cctv,img in first_images.items():
        ss=by_cam.get(cctv,[])
        if not ss:continue
        h,w=img.shape[:2];step=max(2,int(round(max(h,w)/500.0)));gh=max(2,int(math.ceil(h/step)));gw=max(2,int(math.ceil(w/step)))
        ys,xs=np.mgrid[0:gh,0:gw];xs=xs.astype(np.float32)*step;ys=ys.astype(np.float32)*step
        dstack=[]
        for sl in ss:
            px,py=_manual_point(sl);dstack.append((xs-px)**2+(ys-py)**2)
        labels=np.argmin(np.stack(dstack,axis=0),axis=0).astype(np.int16);boundary=np.zeros_like(labels,dtype=np.uint8)
        boundary[:,1:]|=(labels[:,1:]!=labels[:,:-1]).astype(np.uint8);boundary[1:,:]|=(labels[1:,:]!=labels[:-1,:]).astype(np.uint8)
        full=cv2.resize(boundary*255,(w,h),interpolation=cv2.INTER_NEAREST)
        base=img.copy();base[full>0]=(0,220,255);canvas,sc=_debug_canvas(base,settings)
        _label_box(canvas,'VORONOI uses immutable MANUAL points. Learned anchors do not move these points.',(24,44),sc,font_scale=0.58)
        for sl in ss:
            px,py=map(float,sl['point']);cv2.circle(canvas,_p((px,py),sc),max(7,int(round(6*sc))),(255,210,0),max(2,int(round(2*sc))))
            _label_box(canvas,sl['local_id'],_p((px+8,py-8),sc),sc)
        cv2.imwrite(str(out_dir/f'{cctv}_manual_voronoi_preview.jpg'),canvas,[int(cv2.IMWRITE_JPEG_QUALITY),95])


def cluster_candidate_centers(
    observations: List[Tuple[float, Detection]], existing_points: Sequence[Sequence[float]],
    total_sample_count: int, min_presence: float, cluster_dist: float, max_std: float,
    existing_dist: float,
) -> List[Dict]:
    clusters: List[Dict] = []
    for t,d in observations:
        cx,cy=box_center(d.box)
        if any(math.hypot(cx-p[0],cy-p[1])<existing_dist for p in existing_points):
            continue
        best=None; best_dist=float('inf')
        for idx,c in enumerate(clusters):
            mx,my=np.mean(c['centers'],axis=0)
            dist=math.hypot(cx-mx,cy-my)
            if dist<cluster_dist and dist<best_dist: best,best_dist=idx,dist
        if best is None:
            clusters.append({'centers':[(cx,cy)],'times':{round(t,2)},'boxes':[d.box]})
        else:
            clusters[best]['centers'].append((cx,cy));clusters[best]['times'].add(round(t,2));clusters[best]['boxes'].append(d.box)
    out=[]
    for c in clusters:
        centers=np.asarray(c['centers'],dtype=np.float32)
        presence=len(c['times'])/max(1,total_sample_count)
        std=float(np.mean(np.std(centers,axis=0))) if len(centers)>1 else 0.0
        if presence<min_presence or std>max_std: continue
        med=np.median(centers,axis=0)
        out.append({'point':[int(round(float(med[0]))),int(round(float(med[1])))],
                    'presence_ratio':round(float(presence),4),'center_std_px':round(std,2),
                    'samples':len(c['times']),'times_sec':sorted(float(x) for x in c['times']),'boxes':c['boxes'],'status':'PENDING'})
    out.sort(key=lambda x:(-x['presence_ratio'],x['center_std_px']))
    return out



def _apply_candidate_override(cctv: str, cand: Dict, settings: Dict) -> Dict:
    c=dict(cand);px,py=map(float,c.get('point',[0,0]));c.setdefault('candidate_kind','NEW');c.setdefault('status','PENDING')
    for ov in settings.get('candidate_overrides',[]) or []:
        if str(ov.get('cctv','')).lower()!=str(cctv).lower():continue
        ox,oy=map(float,ov.get('point',[1e9,1e9]));tol=float(ov.get('tolerance_px',45.0))
        if math.hypot(px-ox,py-oy)>tol:continue
        kind=str(ov.get('kind','')).upper()
        c['candidate_kind']=kind;c['override_reason']=str(ov.get('reason',''))
        if kind=='FALSE':
            c['status']='FALSE_CONFIRMED'
        elif kind=='DUPLICATE':
            c['status']='DUPLICATE_CONFIRMED';c['target_local_id']=str(ov.get('target_local_id',''));c['target_global_id']=str(ov.get('target_global_id',''))
        break
    return c

def learn_slots_and_candidates(video_path: str, rois: Dict[str,Sequence[int]], slots_path: str,
                               settings: Dict, output_dir: str, progress=None) -> Dict:
    out_dir=ensure_dir(output_dir);templates_dir=ensure_dir(out_dir/'templates')
    detector_cfgs=settings.get('detector_by_cctv',{})
    detectors={c:VehicleDetector(detector_cfgs.get(c,settings['detector'])) for c in rois}
    matching_cfg=settings.get('matching',{})
    slots=load_slots(slots_path)
    # v11: never let geometry learned by an older run redefine slot identity.
    # Every Learn starts again from the administrator's manual point.
    _reset_learned_geometry(slots)
    dev_end=float(settings.get('dev_end_sec',900));sample_sec=float(settings.get('learn_sample_sec',2.0))
    from split_protocol import learning_times
    times=learning_times(settings,sample_sec);cap=open_video(video_path)
    by_cctv=defaultdict(list)
    for s in slots: by_cctv[s['cctv']].append(s)
    box_samples=defaultdict(list);center_samples=defaultdict(list);unmatched_obs=defaultdict(list);first_images={}
    try:
        for ti,t in enumerate(times):
            frame=read_frame_at(cap,t)
            for cctv,roi in rois.items():
                image=crop_roi(frame,roi)
                first_images.setdefault(cctv,image.copy())
                dets=detectors[cctv].detect(image)
                assigned,unmatched,_=assign_detections_to_slots(dets,by_cctv.get(cctv,[]),matching_cfg)
                for lid,d in assigned.items():
                    box_samples[lid].append(d.box);center_samples[lid].append(box_center(d.box))
                for d in unmatched: unmatched_obs[cctv].append((t,d))
            if progress and (ti%max(1,int(10/sample_sec))==0 or ti==len(times)-1):
                progress((ti+1)/len(times),f'Learning anchors {format_timestamp(t)} / {format_timestamp(dev_end)}')
    finally: cap.release()

    # First pass learns anchors; then compact visual regions avoid neighboring-slot spill.
    for cctv,ss in by_cctv.items():
        img=first_images.get(cctv)
        if img is None: continue
        for s in ss:
            _learn_slot_geometry(s,center_samples.get(s['local_id'],[]),box_samples.get(s['local_id'],[]),ss,img.shape,matching_cfg)
            x1,y1,x2,y2=clip_box(s['region'],img.shape[1],img.shape[0])
            template_name=f"{s['local_id']}_initial.jpg"
            cv2.imwrite(str(templates_dir/template_name),img[y1:y2,x1:x2])
            s['template_initial']=str(Path('templates')/template_name)
    save_slots(slots_path,slots)

    diagnostics=slot_learning_diagnostics(slots)
    diagnostics.to_csv(out_dir/'slot_learning_diagnostics.csv',index=False,encoding='utf-8-sig')
    warnings=compute_slot_overlap_warnings(slots,matching_cfg)
    warnings.to_csv(out_dir/'slot_overlap_warnings.csv',index=False,encoding='utf-8-sig')
    _draw_overlap_previews(slots,first_images,warnings,out_dir,settings)
    _draw_voronoi_previews(slots,first_images,out_dir,settings)

    candidate_rows=[]; candidates_json={}
    for cctv,obs in unmatched_obs.items():
        existing=[s['point'] for s in slots if s['cctv']==cctv]
        cands=cluster_candidate_centers(obs,existing,len(times),float(settings.get('candidate_min_presence',0.65)),
            float(settings.get('candidate_cluster_dist_px',45)),float(settings.get('candidate_max_std_px',28)),
            float(settings.get('candidate_existing_slot_dist_px',60)))
        resolved=[]
        for i,c in enumerate(cands,start=1):
            c['candidate_id']=f'{cctv}_CAND_{i:03d}';boxes=c.pop('boxes',[])
            if first_images.get(cctv) is not None:
                c['region']=robust_region(boxes,c['point'],first_images[cctv].shape)
            c=_apply_candidate_override(cctv,c,settings)
            resolved.append(c)
        cands=resolved
        candidates_json[cctv]=cands
        for c in cands:
            candidate_rows.append({'candidate_id':c['candidate_id'],'cctv':cctv,'x':c['point'][0],'y':c['point'][1],
                                   'presence_ratio':c['presence_ratio'],'center_std_px':c['center_std_px'],'samples':c['samples'],'status':c['status'],
                                   'candidate_kind':c.get('candidate_kind','NEW'),'target_local_id':c.get('target_local_id',''),'target_global_id':c.get('target_global_id',''),'override_reason':c.get('override_reason','')})
    save_json(out_dir/'candidate_slots.json',candidates_json)
    pd.DataFrame(candidate_rows).to_csv(out_dir/'candidate_slots.csv',index=False,encoding='utf-8-sig')

    gt_review=[]
    for cctv,img in first_images.items():
        preview,sc=_debug_canvas(img,settings)
        _label_box(preview,'MANUAL=cyan ring | LEARNED=green dot | NEW=orange | FALSE=red | DUPLICATE=purple',(24,44),sc,font_scale=0.58)
        for sl in [z for z in slots if z['cctv']==cctv]:
            mx,my=map(float,sl['point']); ax,ay=_effective_anchor(sl)
            cv2.circle(preview,_p((mx,my),sc),max(7,int(round(7*sc))),(255,210,0),max(2,int(round(2*sc))))
            if math.hypot(ax-mx,ay-my)>1.0:cv2.line(preview,_p((mx,my),sc),_p((ax,ay),sc),(255,255,255),max(1,int(round(sc))),cv2.LINE_AA)
            acolor=(0,220,0) if bool(sl.get('anchor_trusted',False)) else (150,150,150)
            cv2.circle(preview,_p((ax,ay),sc),max(5,int(round(4*sc))),acolor,-1)
            _label_box(preview,f"{sl['local_id']} M | A:{'OK' if sl.get('anchor_trusted') else 'MANUAL'}",_p((mx+8,my-8),sc),sc)
        for c in candidates_json.get(cctv,[]):
            px,py=map(float,c['point']);kind=str(c.get('candidate_kind','NEW')).upper()
            color=(0,0,255) if kind=='FALSE' else ((180,70,220) if kind=='DUPLICATE' else (0,165,255))
            cv2.circle(preview,_p((px,py),sc),max(11,int(round(9*sc))),color,max(3,int(round(2*sc))))
            suffix=''
            if kind=='DUPLICATE':suffix=f" -> {c.get('target_local_id','?')} / {c.get('target_global_id','?')}"
            elif kind=='FALSE':suffix=' (confirmed fixed structure)'
            _label_box(preview,f"{c['candidate_id']} | {kind}{suffix}",_p((px+10,py+18),sc),sc)
            if kind=='DUPLICATE':
                gt_review.append(f"{c['candidate_id']} ({cctv}) is confirmed as the same physical space as {c.get('target_local_id','?')} / {c.get('target_global_id','?')}. Review the {cctv}_count ground-truth column when this vehicle is visible. Do NOT add +1 to global occupied-space count because it is a duplicate view.")
        cv2.imwrite(str(out_dir/f'{cctv}_candidate_preview.jpg'),preview,[int(cv2.IMWRITE_JPEG_QUALITY),95])
        cv2.imwrite(str(out_dir/f'{cctv}_slot_configuration_diagnostics.jpg'),preview,[int(cv2.IMWRITE_JPEG_QUALITY),95])
    if gt_review and bool((settings.get('candidate_policy',{}) or {}).get('write_gt_review_note',True)):
        (out_dir/'GT_REVIEW_REQUIRED.txt').write_text('Parking Slot Engine v15 ground-truth review\n\n'+'\n'.join('- '+x for x in gt_review)+'\n\nGround truth is never auto-edited from model predictions. Confirm the affected timestamps manually before changing the CSV.\n',encoding='utf-8')
        review_rows=[]
        for cc,clist in candidates_json.items():
            for c in clist:
                if str(c.get('candidate_kind','')).upper()!='DUPLICATE':continue
                for tt in c.get('times_sec',[]) or []:
                    review_rows.append({'candidate_id':c.get('candidate_id',''),'cctv':cc,'time_sec':float(tt),'nearest_10s_time_sec':float(round(float(tt)/10.0)*10.0),
                                        'target_local_id':c.get('target_local_id',''),'target_global_id':c.get('target_global_id',''),'action':'REVIEW_LOCAL_CCTV_COUNT_ONLY_GLOBAL_COUNT_UNCHANGED'})
        pd.DataFrame(review_rows).to_csv(out_dir/'gt_review_candidate_times.csv',index=False,encoding='utf-8-sig')
    return {'slots':slots,'candidates':candidates_json,'warnings':warnings.to_dict('records'),'diagnostics':diagnostics.to_dict('records'),'learn_sample_count':len(times),'output_dir':str(out_dir)}


def _resolve_template_path(slot: Dict, learn_dir: Path) -> Optional[Path]:
    rel=slot.get('template_initial')
    if not rel: return None
    p=learn_dir/rel
    return p if p.exists() else None


# -----------------------------------------------------------------------------
# Evidence extraction and state engine
# -----------------------------------------------------------------------------

def extract_evidence(video_path: str, rois: Dict[str,Sequence[int]], slots_path: str, settings: Dict,
                     learn_dir: str, output_dir: str, progress=None) -> pd.DataFrame:
    out_dir=ensure_dir(output_dir);slots=load_slots(slots_path)
    detector_cfgs=settings.get('detector_by_cctv',{})
    detectors={c:VehicleDetector(detector_cfgs.get(c,settings['detector'])) for c in rois}
    matching_cfg=settings.get('matching',{})
    end_sec=float(settings.get('eval_end_sec',1230));sample_sec=float(settings.get('evidence_sample_sec',2.0))
    times=np.arange(0.0,end_sec+1e-6,sample_sec).tolist();by_cctv=defaultdict(list)
    for s in slots:by_cctv[s['cctv']].append(s)
    templates={};learn_path=Path(learn_dir)
    for s in slots:
        p=_resolve_template_path(s,learn_path);templates[s['local_id']]=cv2.imread(str(p)) if p else None
    cap=open_video(video_path);rows=[];summary=[]
    try:
        for ti,t in enumerate(times):
            frame=read_frame_at(cap,t)
            for cctv,roi in rois.items():
                img=crop_roi(frame,roi);cslots=by_cctv.get(cctv,[]);dets=detectors[cctv].detect(img)
                assigned,unmatched,ameta=assign_detections_to_slots(dets,cslots,matching_cfg)
                summary.append({'time_sec':round(float(t),3),'timestamp':format_timestamp(t),'cctv':cctv,
                                'deduplicated_detections':len(dets),'assigned_detections':len(assigned),'unmatched_detections':len(unmatched)})
                for s in cslots:
                    lid=s['local_id'];d=assigned.get(lid);meta=ameta.get(lid,{})
                    region=s.get('region') or robust_region([],s['point'],img.shape)
                    x1,y1,x2,y2=clip_box(region,img.shape[1],img.shape[0]);current=img[y1:y2,x1:x2]
                    diff=normalized_visual_diff(current,templates.get(lid))
                    match_score=float(meta.get('match_score',0.0))
                    det_occ_score=(0.55+0.45*(0.60*float(d.conf)+0.40*match_score)) if d else 0.0
                    dc=box_center(d.box) if d else (np.nan,np.nan)
                    rows.append({'time_sec':round(float(t),3),'timestamp':format_timestamp(t),'cctv':cctv,'local_id':lid,
                        'global_id':s.get('global_id',lid),'initial_state':s.get('initial_state','UNKNOWN'),
                        'detected':int(d is not None),'det_conf':float(d.conf) if d else 0.0,
                        'det_center_x':float(dc[0]),'det_center_y':float(dc[1]),
                        'assignment_cost':float(meta.get('assignment_cost',np.nan)),'match_score':match_score,
                        'match_distance_px':float(meta.get('distance_px',np.nan)),
                        'manual_distance_px':float(meta.get('manual_distance_px',np.nan)),
                        'anchor_distance_px':float(meta.get('anchor_distance_px',np.nan)),
                        'voronoi_best_distance_px':float(meta.get('voronoi_best_distance_px',np.nan)),
                        'voronoi_margin_px':float(meta.get('voronoi_margin_px',np.nan)),
                        'voronoi_allowed':int(meta.get('voronoi_allowed',0)) if d else 0,
                        'anchor_trusted':int(bool(s.get('anchor_trusted',False))),
                        'anchor_shift_px':float(s.get('anchor_shift_px',0.0) or 0.0),
                        'region_iou':float(meta.get('region_iou',0.0)),
                        'det_occupancy_score':float(det_occ_score),'visual_diff_initial':float(diff)})
            if progress and (ti%max(1,int(10/sample_sec))==0 or ti==len(times)-1):
                progress((ti+1)/len(times),f'Extracting 1:1 slot evidence {format_timestamp(t)} / {format_timestamp(end_sec)}')
    finally:cap.release()
    df=pd.DataFrame(rows);df.to_csv(out_dir/'slot_evidence.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(out_dir/'frame_detection_summary.csv',index=False,encoding='utf-8-sig')
    return df


def run_state_engine(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    enter_hits=int(params['enter_hits']);empty_misses=int(params['empty_misses']);visual_thr=float(params['occupied_visual_diff_threshold'])
    fusion=str(params.get('global_merge','ANY')).upper();global_thr=float(params.get('global_confidence_threshold',0.60))
    state={};hit_streak=defaultdict(int);miss_streak=defaultdict(int);prev_local={};prev_global={}
    slot_meta=evidence.sort_values('time_sec').groupby('local_id').first()
    for lid,r in slot_meta.iterrows():
        init=str(r.get('initial_state','UNKNOWN')).upper();state[lid]='OCCUPIED' if init=='OCCUPIED' else 'EMPTY';prev_local[lid]=state[lid]
    local_rows=[];global_rows=[];transitions=[]
    for t,group in evidence.groupby('time_sec',sort=True):
        now=[]
        for _,r in group.iterrows():
            lid=r['local_id'];detected=bool(r['detected']);vis_diff=float(r['visual_diff_initial']);initial=str(r.get('initial_state','UNKNOWN')).upper()
            visual_occ=(initial=='EMPTY' and vis_diff>=visual_thr);occupied_evidence=detected or visual_occ
            if occupied_evidence:
                hit_streak[lid]+=1;miss_streak[lid]=0
                if state[lid]!='OCCUPIED' and hit_streak[lid]>=enter_hits: state[lid]='OCCUPIED'
            else:
                hit_streak[lid]=0
                same_initial_occ=(initial=='OCCUPIED' and vis_diff<=visual_thr)
                if state[lid]=='OCCUPIED' and same_initial_occ:
                    miss_streak[lid]=0
                else:
                    miss_streak[lid]+=1
                    if state[lid]=='OCCUPIED' and miss_streak[lid]>=empty_misses: state[lid]='EMPTY'

            det_score=float(r.get('det_occupancy_score',0.0))
            if detected:
                occ_score=max(0.55,min(0.99,det_score))
            elif initial=='EMPTY' and vis_diff>=visual_thr:
                occ_score=min(0.85,0.55+0.30*min(1.0,(vis_diff-visual_thr)/max(0.02,visual_thr)))
            elif initial=='OCCUPIED' and vis_diff<=visual_thr:
                occ_score=0.76
            else:
                occ_score=max(0.05,0.50-0.40*min(1.0,miss_streak[lid]/max(1,empty_misses)))
            if state[lid]=='OCCUPIED': occ_score=max(0.51,occ_score)
            else: occ_score=min(0.49,occ_score)

            row={'time_sec':t,'timestamp':r['timestamp'],'cctv':r['cctv'],'local_id':lid,'global_id':r['global_id'],
                 'detected':int(detected),'det_conf':float(r['det_conf']),'match_score':float(r.get('match_score',0.0)),
                 'visual_diff_initial':vis_diff,'state':state[lid],'occupied_score':float(occ_score),
                 'hit_streak':hit_streak[lid],'miss_streak':miss_streak[lid]}
            local_rows.append(row);now.append(row)
            if prev_local.get(lid)!=state[lid]:
                transitions.append({'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,'global_id':r['global_id'],
                                    'from_state':prev_local.get(lid,''),'to_state':state[lid],'cctv':r['cctv']})
                prev_local[lid]=state[lid]

        by_global=defaultdict(list)
        for x in now:by_global[x['global_id']].append(x)
        for gid,items in by_global.items():
            occ=sum(1 for x in items if x['state']=='OCCUPIED');scores=[float(x['occupied_score']) for x in items]
            if fusion=='MAJORITY':
                global_occ=occ>=math.ceil(len(items)/2);gscore=float(np.mean(scores))
            elif fusion=='CONF_MAX':
                gscore=max(scores) if scores else 0.0;global_occ=gscore>=global_thr
            elif fusion=='CONF_MEAN':
                gscore=float(np.mean(scores)) if scores else 0.0;global_occ=gscore>=global_thr
            else: # ANY - conservative against false empty
                global_occ=occ>0;gscore=max(scores) if scores else 0.0
            gstate='OCCUPIED' if global_occ else 'EMPTY'
            grow={'time_sec':t,'timestamp':items[0]['timestamp'],'global_id':gid,'state':gstate,'global_score':gscore,
                  'source_local_slots':';'.join(x['local_id'] for x in items),'occupied_votes':occ,'total_votes':len(items)}
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid]!=gstate:
                transitions.append({'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,'global_id':gid,
                                    'from_state':prev_global[gid],'to_state':gstate,'cctv':'MULTI'})
            prev_global[gid]=gstate
    return pd.DataFrame(local_rows),pd.DataFrame(global_rows),pd.DataFrame(transitions)


def evaluate_state_output(global_df: pd.DataFrame, gt: pd.DataFrame, dev_end_sec: float) -> Dict:
    pred=global_df.copy();pred['occupied_pred']=(pred['state']=='OCCUPIED').astype(int)
    counts=pred.groupby('time_sec',as_index=False)['occupied_pred'].sum()
    gt2=gt[['time_sec','timestamp','ground_truth_occupied_space_count']].copy()
    merged=gt2.merge(counts,on='time_sec',how='left');merged['occupied_pred']=merged['occupied_pred'].fillna(0).astype(int)
    merged['error']=merged['occupied_pred']-merged['ground_truth_occupied_space_count'].astype(int);merged['abs_error']=merged['error'].abs()
    merged['split']=np.where(merged['time_sec']<dev_end_sec,'DEV','TEST')
    def metrics(part):
        if len(part)==0:return {'N':0,'exact_rate':0.0,'mae':0.0,'max_abs_error':0,'false_empty_bias_rate':0.0,'over_rate':0.0}
        return {'N':int(len(part)),'exact_rate':float((part['error']==0).mean()),'mae':float(part['abs_error'].mean()),
                'max_abs_error':int(part['abs_error'].max()),'false_empty_bias_rate':float((part['error']<0).mean()),
                'over_rate':float((part['error']>0).mean())}
    return {'timeseries':merged,'DEV':metrics(merged[merged['split']=='DEV']),'TEST':metrics(merged[merged['split']=='TEST'])}


# -----------------------------------------------------------------------------
# Optional slot-level GT: initial state + change events only
# -----------------------------------------------------------------------------

def make_slot_gt_template(slots: Sequence[Dict], output_path: str | Path) -> Path:
    groups=defaultdict(list)
    for s in slots:groups[str(s.get('global_id') or s['local_id'])].append(s)
    rows=[]
    for gid in sorted(groups):
        # Conservative initial merge: if any camera says occupied, initialize occupied.
        st='OCCUPIED' if any(str(x.get('initial_state','EMPTY')).upper()=='OCCUPIED' for x in groups[gid]) else 'EMPTY'
        rows.append({'timestamp':'0:00','global_id':gid,'state':st,'notes':'initial; add rows only when this slot changes'})
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig');return p


def load_slot_gt_events(path: str | Path) -> pd.DataFrame:
    from utils import parse_timestamp
    df=pd.read_csv(path,encoding='utf-8-sig')
    required={'timestamp','global_id','state'}
    if not required.issubset(df.columns):raise ValueError('Slot GT events CSV requires timestamp, global_id, state columns.')
    df=df.copy();df['time_sec']=df['timestamp'].apply(parse_timestamp);df['state']=df['state'].astype(str).str.upper()
    if not set(df['state']).issubset({'OCCUPIED','EMPTY'}):raise ValueError('Slot GT state must be OCCUPIED or EMPTY.')
    return df.sort_values(['global_id','time_sec']).reset_index(drop=True)


def evaluate_slot_level(global_df: pd.DataFrame, slot_gt_events: pd.DataFrame, gt_times: Sequence[float], dev_end_sec: float) -> Dict:
    gids=sorted(set(slot_gt_events['global_id'].astype(str)))
    events={g:slot_gt_events[slot_gt_events['global_id'].astype(str)==g].sort_values('time_sec') for g in gids}
    pred_lookup={(float(r.time_sec),str(r.global_id)):str(r.state) for r in global_df.itertuples()}
    rows=[]
    for t in sorted(float(x) for x in gt_times):
        for gid in gids:
            ev=events[gid];past=ev[ev['time_sec']<=t]
            if past.empty:continue
            gt_state=str(past.iloc[-1]['state'])
            pred_state=pred_lookup.get((t,gid))
            if pred_state is None:
                # global_df is usually 2-sec resolution; exact 10-sec points should exist.
                sub=global_df[(global_df['global_id'].astype(str)==gid)&(global_df['time_sec']<=t)]
                pred_state=str(sub.iloc[-1]['state']) if not sub.empty else 'EMPTY'
            split='DEV' if t<dev_end_sec else 'TEST'
            rows.append({'time_sec':t,'timestamp':format_timestamp(t),'global_id':gid,'gt_state':gt_state,'pred_state':pred_state,
                         'correct':int(gt_state==pred_state),'false_empty':int(gt_state=='OCCUPIED' and pred_state=='EMPTY'),
                         'false_occupied':int(gt_state=='EMPTY' and pred_state=='OCCUPIED'),'split':split})
    df=pd.DataFrame(rows)
    def metrics(part):
        if part.empty:return {'N':0,'slot_accuracy':0.0,'false_empty_rate':0.0,'false_occupied_rate':0.0,'all_slots_exact_time_rate':0.0}
        pertime=part.groupby('time_sec')['correct'].min()
        return {'N':int(len(part)),'slot_accuracy':float(part['correct'].mean()),
                'false_empty_rate':float(part['false_empty'].mean()),'false_occupied_rate':float(part['false_occupied'].mean()),
                'all_slots_exact_time_rate':float(pertime.mean())}
    by_slot=(df.groupby(['split','global_id']) if not df.empty else None)
    by_slot_df=(by_slot.agg(N=('correct','size'),accuracy=('correct','mean'),false_empty=('false_empty','mean'),false_occupied=('false_occupied','mean')).reset_index()
                if by_slot is not None else pd.DataFrame())
    return {'rows':df,'DEV':metrics(df[df['split']=='DEV']),'TEST':metrics(df[df['split']=='TEST']),'by_slot':by_slot_df}


# -----------------------------------------------------------------------------
# Parameter search and outputs
# -----------------------------------------------------------------------------

def _state_param_combos(search: Dict) -> List[Dict]:
    combos=[];seen=set()
    for enter,misses,vthr,fusion,gthr in itertools.product(
        search.get('enter_hits',[1,2]),search.get('empty_misses',[2,3,4]),
        search.get('occupied_visual_diff_threshold',[0.08,0.12,0.16]),
        search.get('global_merge',['ANY']),search.get('global_confidence_threshold',[0.60])):
        # threshold is irrelevant to ANY/MAJORITY, collapse duplicates.
        effective_thr=float(gthr) if str(fusion).upper().startswith('CONF_') else 0.0
        key=(int(enter),int(misses),float(vthr),str(fusion).upper(),effective_thr)
        if key in seen:continue
        seen.add(key)
        combos.append({'enter_hits':int(enter),'empty_misses':int(misses),'occupied_visual_diff_threshold':float(vthr),
                       'global_merge':str(fusion).upper(),'global_confidence_threshold':effective_thr})
    return combos


def search_state_parameters(evidence: pd.DataFrame, gt: pd.DataFrame, settings: Dict, output_dir: str,
                            progress=None, slot_gt_events_path: Optional[str]=None) -> Dict:
    out_dir=ensure_dir(output_dir);search=settings['state_search'];combos=_state_param_combos(search)
    dev_end=float(settings.get('dev_end_sec',900));leaderboard=[];best=None;best_key=None;best_outputs=None
    slot_events=None
    if slot_gt_events_path and Path(slot_gt_events_path).exists():slot_events=load_slot_gt_events(slot_gt_events_path)
    gt_times=gt['time_sec'].tolist()
    for i,params in enumerate(combos):
        local_df,global_df,transitions=run_state_engine(evidence,params);ev=evaluate_state_output(global_df,gt,dev_end);d=ev['DEV']
        slot_eval=evaluate_slot_level(global_df,slot_events,gt_times,dev_end) if slot_events is not None else None
        if slot_eval is not None:
            sd=slot_eval['DEV']
            # Commercial priority: all slots simultaneously correct, then false-empty, then slot accuracy, then count metrics.
            key=(-sd['all_slots_exact_time_rate'],sd['false_empty_rate'],-sd['slot_accuracy'],-d['exact_rate'],d['mae'],d['max_abs_error'])
        else:
            key=(-d['exact_rate'],d['mae'],d['false_empty_bias_rate'],d['max_abs_error'])
        row={**params,'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],'dev_under_rate':d['false_empty_bias_rate'],
             'dev_over_rate':d['over_rate'],'dev_max_abs_error':d['max_abs_error']}
        if slot_eval is not None:
            row.update({'dev_slot_accuracy':slot_eval['DEV']['slot_accuracy'],'dev_slot_false_empty_rate':slot_eval['DEV']['false_empty_rate'],
                        'dev_all_slots_exact_time_rate':slot_eval['DEV']['all_slots_exact_time_rate']})
        leaderboard.append(row)
        if best is None or key<best_key:
            best,best_key=params,key;best_outputs=(local_df,global_df,transitions,ev,slot_eval)
        if progress and (i%max(1,len(combos)//20)==0 or i==len(combos)-1):progress((i+1)/len(combos),f'State search {i+1}/{len(combos)}')

    sort_cols=['dev_exact_rate','dev_MAE','dev_under_rate','dev_max_abs_error'];asc=[False,True,True,True]
    if slot_events is not None:
        sort_cols=['dev_all_slots_exact_time_rate','dev_slot_false_empty_rate','dev_slot_accuracy']+sort_cols
        asc=[False,True,False]+asc
    board=pd.DataFrame(leaderboard).sort_values(sort_cols,ascending=asc).reset_index(drop=True)
    board.to_csv(out_dir/'state_search_leaderboard.csv',index=False,encoding='utf-8-sig')
    local_df,global_df,transitions,ev,slot_eval=best_outputs
    local_df.to_csv(out_dir/'selected_slot_timeseries.csv',index=False,encoding='utf-8-sig')
    global_df.to_csv(out_dir/'selected_global_slot_timeseries.csv',index=False,encoding='utf-8-sig')
    transitions.to_csv(out_dir/'selected_state_transitions.csv',index=False,encoding='utf-8-sig')
    ev['timeseries'].to_csv(out_dir/'selected_count_timeseries.csv',index=False,encoding='utf-8-sig');save_json(out_dir/'selected_state_params.json',best)
    pd.DataFrame([{'split':sp,**ev[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'metrics_dev_test.csv',index=False,encoding='utf-8-sig')
    errors=ev['timeseries'][(ev['timeseries']['split']=='TEST')&(ev['timeseries']['error']!=0)].copy();errors.to_csv(out_dir/'test_error_cases.csv',index=False,encoding='utf-8-sig')

    if slot_eval is not None:
        slot_eval['rows'].to_csv(out_dir/'slot_level_comparison.csv',index=False,encoding='utf-8-sig')
        slot_eval['by_slot'].to_csv(out_dir/'slot_level_by_slot.csv',index=False,encoding='utf-8-sig')
        slot_eval['rows'][(slot_eval['rows']['split']=='TEST')&(slot_eval['rows']['correct']==0)].to_csv(out_dir/'slot_level_test_errors.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{'split':sp,**slot_eval[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'slot_level_metrics.csv',index=False,encoding='utf-8-sig')

    with (out_dir/'REPORT.txt').open('w',encoding='utf-8') as f:
        f.write('Parking Slot State Engine v9 - manual Voronoi + constrained anchor + confidence fusion\n\n')
        f.write('SELECTED PARAMETERS\n');[f.write(f'{k}={v}\n') for k,v in best.items()]
        for sp in ['DEV','TEST']:
            m=ev[sp];f.write(f'\n[{sp} COUNT]\nN={m["N"]}\nExact match rate={m["exact_rate"]*100:.2f}%\nMAE={m["mae"]:.4f}\n')
            f.write(f'Under-count rate={m["false_empty_bias_rate"]*100:.2f}%\nOver-count rate={m["over_rate"]*100:.2f}%\nMax absolute error={m["max_abs_error"]}\n')
            if slot_eval is not None:
                sm=slot_eval[sp];f.write(f'[{sp} SLOT]\nSlot accuracy={sm["slot_accuracy"]*100:.2f}%\nFalse-empty rate={sm["false_empty_rate"]*100:.2f}%\n')
                f.write(f'False-occupied rate={sm["false_occupied_rate"]*100:.2f}%\nAll-slots-exact time rate={sm["all_slots_exact_time_rate"]*100:.2f}%\n')

    files=['REPORT.txt','metrics_dev_test.csv','state_search_leaderboard.csv','selected_state_params.json','selected_count_timeseries.csv',
           'selected_slot_timeseries.csv','selected_global_slot_timeseries.csv','selected_state_transitions.csv','test_error_cases.csv']
    if slot_eval is not None:files+=['slot_level_metrics.csv','slot_level_by_slot.csv','slot_level_comparison.csv','slot_level_test_errors.csv']
    zip_paths(out_dir/'UPLOAD_TO_CHATGPT.zip',[(out_dir/x,x) for x in files])
    return {'best_params':best,'evaluation':ev,'slot_evaluation':slot_eval,'leaderboard':board,'output_dir':str(out_dir)}

# =============================================================================
# v10 temporal engine override
# Rolling 10-second history + online vehicle tracking + maneuver-aware state logic.
# These definitions intentionally override the v9 functions above while preserving
# the public API used by app.py.
# =============================================================================
from collections import Counter, deque


class _OnlineCentroidTracker:
    """Small online tracker for fixed-CCTV vehicle detections.

    It is deliberately dependency-free: detections are associated with existing
    tracks by Hungarian matching using center distance + IoU. The tracker never
    looks into future frames, so its output is valid for a real-time deployment.
    """
    def __init__(self, cfg: Dict):
        self.max_gap = float(cfg.get('tracker_max_gap_sec', 3.0))
        self.max_px = float(cfg.get('tracker_match_max_px', 110.0))
        self.min_iou = float(cfg.get('tracker_min_iou', 0.02))
        self.iou_weight = float(cfg.get('tracker_iou_weight', 0.25))
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    def _new_track(self, det: Detection, t: float) -> Dict:
        tid = self.next_id
        self.next_id += 1
        c = box_center(det.box)
        tr = {
            'id': tid,
            'first_t': float(t),
            'last_t': float(t),
            'center': (float(c[0]), float(c[1])),
            'box': list(det.box),
            'history': deque([(float(t), float(c[0]), float(c[1]))], maxlen=64),
            'speed_history': deque([], maxlen=32),
        }
        self.tracks[tid] = tr
        return tr

    @staticmethod
    def _track_meta(tr: Dict, t: float) -> Dict:
        hist = list(tr['history'])
        recent = [x for x in hist if float(t) - x[0] <= 10.0 + 1e-6]
        if not recent:
            recent = hist[-1:]
        xs = [x[1] for x in recent]
        ys = [x[2] for x in recent]
        motion_span = 0.0
        if len(recent) >= 2:
            x0, y0 = recent[0][1], recent[0][2]
            motion_span = max(math.hypot(x-x0, y-y0) for _, x, y in recent)
        speeds = list(tr['speed_history'])
        recent_speeds = [s for tt, s in speeds if float(t) - tt <= 3.0 + 1e-6]
        speed = float(np.median(recent_speeds)) if recent_speeds else 0.0
        return {
            'track_id': int(tr['id']),
            'track_age_sec': max(0.0, float(t) - float(tr['first_t'])),
            'track_speed_px_s': speed,
            'track_motion_span_px': float(motion_span),
        }

    def update(self, dets: Sequence[Detection], t: float) -> Dict[int, Dict]:
        dets = list(dets)
        t = float(t)
        # Drop stale tracks from active association. A longer history is unnecessary
        # because the temporal state engine maintains its own rolling history.
        active_ids = [
            tid for tid, tr in self.tracks.items()
            if t - float(tr['last_t']) <= self.max_gap + 1e-6
        ]
        result: Dict[int, Dict] = {}

        if not dets:
            return result
        if not active_ids:
            for di, d in enumerate(dets):
                tr = self._new_track(d, t)
                result[di] = self._track_meta(tr, t)
            return result

        n_t, n_d = len(active_ids), len(dets)
        unmatched_cost = 1.25
        invalid_cost = 5.0
        cost = np.full((n_t, n_d + n_t), unmatched_cost, dtype=np.float64)
        for ti, tid in enumerate(active_ids):
            tr = self.tracks[tid]
            tcx, tcy = tr['center']
            gap = max(0.001, t - float(tr['last_t']))
            gate = self.max_px * (1.0 + 0.20 * max(0.0, gap - 1.0))
            for di, d in enumerate(dets):
                dcx, dcy = box_center(d.box)
                dist = math.hypot(dcx-tcx, dcy-tcy)
                iou = box_iou(tr['box'], d.box)
                if dist > gate and iou < self.min_iou:
                    cost[ti, di] = invalid_cost
                else:
                    distance_term = min(2.0, dist / max(1.0, gate))
                    cost[ti, di] = distance_term + self.iou_weight * (1.0 - iou)
            for k in range(n_t):
                cost[ti, n_d+k] = unmatched_cost + (0.0001 if k != ti else 0.0)

        pairs = _hungarian_minimize(cost)
        used_dets = set()
        for ti, col in pairs:
            if ti >= n_t or col >= n_d:
                continue
            if float(cost[ti, col]) >= unmatched_cost:
                continue
            tid = active_ids[ti]
            tr = self.tracks[tid]
            d = dets[col]
            oldx, oldy = tr['center']
            newx, newy = box_center(d.box)
            dt = max(0.001, t - float(tr['last_t']))
            speed = math.hypot(newx-oldx, newy-oldy) / dt
            tr['last_t'] = t
            tr['center'] = (float(newx), float(newy))
            tr['box'] = list(d.box)
            tr['history'].append((t, float(newx), float(newy)))
            tr['speed_history'].append((t, float(speed)))
            result[col] = self._track_meta(tr, t)
            used_dets.add(col)

        for di, d in enumerate(dets):
            if di in used_dets:
                continue
            tr = self._new_track(d, t)
            result[di] = self._track_meta(tr, t)
        return result


def extract_evidence(video_path: str, rois: Dict[str,Sequence[int]], slots_path: str, settings: Dict,
                     learn_dir: str, output_dir: str, progress=None) -> pd.DataFrame:
    """v10 evidence extraction at 1 FPS by default, with online track IDs.

    The tracker uses only current/past frames. Each slot row therefore carries enough
    information for the rolling temporal engine to distinguish stable parking from
    forward/backward parking maneuvers and brief detector misses.
    """
    out_dir = ensure_dir(output_dir)
    slots = load_slots(slots_path)
    detector_cfgs = settings.get('detector_by_cctv', {})
    detectors = {c: VehicleDetector(detector_cfgs.get(c, settings['detector'])) for c in rois}
    matching_cfg = settings.get('matching', {})
    temporal_cfg = settings.get('temporal', {})
    trackers = {c: _OnlineCentroidTracker(temporal_cfg) for c in rois}
    end_sec = float(settings.get('eval_end_sec', 1230))
    sample_sec = float(settings.get('evidence_sample_sec', 1.0))
    times = np.arange(0.0, end_sec + 1e-6, sample_sec).tolist()
    by_cctv = defaultdict(list)
    for s in slots:
        by_cctv[s['cctv']].append(s)

    templates = {}
    learn_path = Path(learn_dir)
    for s in slots:
        p = _resolve_template_path(s, learn_path)
        templates[s['local_id']] = cv2.imread(str(p)) if p else None

    cap = open_video(video_path)
    rows = []
    summary = []
    track_rows = []
    try:
        for ti, t in enumerate(times):
            frame = read_frame_at(cap, t)
            for cctv, roi in rois.items():
                img = crop_roi(frame, roi)
                cslots = by_cctv.get(cctv, [])
                dets = detectors[cctv].detect(img)
                track_meta = trackers[cctv].update(dets, float(t))
                assigned, unmatched, ameta = assign_detections_to_slots(dets, cslots, matching_cfg)
                assigned_by_index = {
                    int(meta.get('det_index')): lid
                    for lid, meta in ameta.items()
                    if meta.get('det_index') is not None
                }
                global_by_local = {str(s['local_id']): str(s.get('global_id', s['local_id'])) for s in cslots}
                summary.append({
                    'time_sec': round(float(t), 3), 'timestamp': format_timestamp(t), 'cctv': cctv,
                    'deduplicated_detections': len(dets), 'assigned_detections': len(assigned),
                    'unmatched_detections': len(unmatched),
                    'active_track_detections': len(track_meta),
                })
                for di, d in enumerate(dets):
                    tm = track_meta.get(di, {})
                    lid = assigned_by_index.get(di, '')
                    dc = box_center(d.box)
                    track_rows.append({
                        'time_sec': round(float(t),3), 'timestamp': format_timestamp(t), 'cctv': cctv,
                        'track_id': int(tm.get('track_id', -1)),
                        'track_age_sec': float(tm.get('track_age_sec', 0.0)),
                        'track_speed_px_s': float(tm.get('track_speed_px_s', 0.0)),
                        'track_motion_span_px': float(tm.get('track_motion_span_px', 0.0)),
                        'det_center_x': float(dc[0]), 'det_center_y': float(dc[1]),
                        'det_conf': float(d.conf), 'assigned_local_id': lid,
                        'assigned_global_id': global_by_local.get(lid, '') if lid else '',
                    })
                for s in cslots:
                    lid = s['local_id']
                    d = assigned.get(lid)
                    meta = ameta.get(lid, {})
                    region = s.get('region') or robust_region([], s['point'], img.shape)
                    x1,y1,x2,y2 = clip_box(region, img.shape[1], img.shape[0])
                    current = img[y1:y2,x1:x2]
                    diff = normalized_visual_diff(current, templates.get(lid))
                    match_score = float(meta.get('match_score', 0.0))
                    det_occ_score = (0.55 + 0.45*(0.60*float(d.conf)+0.40*match_score)) if d else 0.0
                    dc = box_center(d.box) if d else (np.nan, np.nan)
                    det_index = int(meta.get('det_index', -1)) if d else -1
                    tm = track_meta.get(det_index, {}) if d else {}
                    rows.append({
                        'time_sec': round(float(t),3), 'timestamp': format_timestamp(t), 'cctv': cctv,
                        'local_id': lid, 'global_id': s.get('global_id', lid),
                        'initial_state': s.get('initial_state','UNKNOWN'),
                        'detected': int(d is not None), 'det_conf': float(d.conf) if d else 0.0,
                        'det_center_x': float(dc[0]), 'det_center_y': float(dc[1]),
                        'track_id': int(tm.get('track_id', -1)) if d else -1,
                        'track_age_sec': float(tm.get('track_age_sec', 0.0)) if d else 0.0,
                        'track_speed_px_s': float(tm.get('track_speed_px_s', 0.0)) if d else 0.0,
                        'track_motion_span_px': float(tm.get('track_motion_span_px', 0.0)) if d else 0.0,
                        'assignment_cost': float(meta.get('assignment_cost', np.nan)),
                        'match_score': match_score,
                        'match_distance_px': float(meta.get('distance_px', np.nan)),
                        'manual_distance_px': float(meta.get('manual_distance_px', np.nan)),
                        'anchor_distance_px': float(meta.get('anchor_distance_px', np.nan)),
                        'voronoi_best_distance_px': float(meta.get('voronoi_best_distance_px', np.nan)),
                        'voronoi_margin_px': float(meta.get('voronoi_margin_px', np.nan)),
                        'voronoi_allowed': int(meta.get('voronoi_allowed', 0)) if d else 0,
                        'anchor_trusted': int(bool(s.get('anchor_trusted', False))),
                        'anchor_shift_px': float(s.get('anchor_shift_px', 0.0) or 0.0),
                        'region_iou': float(meta.get('region_iou', 0.0)),
                        'det_occupancy_score': float(det_occ_score),
                        'visual_diff_initial': float(diff),
                    })
            if progress and (ti % max(1, int(10/sample_sec)) == 0 or ti == len(times)-1):
                progress((ti+1)/len(times), f'Extracting tracked 1 FPS evidence {format_timestamp(t)} / {format_timestamp(end_sec)}')
    finally:
        cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir/'slot_evidence.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(out_dir/'frame_detection_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(track_rows).to_csv(out_dir/'vehicle_tracks.csv', index=False, encoding='utf-8-sig')
    return df


def _appearance_occupied(initial_state: str, visual_diff: float, threshold: float) -> bool:
    init = str(initial_state).upper()
    # If the slot began EMPTY, moving away from the initial appearance supports occupancy.
    # If it began OCCUPIED, staying similar to the initial appearance supports occupancy.
    if init == 'EMPTY':
        return float(visual_diff) >= float(threshold)
    return float(visual_diff) <= float(threshold)


def run_state_engine(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """v10 maneuver-aware online temporal state engine.

    Every decision at time t uses only evidence from [t-window, t]. A tracked vehicle
    must settle predominantly on one slot before an EMPTY slot becomes OCCUPIED.
    Existing occupied slots are kept through short detector misses and parking
    maneuvers until the 10-second history and appearance both support EMPTY.
    """
    window_sec = float(params.get('temporal_window_sec', 10.0))
    recent_sec = float(params.get('recent_window_sec', 3.0))
    entry_ratio = float(params.get('entry_hit_ratio', 0.50))
    exit_ratio = float(params.get('exit_hit_ratio', 0.10))
    stable_track_ratio_req = float(params.get('stable_track_ratio', 0.65))
    visual_thr = float(params.get('occupied_visual_diff_threshold', 0.12))
    stationary_speed = float(params.get('stationary_speed_px_s', 14.0))
    stationary_span = float(params.get('stationary_motion_span_px', 32.0))
    max_switches = int(params.get('max_track_switches_for_entry', 1))
    fusion = str(params.get('global_merge', 'ANY')).upper()
    global_thr = float(params.get('global_confidence_threshold', 0.60))

    evidence = evidence.sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    state: Dict[str,str] = {}
    phase: Dict[str,str] = {}
    owner_track: Dict[str,Optional[int]] = {}
    prev_local: Dict[str,str] = {}
    prev_global: Dict[str,str] = {}
    slot_hist: Dict[str,deque] = defaultdict(deque)
    track_hist: Dict[Tuple[str,int],deque] = defaultdict(deque)

    slot_meta = evidence.groupby('local_id', sort=False).first()
    for lid, r in slot_meta.iterrows():
        init = str(r.get('initial_state','UNKNOWN')).upper()
        st = 'OCCUPIED' if init == 'OCCUPIED' else 'EMPTY'
        state[str(lid)] = st
        phase[str(lid)] = st
        owner_track[str(lid)] = None
        prev_local[str(lid)] = st

    local_rows = []
    global_rows = []
    transitions = []

    for t, group in evidence.groupby('time_sec', sort=True):
        t = float(t)
        # First append the current observations, then derive all features from the
        # same trailing window. This is causal and therefore deployable in real time.
        for _, r in group.iterrows():
            lid = str(r['local_id'])
            rec = {
                't': t,
                'detected': int(r.get('detected',0)),
                'track_id': int(r.get('track_id',-1)),
                'det_conf': float(r.get('det_conf',0.0)),
                'match_score': float(r.get('match_score',0.0)),
                'speed': float(r.get('track_speed_px_s',0.0)),
                'motion_span': float(r.get('track_motion_span_px',0.0)),
                'visual_diff': float(r.get('visual_diff_initial',0.0)),
                'cctv': str(r.get('cctv','')),
            }
            slot_hist[lid].append(rec)
            while slot_hist[lid] and t - float(slot_hist[lid][0]['t']) > window_sec + 1e-6:
                slot_hist[lid].popleft()
            if rec['detected'] and rec['track_id'] >= 0:
                key = (rec['cctv'], rec['track_id'])
                track_hist[key].append({
                    't': t, 'lid': lid, 'speed': rec['speed'], 'motion_span': rec['motion_span']
                })
                while track_hist[key] and t - float(track_hist[key][0]['t']) > window_sec + 1e-6:
                    track_hist[key].popleft()

        # Drop stale track histories.
        for key in list(track_hist.keys()):
            if not track_hist[key] or t - float(track_hist[key][-1]['t']) > window_sec + 1e-6:
                del track_hist[key]

        now = []
        # Track ownership lookup prevents one tracked vehicle from becoming two
        # simultaneously occupied local slots while it maneuvers across a boundary.
        owned_by_track: Dict[Tuple[str,int],str] = {}
        for lid, tid in owner_track.items():
            if tid is None or state.get(lid) != 'OCCUPIED':
                continue
            try:
                cctv = str(slot_meta.loc[lid].get('cctv',''))
            except Exception:
                cctv = ''
            owned_by_track[(cctv, int(tid))] = lid

        for _, r in group.iterrows():
            lid = str(r['local_id'])
            cctv = str(r['cctv'])
            hist = list(slot_hist[lid])
            hits = [x for x in hist if x['detected']]
            recent = [x for x in hist if t - float(x['t']) <= recent_sec + 1e-6]
            recent_hits = [x for x in recent if x['detected']]
            hit_ratio = len(hits) / max(1, len(hist))
            recent_hit_ratio = len(recent_hits) / max(1, len(recent))
            mean_conf = float(np.mean([x['det_conf'] for x in hits])) if hits else 0.0

            track_counts = Counter(int(x['track_id']) for x in hits if int(x['track_id']) >= 0)
            dom_track = track_counts.most_common(1)[0][0] if track_counts else -1
            dom_track_ratio_in_slot = (track_counts.get(dom_track,0) / max(1,len(hits))) if dom_track >= 0 else 0.0
            track_slot_ratio = 0.0
            track_switches = 0
            track_speed = 0.0
            track_motion = 0.0
            if dom_track >= 0:
                th = list(track_hist.get((cctv, dom_track), []))
                if th:
                    track_slot_ratio = sum(1 for x in th if x['lid'] == lid) / max(1, len(th))
                    lids_seq = [x['lid'] for x in th]
                    track_switches = sum(1 for a,b in zip(lids_seq, lids_seq[1:]) if a != b)
                    tr_recent = [x for x in th if t - float(x['t']) <= recent_sec + 1e-6]
                    src = tr_recent if tr_recent else th
                    track_speed = float(np.median([x['speed'] for x in src])) if src else 0.0
                    track_motion = float(max([x['motion_span'] for x in src], default=0.0))

            initial = str(r.get('initial_state','UNKNOWN')).upper()
            vis_diff = float(r.get('visual_diff_initial',0.0))
            appearance_occ = _appearance_occupied(initial, vis_diff, visual_thr)
            stable_track = (
                dom_track >= 0
                and track_slot_ratio >= stable_track_ratio_req
                and dom_track_ratio_in_slot >= 0.50
                and track_switches <= max_switches
                and track_speed <= stationary_speed
                and track_motion <= stationary_span
                and recent_hit_ratio >= 0.34
            )
            maneuvering = (
                dom_track >= 0
                and (track_switches > max_switches or track_speed > stationary_speed or track_motion > stationary_span)
            )

            if state[lid] == 'EMPTY':
                strong_stationary = (
                    hit_ratio >= max(entry_ratio, 0.65)
                    and recent_hit_ratio >= 0.66
                    and track_speed <= stationary_speed
                    and track_motion <= stationary_span
                )
                entry_ready = (stable_track and hit_ratio >= entry_ratio) or strong_stationary
                conflict_lid = owned_by_track.get((cctv, dom_track)) if dom_track >= 0 else None
                if entry_ready and (not conflict_lid or conflict_lid == lid):
                    state[lid] = 'OCCUPIED'
                    phase[lid] = 'OCCUPIED'
                    owner_track[lid] = dom_track if dom_track >= 0 else None
                    if dom_track >= 0:
                        owned_by_track[(cctv, dom_track)] = lid
                elif maneuvering or (dom_track >= 0 and hit_ratio > 0):
                    phase[lid] = 'MANEUVERING'
                else:
                    phase[lid] = 'EMPTY'
            else:  # OCCUPIED
                if dom_track >= 0 and recent_hit_ratio > 0:
                    if owner_track[lid] is None or stable_track:
                        owner_track[lid] = dom_track
                        owned_by_track[(cctv, dom_track)] = lid
                clear_empty = (
                    hit_ratio <= exit_ratio
                    and recent_hit_ratio <= exit_ratio
                    and not appearance_occ
                )
                if clear_empty:
                    old_owner = owner_track[lid]
                    state[lid] = 'EMPTY'
                    phase[lid] = 'EMPTY'
                    owner_track[lid] = None
                    if old_owner is not None and owned_by_track.get((cctv,int(old_owner))) == lid:
                        owned_by_track.pop((cctv,int(old_owner)), None)
                elif maneuvering and recent_hit_ratio > 0:
                    phase[lid] = 'LEAVING' if owner_track[lid] == dom_track else 'OCCUPIED'
                else:
                    phase[lid] = 'OCCUPIED'

            temporal_score = (
                0.45*hit_ratio + 0.20*recent_hit_ratio + 0.20*track_slot_ratio + 0.15*(1.0 if appearance_occ else 0.0)
            )
            if state[lid] == 'OCCUPIED':
                occ_score = max(0.51, min(0.99, temporal_score))
            else:
                occ_score = min(0.49, max(0.01, temporal_score))

            row = {
                'time_sec': t, 'timestamp': r['timestamp'], 'cctv': cctv,
                'local_id': lid, 'global_id': r['global_id'],
                'detected': int(r.get('detected',0)), 'det_conf': float(r.get('det_conf',0.0)),
                'track_id': int(r.get('track_id',-1)), 'dominant_track_id': int(dom_track),
                'state': state[lid], 'phase': phase[lid], 'occupied_score': float(occ_score),
                'window_hit_ratio': float(hit_ratio), 'recent_hit_ratio': float(recent_hit_ratio),
                'dominant_track_ratio_in_slot': float(dom_track_ratio_in_slot),
                'track_slot_ratio': float(track_slot_ratio), 'track_switches_window': int(track_switches),
                'track_speed_px_s': float(track_speed), 'track_motion_span_px': float(track_motion),
                'appearance_occupied': int(bool(appearance_occ)), 'visual_diff_initial': vis_diff,
                'owner_track_id': int(owner_track[lid]) if owner_track[lid] is not None else -1,
                'warmup': int(t < window_sec - 1e-6),
            }
            local_rows.append(row)
            now.append(row)
            if prev_local.get(lid) != state[lid]:
                transitions.append({
                    'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,
                    'global_id':r['global_id'],'from_state':prev_local.get(lid,''),
                    'to_state':state[lid],'cctv':cctv,'phase':phase[lid],
                    'dominant_track_id':int(dom_track),
                })
                prev_local[lid] = state[lid]

        by_global = defaultdict(list)
        for x in now:
            by_global[x['global_id']].append(x)
        for gid, items in by_global.items():
            occ = sum(1 for x in items if x['state'] == 'OCCUPIED')
            scores = [float(x['occupied_score']) for x in items]
            if fusion == 'CONF_MAX':
                gscore = max(scores) if scores else 0.0
                global_occ = gscore >= global_thr
            elif fusion == 'CONF_MEAN':
                gscore = float(np.mean(scores)) if scores else 0.0
                global_occ = gscore >= global_thr
            elif fusion == 'MAJORITY':
                gscore = float(np.mean(scores)) if scores else 0.0
                global_occ = occ >= math.ceil(len(items)/2)
            else:  # ANY: conservative against false empty
                gscore = max(scores) if scores else 0.0
                global_occ = occ > 0
            gstate = 'OCCUPIED' if global_occ else 'EMPTY'
            phases = [str(x.get('phase','')) for x in items]
            if 'MANEUVERING' in phases:
                gphase = 'MANEUVERING'
            elif 'LEAVING' in phases:
                gphase = 'LEAVING'
            else:
                gphase = gstate
            grow = {
                'time_sec': t, 'timestamp': items[0]['timestamp'], 'global_id': gid,
                'state': gstate, 'phase': gphase, 'global_score': gscore,
                'source_local_slots': ';'.join(x['local_id'] for x in items),
                'occupied_votes': occ, 'total_votes': len(items),
                'warmup': int(t < window_sec - 1e-6),
            }
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid] != gstate:
                transitions.append({
                    'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,
                    'global_id':gid,'from_state':prev_global[gid],'to_state':gstate,
                    'cctv':'MULTI','phase':gphase,'dominant_track_id':-1,
                })
            prev_global[gid] = gstate

    return pd.DataFrame(local_rows), pd.DataFrame(global_rows), pd.DataFrame(transitions)


def evaluate_state_output(global_df: pd.DataFrame, gt: pd.DataFrame, dev_end_sec: float, warmup_sec: float = 10.0) -> Dict:
    pred = global_df.copy()
    pred['occupied_pred'] = (pred['state'] == 'OCCUPIED').astype(int)
    counts = pred.groupby('time_sec', as_index=False)['occupied_pred'].sum()
    gt2 = gt[['time_sec','timestamp','ground_truth_occupied_space_count']].copy()
    gt2 = gt2[gt2['time_sec'] >= float(warmup_sec) - 1e-6].copy()
    merged = gt2.merge(counts, on='time_sec', how='left')
    merged['occupied_pred'] = merged['occupied_pred'].fillna(0).astype(int)
    merged['error'] = merged['occupied_pred'] - merged['ground_truth_occupied_space_count'].astype(int)
    merged['abs_error'] = merged['error'].abs()
    from split_protocol import labels
    merged['split'] = labels(merged['time_sec'],dev_end_sec)
    def metrics(part):
        if len(part) == 0:
            return {'N':0,'exact_rate':0.0,'mae':0.0,'max_abs_error':0,'false_empty_bias_rate':0.0,'over_rate':0.0}
        return {
            'N': int(len(part)), 'exact_rate': float((part['error']==0).mean()),
            'mae': float(part['abs_error'].mean()), 'max_abs_error': int(part['abs_error'].max()),
            'false_empty_bias_rate': float((part['error']<0).mean()),
            'over_rate': float((part['error']>0).mean()),
        }
    return {'timeseries': merged, 'DEV': metrics(merged[merged['split']=='DEV']), 'TEST': metrics(merged[merged['split']=='TEST'])}


def evaluate_slot_level(global_df: pd.DataFrame, slot_gt_events: pd.DataFrame, gt_times: Sequence[float], dev_end_sec: float, warmup_sec: float = 10.0) -> Dict:
    gids = sorted(set(slot_gt_events['global_id'].astype(str)))
    events = {g: slot_gt_events[slot_gt_events['global_id'].astype(str)==g].sort_values('time_sec') for g in gids}
    pred_lookup = {(float(r.time_sec),str(r.global_id)):str(r.state) for r in global_df.itertuples()}
    rows = []
    for t in sorted(float(x) for x in gt_times if float(x) >= float(warmup_sec)-1e-6):
        for gid in gids:
            ev = events[gid]
            past = ev[ev['time_sec'] <= t]
            if past.empty:
                continue
            gt_state = str(past.iloc[-1]['state'])
            pred_state = pred_lookup.get((t,gid))
            if pred_state is None:
                sub = global_df[(global_df['global_id'].astype(str)==gid)&(global_df['time_sec']<=t)]
                pred_state = str(sub.iloc[-1]['state']) if not sub.empty else 'EMPTY'
            from split_protocol import labels
            split = str(labels([t],dev_end_sec)[0])
            if split == 'IGNORED': continue
            rows.append({
                'time_sec':t,'timestamp':format_timestamp(t),'global_id':gid,
                'gt_state':gt_state,'pred_state':pred_state,'correct':int(gt_state==pred_state),
                'false_empty':int(gt_state=='OCCUPIED' and pred_state=='EMPTY'),
                'false_occupied':int(gt_state=='EMPTY' and pred_state=='OCCUPIED'),'split':split,
            })
    df = pd.DataFrame(rows)
    def metrics(part):
        if part.empty:
            return {'N':0,'slot_accuracy':0.0,'false_empty_rate':0.0,'false_occupied_rate':0.0,'all_slots_exact_time_rate':0.0}
        pertime = part.groupby('time_sec')['correct'].min()
        return {
            'N':int(len(part)), 'slot_accuracy':float(part['correct'].mean()),
            'false_empty_rate':float(part['false_empty'].mean()),
            'false_occupied_rate':float(part['false_occupied'].mean()),
            'all_slots_exact_time_rate':float(pertime.mean()),
        }
    if df.empty:
        by_slot_df = pd.DataFrame()
    else:
        by_slot_df = df.groupby(['split','global_id']).agg(
            N=('correct','size'),accuracy=('correct','mean'),false_empty=('false_empty','mean'),
            false_occupied=('false_occupied','mean')).reset_index()
    return {'rows':df,'DEV':metrics(df[df['split']=='DEV']) if not df.empty else metrics(df),
            'TEST':metrics(df[df['split']=='TEST']) if not df.empty else metrics(df),'by_slot':by_slot_df}


def _state_param_combos(search: Dict) -> List[Dict]:
    combos = []
    seen = set()
    for entry_ratio, exit_ratio, stable_ratio, vthr, fusion, gthr in itertools.product(
        search.get('entry_hit_ratio',[0.35,0.50]),
        search.get('exit_hit_ratio',[0.10,0.20]),
        search.get('stable_track_ratio',[0.55,0.70]),
        search.get('occupied_visual_diff_threshold',[0.08,0.12,0.16]),
        search.get('global_merge',['ANY']),
        search.get('global_confidence_threshold',[0.60]),
    ):
        effective_thr = float(gthr) if str(fusion).upper().startswith('CONF_') else 0.0
        key = (float(entry_ratio),float(exit_ratio),float(stable_ratio),float(vthr),str(fusion).upper(),effective_thr)
        if key in seen:
            continue
        seen.add(key)
        combos.append({
            'entry_hit_ratio':float(entry_ratio), 'exit_hit_ratio':float(exit_ratio),
            'stable_track_ratio':float(stable_ratio), 'occupied_visual_diff_threshold':float(vthr),
            'global_merge':str(fusion).upper(), 'global_confidence_threshold':effective_thr,
        })
    return combos


def search_state_parameters(evidence: pd.DataFrame, gt: pd.DataFrame, settings: Dict, output_dir: str,
                            progress=None, slot_gt_events_path: Optional[str]=None) -> Dict:
    out_dir = ensure_dir(output_dir)
    search = settings['state_search']
    combos = _state_param_combos(search)
    temporal = settings.get('temporal', {})
    fixed = {
        'temporal_window_sec': float(temporal.get('window_sec',10.0)),
        'recent_window_sec': float(temporal.get('recent_window_sec',3.0)),
        'stationary_speed_px_s': float(temporal.get('stationary_speed_px_s',14.0)),
        'stationary_motion_span_px': float(temporal.get('stationary_motion_span_px',32.0)),
        'max_track_switches_for_entry': int(temporal.get('max_track_switches_for_entry',1)),
    }
    dev_end = float(settings.get('dev_end_sec',900))
    warmup = float(settings.get('evaluation_warmup_sec', temporal.get('warmup_sec',10.0)))
    leaderboard = []
    best = None
    best_key = None
    best_outputs = None
    slot_events = None
    if slot_gt_events_path and Path(slot_gt_events_path).exists():
        slot_events = load_slot_gt_events(slot_gt_events_path)
    gt_times = gt['time_sec'].tolist()

    for i, base in enumerate(combos):
        params = {**base, **fixed}
        local_df, global_df, transitions = run_state_engine(evidence, params)
        ev = evaluate_state_output(global_df, gt, dev_end, warmup)
        d = ev['DEV']
        slot_eval = evaluate_slot_level(global_df, slot_events, gt_times, dev_end, warmup) if slot_events is not None else None
        if slot_eval is not None:
            sd = slot_eval['DEV']
            key = (-sd['all_slots_exact_time_rate'], sd['false_empty_rate'], -sd['slot_accuracy'], -d['exact_rate'], d['mae'], d['max_abs_error'])
        else:
            key = (-d['exact_rate'], d['mae'], d['false_empty_bias_rate'], d['max_abs_error'], d['over_rate'])
        row = {
            **params, 'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],
            'dev_under_rate':d['false_empty_bias_rate'],'dev_over_rate':d['over_rate'],
            'dev_max_abs_error':d['max_abs_error'],
        }
        if slot_eval is not None:
            row.update({
                'dev_slot_accuracy':slot_eval['DEV']['slot_accuracy'],
                'dev_slot_false_empty_rate':slot_eval['DEV']['false_empty_rate'],
                'dev_all_slots_exact_time_rate':slot_eval['DEV']['all_slots_exact_time_rate'],
            })
        leaderboard.append(row)
        if best is None or key < best_key:
            best, best_key = params, key
            best_outputs = (local_df,global_df,transitions,ev,slot_eval)
        if progress and (i % max(1,len(combos)//20) == 0 or i == len(combos)-1):
            progress((i+1)/len(combos), f'10s temporal state search {i+1}/{len(combos)}')

    sort_cols = ['dev_exact_rate','dev_MAE','dev_under_rate','dev_max_abs_error','dev_over_rate']
    asc = [False,True,True,True,True]
    if slot_events is not None:
        sort_cols = ['dev_all_slots_exact_time_rate','dev_slot_false_empty_rate','dev_slot_accuracy'] + sort_cols
        asc = [False,True,False] + asc
    board = pd.DataFrame(leaderboard).sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    board.to_csv(out_dir/'state_search_leaderboard.csv', index=False, encoding='utf-8-sig')
    local_df,global_df,transitions,ev,slot_eval = best_outputs
    local_df.to_csv(out_dir/'selected_slot_timeseries.csv', index=False, encoding='utf-8-sig')
    global_df.to_csv(out_dir/'selected_global_slot_timeseries.csv', index=False, encoding='utf-8-sig')
    transitions.to_csv(out_dir/'selected_state_transitions.csv', index=False, encoding='utf-8-sig')
    ev['timeseries'].to_csv(out_dir/'selected_count_timeseries.csv', index=False, encoding='utf-8-sig')
    save_json(out_dir/'selected_state_params.json', best)
    pd.DataFrame([{'split':sp,**ev[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'metrics_dev_test.csv', index=False, encoding='utf-8-sig')
    errors = ev['timeseries'][(ev['timeseries']['split']=='TEST') & (ev['timeseries']['error']!=0)].copy()
    errors.to_csv(out_dir/'test_error_cases.csv', index=False, encoding='utf-8-sig')

    if slot_eval is not None:
        slot_eval['rows'].to_csv(out_dir/'slot_level_comparison.csv', index=False, encoding='utf-8-sig')
        slot_eval['by_slot'].to_csv(out_dir/'slot_level_by_slot.csv', index=False, encoding='utf-8-sig')
        slot_eval['rows'][(slot_eval['rows']['split']=='TEST')&(slot_eval['rows']['correct']==0)].to_csv(out_dir/'slot_level_test_errors.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame([{'split':sp,**slot_eval[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'slot_level_metrics.csv', index=False, encoding='utf-8-sig')

    with (out_dir/'REPORT.txt').open('w',encoding='utf-8') as f:
        f.write('Parking Slot State Engine v10 - 10s rolling temporal tracking + maneuver-aware slot state\n\n')
        f.write(f'Warm-up excluded from metrics: 0 <= t < {warmup:.1f}s\n')
        f.write('Every decision uses only current/past frames; no future frames are used.\n\n')
        f.write('SELECTED PARAMETERS\n')
        for k,v in best.items():
            f.write(f'{k}={v}\n')
        for sp in ['DEV','TEST']:
            m = ev[sp]
            f.write(f'\n[{sp} COUNT]\nN={m["N"]}\nExact match rate={m["exact_rate"]*100:.2f}%\nMAE={m["mae"]:.4f}\n')
            f.write(f'Under-count rate={m["false_empty_bias_rate"]*100:.2f}%\nOver-count rate={m["over_rate"]*100:.2f}%\nMax absolute error={m["max_abs_error"]}\n')
            if slot_eval is not None:
                sm = slot_eval[sp]
                f.write(f'[{sp} SLOT]\nSlot accuracy={sm["slot_accuracy"]*100:.2f}%\nFalse-empty rate={sm["false_empty_rate"]*100:.2f}%\n')
                f.write(f'False-occupied rate={sm["false_occupied_rate"]*100:.2f}%\nAll-slots-exact time rate={sm["all_slots_exact_time_rate"]*100:.2f}%\n')

    files = [
        'REPORT.txt','metrics_dev_test.csv','state_search_leaderboard.csv','selected_state_params.json',
        'selected_count_timeseries.csv','selected_slot_timeseries.csv','selected_global_slot_timeseries.csv',
        'selected_state_transitions.csv','test_error_cases.csv'
    ]
    if slot_eval is not None:
        files += ['slot_level_metrics.csv','slot_level_by_slot.csv','slot_level_comparison.csv','slot_level_test_errors.csv']
    zip_paths(out_dir/'UPLOAD_TO_CHATGPT.zip', [(out_dir/x,x) for x in files])
    return {'best_params':best,'evaluation':ev,'slot_evaluation':slot_eval,'leaderboard':board,'output_dir':str(out_dir)}

# =============================================================================
# v11 detector-mode experiment override
# FULL CCTV YOLO vs per-slot crop YOLO vs HYBRID, while preserving the causal
# rolling 10-second temporal engine. Definitions below intentionally override
# v10 public functions.
# =============================================================================

def _slot_crop_geometry(slot: Dict, slots_same_cam: Sequence[Dict], img_shape, crop_cfg: Dict) -> Dict:
    """Return a perspective-warped crop centered around the immutable manual slot point."""
    h, w = int(img_shape[0]), int(img_shape[1])
    px, py = _manual_point(slot)
    neigh = []
    for other in slots_same_cam:
        if str(other.get('local_id')) == str(slot.get('local_id')):
            continue
        ox, oy = _manual_point(other)
        neigh.append(math.hypot(px-ox, py-oy))
    nearest = min(neigh) if neigh else float(crop_cfg.get('fallback_neighbor_px', 90.0))
    nearest = _clamp(nearest, float(crop_cfg.get('neighbor_min_px', 36.0)), float(crop_cfg.get('neighbor_max_px', 180.0)))
    cw = _clamp(nearest * float(crop_cfg.get('width_neighbor_factor', 2.20)),
                float(crop_cfg.get('min_width_px', 96.0)), float(crop_cfg.get('max_width_px', 360.0)))
    ch = _clamp(nearest * float(crop_cfg.get('height_neighbor_factor', 2.60)),
                float(crop_cfg.get('min_height_px', 112.0)), float(crop_cfg.get('max_height_px', 420.0)))
    # Parking-space points are often near the ground footprint. Shift the visual
    # context slightly upward so the crop contains the vehicle body as well.
    cy = py - nearest * float(crop_cfg.get('vertical_center_offset_neighbor_factor', 0.30))
    x1, y1, x2, y2 = clip_box((px-cw/2, cy-ch/2, px+cw/2, cy+ch/2), w, h)
    core_r = _clamp(nearest * float(crop_cfg.get('core_radius_neighbor_factor', 0.62)),
                    float(crop_cfg.get('core_radius_min_px', 22.0)), float(crop_cfg.get('core_radius_max_px', 95.0)))
    return {
        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
        'manual_x': float(px), 'manual_y': float(py), 'nearest_slot_px': float(nearest),
        'core_radius_px': float(core_r),
    }


def _crop_detector_config(full_cfg: Dict, crop_cfg: Dict) -> Dict:
    cfg = dict(full_cfg)
    cfg['imgsz'] = int(crop_cfg.get('imgsz', 640))
    cfg['conf'] = float(crop_cfg.get('inference_conf_floor', 0.03))
    cfg['tiling'] = False
    return cfg


def _best_slot_crop_detection(dets: Sequence[Detection], geom: Dict, slot: Dict, crop_cfg: Dict) -> Tuple[Optional[Detection], Dict]:
    """Choose the crop detection that actually belongs to this slot's central core."""
    px, py = _manual_point(slot)
    x1, y1 = float(geom['x1']), float(geom['y1'])
    core = max(1.0, float(geom['core_radius_px']))
    best = None
    best_meta = {'crop_score': 0.0, 'crop_center_distance_px': np.nan}
    for d in dets:
        gx1, gy1, gx2, gy2 = d.x1+x1, d.y1+y1, d.x2+x1, d.y2+y1
        gcx, gcy = (gx1+gx2)/2.0, (gy1+gy2)/2.0
        dist = math.hypot(gcx-px, gcy-py)
        global_box = [gx1, gy1, gx2, gy2]
        central = dist <= core or point_in_box((px,py), global_box, float(crop_cfg.get('point_containment_expand', 0.06)))
        if not central:
            continue
        distance_penalty = min(1.0, dist/core)
        score = float(d.conf) * (1.0 - float(crop_cfg.get('distance_penalty_weight', 0.30))*distance_penalty)
        if best is None or score > best_meta['crop_score']:
            best = Detection(gx1, gy1, gx2, gy2, float(d.conf), int(d.cls))
            best_meta = {'crop_score': float(score), 'crop_center_distance_px': float(dist)}
    return best, best_meta


def _batched_crop_detect(detector: VehicleDetector, images: Sequence[np.ndarray], batch_size: int) -> List[List[Detection]]:
    out: List[List[Detection]] = []
    bs = max(1, int(batch_size))
    for i in range(0, len(images), bs):
        out.extend(detector.detect_batch(images[i:i+bs]))
    return out


def extract_evidence(video_path: str, rois: Dict[str,Sequence[int]], slots_path: str, settings: Dict,
                     learn_dir: str, output_dir: str, progress=None) -> pd.DataFrame:
    """v11 evidence extraction.

    One pass records both signals needed for the controlled experiment:
      FULL   = YOLO on the complete warped CCTV image + one-to-one slot assignment.
      CROP   = YOLO on each slot-centered crop, batched per CCTV.
      HYBRID = decided later from both signals without re-running YOLO.
    """
    out_dir = ensure_dir(output_dir)
    slots = load_slots(slots_path)
    detector_cfgs = settings.get('detector_by_cctv', {})
    full_detectors = {c: VehicleDetector(detector_cfgs.get(c, settings['detector'])) for c in rois}
    crop_cfg = settings.get('slot_crop_detector', {})
    crop_detectors = {
        c: VehicleDetector(_crop_detector_config(detector_cfgs.get(c, settings['detector']), crop_cfg))
        for c in rois
    }
    matching_cfg = settings.get('matching', {})
    temporal_cfg = settings.get('temporal', {})
    trackers = {c: _OnlineCentroidTracker(temporal_cfg) for c in rois}
    end_sec = float(settings.get('eval_end_sec', 1230))
    sample_sec = float(settings.get('evidence_sample_sec', 1.0))
    times = np.arange(0.0, end_sec + 1e-6, sample_sec).tolist()
    by_cctv = defaultdict(list)
    for s in slots:
        by_cctv[s['cctv']].append(s)

    templates = {}
    learn_path = Path(learn_dir)
    for s in slots:
        p = _resolve_template_path(s, learn_path)
        templates[s['local_id']] = cv2.imread(str(p)) if p else None

    cap = open_video(video_path)
    rows = []
    summary = []
    track_rows = []
    crop_geometry_rows = []
    crop_geometry_by_cctv: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    geometry_initialized = False
    batch_size = int(crop_cfg.get('batch_size', 8))
    try:
        for ti, t in enumerate(times):
            frame = read_frame_at(cap, t)
            for cctv, roi in rois.items():
                img = crop_roi(frame, roi)
                cslots = by_cctv.get(cctv, [])
                if not geometry_initialized or not crop_geometry_by_cctv.get(cctv):
                    for s in cslots:
                        g = _slot_crop_geometry(s, cslots, img.shape, crop_cfg)
                        crop_geometry_by_cctv[cctv][str(s['local_id'])] = g
                        crop_geometry_rows.append({'cctv':cctv,'local_id':s['local_id'],'global_id':s.get('global_id',s['local_id']),**g})

                # FULL CCTV detector and tracker.
                full_dets = full_detectors[cctv].detect(img)
                track_meta = trackers[cctv].update(full_dets, float(t))
                assigned, unmatched, ameta = assign_detections_to_slots(full_dets, cslots, matching_cfg)
                assigned_by_index = {
                    int(meta.get('det_index')): lid
                    for lid, meta in ameta.items() if meta.get('det_index') is not None
                }

                # Per-slot crop detector. The visual context is wide, but a detection
                # is accepted only when it belongs to the slot's central core.
                crop_images = []
                crop_slot_order = []
                for s in cslots:
                    g = crop_geometry_by_cctv[cctv][str(s['local_id'])]
                    crop_images.append(img[g['y1']:g['y2'], g['x1']:g['x2']].copy())
                    crop_slot_order.append(s)
                crop_batches = _batched_crop_detect(crop_detectors[cctv], crop_images, batch_size) if crop_images else []
                crop_best: Dict[str, Optional[Detection]] = {}
                crop_meta: Dict[str, Dict] = {}
                for s, det_list in zip(crop_slot_order, crop_batches):
                    lid = str(s['local_id'])
                    d, meta = _best_slot_crop_detection(det_list, crop_geometry_by_cctv[cctv][lid], s, crop_cfg)
                    crop_best[lid] = d
                    crop_meta[lid] = meta

                global_by_local = {str(s['local_id']): str(s.get('global_id', s['local_id'])) for s in cslots}
                summary.append({
                    'time_sec': round(float(t),3), 'timestamp':format_timestamp(t), 'cctv':cctv,
                    'full_deduplicated_detections':len(full_dets), 'full_assigned_detections':len(assigned),
                    'full_unmatched_detections':len(unmatched),
                    'crop_positive_slots':sum(1 for d in crop_best.values() if d is not None),
                    'active_track_detections':len(track_meta),
                })
                for di, d in enumerate(full_dets):
                    tm = track_meta.get(di,{})
                    lid = assigned_by_index.get(di,'')
                    dc = box_center(d.box)
                    track_rows.append({
                        'time_sec':round(float(t),3),'timestamp':format_timestamp(t),'cctv':cctv,
                        'track_id':int(tm.get('track_id',-1)), 'track_age_sec':float(tm.get('track_age_sec',0.0)),
                        'track_speed_px_s':float(tm.get('track_speed_px_s',0.0)),
                        'track_motion_span_px':float(tm.get('track_motion_span_px',0.0)),
                        'det_center_x':float(dc[0]),'det_center_y':float(dc[1]),'det_conf':float(d.conf),
                        'assigned_local_id':lid,'assigned_global_id':global_by_local.get(lid,'') if lid else '',
                    })

                for s in cslots:
                    lid = str(s['local_id'])
                    fd = assigned.get(lid)
                    fmeta = ameta.get(lid,{})
                    cd = crop_best.get(lid)
                    cmeta = crop_meta.get(lid,{})
                    region = s.get('region') or robust_region([], s['point'], img.shape)
                    rx1,ry1,rx2,ry2 = clip_box(region, img.shape[1], img.shape[0])
                    current = img[ry1:ry2,rx1:rx2]
                    diff = normalized_visual_diff(current, templates.get(lid))

                    fdc = box_center(fd.box) if fd else (np.nan,np.nan)
                    fdet_index = int(fmeta.get('det_index',-1)) if fd else -1
                    tm = track_meta.get(fdet_index,{}) if fd else {}
                    cdc = box_center(cd.box) if cd else (np.nan,np.nan)
                    match_score = float(fmeta.get('match_score',0.0))
                    full_occ_score = (0.55 + 0.45*(0.60*float(fd.conf)+0.40*match_score)) if fd else 0.0
                    crop_occ_score = float(cd.conf) if cd else 0.0

                    rows.append({
                        'time_sec':round(float(t),3),'timestamp':format_timestamp(t),'cctv':cctv,
                        'local_id':lid,'global_id':s.get('global_id',lid),'initial_state':s.get('initial_state','UNKNOWN'),
                        # Legacy fields remain FULL for backwards-readable CSVs.
                        'detected':int(fd is not None),'det_conf':float(fd.conf) if fd else 0.0,
                        'det_center_x':float(fdc[0]),'det_center_y':float(fdc[1]),
                        'track_id':int(tm.get('track_id',-1)) if fd else -1,
                        'track_age_sec':float(tm.get('track_age_sec',0.0)) if fd else 0.0,
                        'track_speed_px_s':float(tm.get('track_speed_px_s',0.0)) if fd else 0.0,
                        'track_motion_span_px':float(tm.get('track_motion_span_px',0.0)) if fd else 0.0,
                        # Explicit FULL signal.
                        'full_detected':int(fd is not None),'full_det_conf':float(fd.conf) if fd else 0.0,
                        'full_center_x':float(fdc[0]),'full_center_y':float(fdc[1]),
                        'full_track_id':int(tm.get('track_id',-1)) if fd else -1,
                        'full_track_age_sec':float(tm.get('track_age_sec',0.0)) if fd else 0.0,
                        'full_track_speed_px_s':float(tm.get('track_speed_px_s',0.0)) if fd else 0.0,
                        'full_track_motion_span_px':float(tm.get('track_motion_span_px',0.0)) if fd else 0.0,
                        'full_occupancy_score':float(full_occ_score),
                        # Explicit per-slot CROP signal.
                        'crop_detected':int(cd is not None),'crop_det_conf':float(cd.conf) if cd else 0.0,
                        'crop_center_x':float(cdc[0]),'crop_center_y':float(cdc[1]),
                        'crop_score':float(cmeta.get('crop_score',0.0)),
                        'crop_center_distance_px':float(cmeta.get('crop_center_distance_px',np.nan)),
                        'crop_occupancy_score':float(crop_occ_score),
                        'crop_x1':int(crop_geometry_by_cctv[cctv][lid]['x1']),
                        'crop_y1':int(crop_geometry_by_cctv[cctv][lid]['y1']),
                        'crop_x2':int(crop_geometry_by_cctv[cctv][lid]['x2']),
                        'crop_y2':int(crop_geometry_by_cctv[cctv][lid]['y2']),
                        'crop_core_radius_px':float(crop_geometry_by_cctv[cctv][lid]['core_radius_px']),
                        # Spatial assignment/debug and appearance.
                        'assignment_cost':float(fmeta.get('assignment_cost',np.nan)),
                        'match_score':match_score,'match_distance_px':float(fmeta.get('distance_px',np.nan)),
                        'manual_distance_px':float(fmeta.get('manual_distance_px',np.nan)),
                        'anchor_distance_px':float(fmeta.get('anchor_distance_px',np.nan)),
                        'voronoi_best_distance_px':float(fmeta.get('voronoi_best_distance_px',np.nan)),
                        'voronoi_margin_px':float(fmeta.get('voronoi_margin_px',np.nan)),
                        'voronoi_allowed':int(fmeta.get('voronoi_allowed',0)) if fd else 0,
                        'anchor_trusted':int(bool(s.get('anchor_trusted',False))),
                        'anchor_shift_px':float(s.get('anchor_shift_px',0.0) or 0.0),
                        'region_iou':float(fmeta.get('region_iou',0.0)),
                        'det_occupancy_score':float(full_occ_score),
                        'visual_diff_initial':float(diff),
                    })
            geometry_initialized = True
            if progress and (ti % max(1,int(10/sample_sec)) == 0 or ti == len(times)-1):
                progress((ti+1)/len(times), f'FULL + SLOT-CROP evidence {format_timestamp(t)} / {format_timestamp(end_sec)}')
    finally:
        cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir/'slot_evidence.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(out_dir/'frame_detection_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(track_rows).to_csv(out_dir/'vehicle_tracks.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(crop_geometry_rows).drop_duplicates(['cctv','local_id']).to_csv(out_dir/'slot_crop_geometry.csv',index=False,encoding='utf-8-sig')
    return df


def _effective_observation(r: pd.Series, params: Dict) -> Dict:
    mode = str(params.get('evidence_mode','HYBRID')).upper()
    full_conf_min = float(params.get('full_min_conf',0.0))
    crop_conf_min = float(params.get('crop_min_conf',0.10))
    hybrid_single = float(params.get('hybrid_single_min_conf',0.18))
    hybrid_both = float(params.get('hybrid_both_min_conf',0.05))

    fd = int(r.get('full_detected',r.get('detected',0))) > 0
    fc = float(r.get('full_det_conf',r.get('det_conf',0.0)))
    cd = int(r.get('crop_detected',0)) > 0
    cc = float(r.get('crop_det_conf',0.0))
    full_ok = fd and fc >= full_conf_min
    crop_ok = cd and cc >= crop_conf_min

    detected = False
    source = 'NONE'
    if mode == 'FULL':
        detected = full_ok
        source = 'FULL' if detected else 'NONE'
    elif mode == 'CROP':
        detected = crop_ok
        source = 'CROP' if detected else 'NONE'
    else:
        both_support = fd and cd and fc >= hybrid_both and cc >= hybrid_both
        full_strong = fd and fc >= hybrid_single
        crop_strong = cd and cc >= hybrid_single
        detected = bool(both_support or full_strong or crop_strong)
        if detected:
            source = 'BOTH' if both_support else ('FULL' if full_strong else 'CROP')

    if source in ('FULL','BOTH'):
        track_id = int(r.get('full_track_id',r.get('track_id',-1)))
        speed = float(r.get('full_track_speed_px_s',r.get('track_speed_px_s',0.0)))
        motion = float(r.get('full_track_motion_span_px',r.get('track_motion_span_px',0.0)))
        cx = float(r.get('full_center_x',r.get('det_center_x',np.nan)))
        cy = float(r.get('full_center_y',r.get('det_center_y',np.nan)))
    else:
        track_id = -1
        speed = 0.0
        motion = 0.0
        cx = float(r.get('crop_center_x',np.nan))
        cy = float(r.get('crop_center_y',np.nan))
    conf = max(fc if fd else 0.0, cc if cd else 0.0) if mode == 'HYBRID' else (fc if mode=='FULL' else cc)
    return {
        'detected':int(detected),'det_conf':float(conf if detected else 0.0),'track_id':int(track_id if detected else -1),
        'speed':float(speed if detected else 0.0),'motion_span':float(motion if detected else 0.0),
        'center_x':float(cx) if detected else np.nan,'center_y':float(cy) if detected else np.nan,
        'source':source,
    }


def run_state_engine(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """v11 causal 10-second state engine with selectable FULL/CROP/HYBRID evidence."""
    window_sec = float(params.get('temporal_window_sec',10.0))
    recent_sec = float(params.get('recent_window_sec',3.0))
    entry_ratio = float(params.get('entry_hit_ratio',0.50))
    exit_ratio = float(params.get('exit_hit_ratio',0.10))
    stable_track_ratio_req = float(params.get('stable_track_ratio',0.65))
    visual_thr = float(params.get('occupied_visual_diff_threshold',0.12))
    stationary_speed = float(params.get('stationary_speed_px_s',14.0))
    stationary_span = float(params.get('stationary_motion_span_px',32.0))
    crop_stationary_span = float(params.get('crop_stationary_motion_span_px',28.0))
    max_switches = int(params.get('max_track_switches_for_entry',1))
    entry_mean_conf_min = float(params.get('entry_mean_conf_min',0.10))
    fusion = str(params.get('global_merge','ANY')).upper()
    global_thr = float(params.get('global_confidence_threshold',0.60))
    evidence_mode = str(params.get('evidence_mode','HYBRID')).upper()

    evidence = evidence.sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    state: Dict[str,str] = {}; phase: Dict[str,str] = {}; owner_track: Dict[str,Optional[int]] = {}
    prev_local: Dict[str,str] = {}; prev_global: Dict[str,str] = {}
    slot_hist: Dict[str,deque] = defaultdict(deque); track_hist: Dict[Tuple[str,int],deque] = defaultdict(deque)
    slot_meta = evidence.groupby('local_id',sort=False).first()
    for lid,r in slot_meta.iterrows():
        init=str(r.get('initial_state','UNKNOWN')).upper(); st='OCCUPIED' if init=='OCCUPIED' else 'EMPTY'
        lid=str(lid); state[lid]=st; phase[lid]=st; owner_track[lid]=None; prev_local[lid]=st

    local_rows=[];global_rows=[];transitions=[]
    for t,group in evidence.groupby('time_sec',sort=True):
        t=float(t)
        for _,r in group.iterrows():
            lid=str(r['local_id']); obs=_effective_observation(r,params)
            rec={'t':t,**obs,'visual_diff':float(r.get('visual_diff_initial',0.0)),'cctv':str(r.get('cctv',''))}
            slot_hist[lid].append(rec)
            while slot_hist[lid] and t-float(slot_hist[lid][0]['t'])>window_sec+1e-6: slot_hist[lid].popleft()
            if rec['detected'] and rec['track_id']>=0:
                key=(rec['cctv'],rec['track_id']);track_hist[key].append({'t':t,'lid':lid,'speed':rec['speed'],'motion_span':rec['motion_span']})
                while track_hist[key] and t-float(track_hist[key][0]['t'])>window_sec+1e-6: track_hist[key].popleft()
        for key in list(track_hist.keys()):
            if not track_hist[key] or t-float(track_hist[key][-1]['t'])>window_sec+1e-6: del track_hist[key]

        owned_by_track={}
        for lid,tid in owner_track.items():
            if tid is None or state.get(lid)!='OCCUPIED': continue
            try:cctv=str(slot_meta.loc[lid].get('cctv',''))
            except Exception:cctv=''
            owned_by_track[(cctv,int(tid))]=lid
        now=[]
        for _,r in group.iterrows():
            lid=str(r['local_id']);cctv=str(r['cctv']);hist=list(slot_hist[lid]);hits=[x for x in hist if x['detected']]
            recent=[x for x in hist if t-float(x['t'])<=recent_sec+1e-6];recent_hits=[x for x in recent if x['detected']]
            hit_ratio=len(hits)/max(1,len(hist));recent_hit_ratio=len(recent_hits)/max(1,len(recent))
            mean_conf=float(np.mean([x['det_conf'] for x in hits])) if hits else 0.0
            track_counts=Counter(int(x['track_id']) for x in hits if int(x['track_id'])>=0)
            dom_track=track_counts.most_common(1)[0][0] if track_counts else -1
            dom_ratio=(track_counts.get(dom_track,0)/max(1,len(hits))) if dom_track>=0 else 0.0
            track_slot_ratio=0.0;track_switches=0;track_speed=0.0;track_motion=0.0
            if dom_track>=0:
                th=list(track_hist.get((cctv,dom_track),[]))
                if th:
                    track_slot_ratio=sum(1 for x in th if x['lid']==lid)/max(1,len(th));seq=[x['lid'] for x in th]
                    track_switches=sum(1 for a,b in zip(seq,seq[1:]) if a!=b)
                    tr_recent=[x for x in th if t-float(x['t'])<=recent_sec+1e-6];src=tr_recent if tr_recent else th
                    track_speed=float(np.median([x['speed'] for x in src])) if src else 0.0
                    track_motion=float(max([x['motion_span'] for x in src],default=0.0))
            centers=[(float(x['center_x']),float(x['center_y'])) for x in hits if np.isfinite(x['center_x']) and np.isfinite(x['center_y'])]
            local_motion=0.0
            if len(centers)>=2:
                mx=float(np.median([x for x,_ in centers]));my=float(np.median([y for _,y in centers]))
                local_motion=max(math.hypot(x-mx,y-my) for x,y in centers)
            init=str(r.get('initial_state','UNKNOWN')).upper();vis=float(r.get('visual_diff_initial',0.0));appearance_occ=_appearance_occupied(init,vis,visual_thr)
            stable_track=(dom_track>=0 and track_slot_ratio>=stable_track_ratio_req and dom_ratio>=0.50 and track_switches<=max_switches and track_speed<=stationary_speed and track_motion<=stationary_span and recent_hit_ratio>=0.34)
            stable_slot=(hit_ratio>=max(entry_ratio,0.55) and recent_hit_ratio>=0.50 and local_motion<=crop_stationary_span)
            maneuvering=(dom_track>=0 and (track_switches>max_switches or track_speed>stationary_speed or track_motion>stationary_span)) or (local_motion>crop_stationary_span and hit_ratio>0)

            if state[lid]=='EMPTY':
                strong_stationary=(hit_ratio>=max(entry_ratio,0.65) and recent_hit_ratio>=0.66 and local_motion<=crop_stationary_span)
                # If a reliable FULL-CCTV Track exists, its maneuver history has priority.
                # Slot-only stability is the fallback for CROP evidence or broken tracks; it
                # must not override an actively maneuvering tracked vehicle.
                evidence_stable = stable_track if dom_track >= 0 else (stable_slot or strong_stationary)
                entry_ready = evidence_stable and hit_ratio>=entry_ratio and mean_conf>=entry_mean_conf_min
                conflict=owned_by_track.get((cctv,dom_track)) if dom_track>=0 else None
                if entry_ready and (not conflict or conflict==lid):
                    state[lid]='OCCUPIED';phase[lid]='OCCUPIED';owner_track[lid]=dom_track if dom_track>=0 else None
                    if dom_track>=0:owned_by_track[(cctv,dom_track)]=lid
                elif maneuvering or hit_ratio>0: phase[lid]='MANEUVERING'
                else: phase[lid]='EMPTY'
            else:
                if dom_track>=0 and recent_hit_ratio>0 and (owner_track[lid] is None or stable_track):
                    owner_track[lid]=dom_track;owned_by_track[(cctv,dom_track)]=lid
                clear_empty=(hit_ratio<=exit_ratio and recent_hit_ratio<=exit_ratio and not appearance_occ)
                if clear_empty:
                    old=owner_track[lid];state[lid]='EMPTY';phase[lid]='EMPTY';owner_track[lid]=None
                    if old is not None and owned_by_track.get((cctv,int(old)))==lid:owned_by_track.pop((cctv,int(old)),None)
                elif maneuvering and recent_hit_ratio>0: phase[lid]='LEAVING' if owner_track[lid]==dom_track else 'OCCUPIED'
                else: phase[lid]='OCCUPIED'

            temporal_score=0.40*hit_ratio+0.18*recent_hit_ratio+0.12*track_slot_ratio+0.15*(1.0 if appearance_occ else 0.0)+0.15*min(1.0,mean_conf)
            occ_score=max(0.51,min(0.99,temporal_score)) if state[lid]=='OCCUPIED' else min(0.49,max(0.01,temporal_score))
            obs_now=_effective_observation(r,params)
            row={'time_sec':t,'timestamp':r['timestamp'],'cctv':cctv,'local_id':lid,'global_id':r['global_id'],
                 'evidence_mode':evidence_mode,'evidence_source':obs_now['source'],'detected':obs_now['detected'],'det_conf':obs_now['det_conf'],
                 'track_id':obs_now['track_id'],'dominant_track_id':int(dom_track),'state':state[lid],'phase':phase[lid],
                 'occupied_score':float(occ_score),'window_hit_ratio':float(hit_ratio),'recent_hit_ratio':float(recent_hit_ratio),
                 'mean_detection_confidence':float(mean_conf),'dominant_track_ratio_in_slot':float(dom_ratio),'track_slot_ratio':float(track_slot_ratio),
                 'track_switches_window':int(track_switches),'track_speed_px_s':float(track_speed),'track_motion_span_px':float(track_motion),
                 'slot_detection_motion_span_px':float(local_motion),'appearance_occupied':int(bool(appearance_occ)),'visual_diff_initial':vis,
                 'owner_track_id':int(owner_track[lid]) if owner_track[lid] is not None else -1,'warmup':int(t<window_sec-1e-6)}
            local_rows.append(row);now.append(row)
            if prev_local.get(lid)!=state[lid]:
                transitions.append({'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,'global_id':r['global_id'],
                    'from_state':prev_local.get(lid,''),'to_state':state[lid],'cctv':cctv,'phase':phase[lid],
                    'dominant_track_id':int(dom_track),'evidence_mode':evidence_mode})
                prev_local[lid]=state[lid]

        by_global=defaultdict(list)
        for x in now:by_global[x['global_id']].append(x)
        for gid,items in by_global.items():
            occ=sum(1 for x in items if x['state']=='OCCUPIED');scores=[float(x['occupied_score']) for x in items]
            if fusion=='CONF_MAX':gscore=max(scores) if scores else 0.0;global_occ=gscore>=global_thr
            elif fusion=='CONF_MEAN':gscore=float(np.mean(scores)) if scores else 0.0;global_occ=gscore>=global_thr
            elif fusion=='MAJORITY':gscore=float(np.mean(scores)) if scores else 0.0;global_occ=occ>=math.ceil(len(items)/2)
            else:gscore=max(scores) if scores else 0.0;global_occ=occ>0
            gstate='OCCUPIED' if global_occ else 'EMPTY';phases=[str(x.get('phase','')) for x in items]
            gphase='MANEUVERING' if 'MANEUVERING' in phases else ('LEAVING' if 'LEAVING' in phases else gstate)
            grow={'time_sec':t,'timestamp':items[0]['timestamp'],'global_id':gid,'state':gstate,'phase':gphase,'global_score':gscore,
                  'evidence_mode':evidence_mode,'source_local_slots':';'.join(x['local_id'] for x in items),'occupied_votes':occ,'total_votes':len(items),
                  'warmup':int(t<window_sec-1e-6)}
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid]!=gstate:
                transitions.append({'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,'global_id':gid,
                    'from_state':prev_global[gid],'to_state':gstate,'cctv':'MULTI','phase':gphase,'dominant_track_id':-1,'evidence_mode':evidence_mode})
            prev_global[gid]=gstate
    return pd.DataFrame(local_rows),pd.DataFrame(global_rows),pd.DataFrame(transitions)


def _state_param_combos(search: Dict) -> List[Dict]:
    combos=[];seen=set()
    modes=[str(x).upper() for x in search.get('evidence_mode',['FULL','CROP','HYBRID'])]
    for mode,entry_ratio,exit_ratio,stable_ratio,vthr,fusion,gthr,entry_conf in itertools.product(
        modes,
        search.get('entry_hit_ratio',[0.35,0.50]),search.get('exit_hit_ratio',[0.10,0.20]),
        search.get('stable_track_ratio',[0.55,0.70]),search.get('occupied_visual_diff_threshold',[0.08,0.12,0.16]),
        search.get('global_merge',['ANY']),search.get('global_confidence_threshold',[0.60]),
        search.get('entry_mean_conf_min',[0.08,0.15]),
    ):
        effective_thr=float(gthr) if str(fusion).upper().startswith('CONF_') else 0.0
        key=(mode,float(entry_ratio),float(exit_ratio),float(stable_ratio),float(vthr),str(fusion).upper(),effective_thr,float(entry_conf))
        if key in seen:continue
        seen.add(key)
        combos.append({
            'evidence_mode':mode,'entry_hit_ratio':float(entry_ratio),'exit_hit_ratio':float(exit_ratio),
            'stable_track_ratio':float(stable_ratio),'occupied_visual_diff_threshold':float(vthr),
            'global_merge':str(fusion).upper(),'global_confidence_threshold':effective_thr,'entry_mean_conf_min':float(entry_conf),
            'full_min_conf':float(search.get('full_min_conf',[0.0])[0] if isinstance(search.get('full_min_conf',[0.0]),list) else search.get('full_min_conf',0.0)),
            'crop_min_conf':float(search.get('crop_min_conf',[0.10])[0] if isinstance(search.get('crop_min_conf',[0.10]),list) else search.get('crop_min_conf',0.10)),
            'hybrid_single_min_conf':float(search.get('hybrid_single_min_conf',[0.18])[0] if isinstance(search.get('hybrid_single_min_conf',[0.18]),list) else search.get('hybrid_single_min_conf',0.18)),
            'hybrid_both_min_conf':float(search.get('hybrid_both_min_conf',[0.05])[0] if isinstance(search.get('hybrid_both_min_conf',[0.05]),list) else search.get('hybrid_both_min_conf',0.05)),
        })
    return combos


def search_state_parameters(evidence: pd.DataFrame, gt: pd.DataFrame, settings: Dict, output_dir: str,
                            progress=None, slot_gt_events_path: Optional[str]=None) -> Dict:
    out_dir=ensure_dir(output_dir);search=settings['state_search'];combos=_state_param_combos(search);temporal=settings.get('temporal',{})
    fixed={'temporal_window_sec':float(temporal.get('window_sec',10.0)),'recent_window_sec':float(temporal.get('recent_window_sec',3.0)),
           'stationary_speed_px_s':float(temporal.get('stationary_speed_px_s',14.0)),'stationary_motion_span_px':float(temporal.get('stationary_motion_span_px',32.0)),
           'crop_stationary_motion_span_px':float(temporal.get('crop_stationary_motion_span_px',28.0)),
           'max_track_switches_for_entry':int(temporal.get('max_track_switches_for_entry',1)),
           'zone_recent_sec':float(temporal.get('zone_recent_sec',4.0)),
           'zone_motion_step_px':float(temporal.get('zone_motion_step_px',14.0)),
           'zone_recent_motion_max_px':float(temporal.get('zone_recent_motion_max_px',24.0)),
           'entry_zone_settle_sec':float(temporal.get('entry_zone_settle_sec',8.0)),
           'exit_no_full_sec':float(temporal.get('exit_no_full_sec',4.0)),
           'exit_force_no_full_sec':float(temporal.get('exit_force_no_full_sec',9.0)),
           'exit_aux_window_sec':float(temporal.get('exit_aux_window_sec',5.0)),
           'exit_aux_min_checks':int(temporal.get('exit_aux_min_checks',2)),
           'aux_recovery_conf':float(temporal.get('aux_recovery_conf',0.12)),
           'aux_recovery_min_scales':int(temporal.get('aux_recovery_min_scales',1))}
    dev_end=float(settings.get('dev_end_sec',900));warmup=float(settings.get('evaluation_warmup_sec',temporal.get('warmup_sec',10.0)))
    leaderboard=[];best=None;best_key=None;best_outputs=None;slot_events=None
    if slot_gt_events_path and Path(slot_gt_events_path).exists():slot_events=load_slot_gt_events(slot_gt_events_path)
    gt_times=gt['time_sec'].tolist()
    for i,base in enumerate(combos):
        params={**base,**fixed};local_df,global_df,transitions=run_state_engine(evidence,params);ev=evaluate_state_output(global_df,gt,dev_end,warmup);d=ev['DEV'];te=ev['TEST']
        slot_eval=evaluate_slot_level(global_df,slot_events,gt_times,dev_end,warmup) if slot_events is not None else None
        if slot_eval is not None:
            sd=slot_eval['DEV'];key=(-sd['all_slots_exact_time_rate'],sd['false_empty_rate'],-sd['slot_accuracy'],-d['exact_rate'],d['mae'],d['max_abs_error'])
        else:key=(-d['exact_rate'],d['mae'],d['false_empty_bias_rate'],d['max_abs_error'],d['over_rate'])
        row={**params,'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],'dev_under_rate':d['false_empty_bias_rate'],
             'dev_over_rate':d['over_rate'],'dev_max_abs_error':d['max_abs_error'],
             'test_N':te['N'],'test_exact_rate':te['exact_rate'],'test_MAE':te['mae'],'test_under_rate':te['false_empty_bias_rate'],
             'test_over_rate':te['over_rate'],'test_max_abs_error':te['max_abs_error']}
        if slot_eval is not None:row.update({'dev_slot_accuracy':slot_eval['DEV']['slot_accuracy'],'dev_slot_false_empty_rate':slot_eval['DEV']['false_empty_rate'],'dev_all_slots_exact_time_rate':slot_eval['DEV']['all_slots_exact_time_rate']})
        leaderboard.append(row)
        if best is None or key<best_key:best,best_key=params,key;best_outputs=(local_df,global_df,transitions,ev,slot_eval)
        if progress and (i%max(1,len(combos)//20)==0 or i==len(combos)-1):progress((i+1)/len(combos),f'FULL/CROP/HYBRID temporal search {i+1}/{len(combos)}')
    sort_cols=['dev_exact_rate','dev_MAE','dev_under_rate','dev_max_abs_error','dev_over_rate'];asc=[False,True,True,True,True]
    if slot_events is not None:sort_cols=['dev_all_slots_exact_time_rate','dev_slot_false_empty_rate','dev_slot_accuracy']+sort_cols;asc=[False,True,False]+asc
    board=pd.DataFrame(leaderboard).sort_values(sort_cols,ascending=asc).reset_index(drop=True);board.to_csv(out_dir/'state_search_leaderboard.csv',index=False,encoding='utf-8-sig')
    # Best DEV-selected configuration inside each detection mode. TEST columns are report-only.
    mode_rows=[]
    for mode in ['FULL','CROP','HYBRID']:
        sub=board[board['evidence_mode']==mode]
        if not sub.empty:mode_rows.append(sub.iloc[0].to_dict())
    pd.DataFrame(mode_rows).to_csv(out_dir/'detection_mode_comparison.csv',index=False,encoding='utf-8-sig')
    local_df,global_df,transitions,ev,slot_eval=best_outputs
    local_df.to_csv(out_dir/'selected_slot_timeseries.csv',index=False,encoding='utf-8-sig');global_df.to_csv(out_dir/'selected_global_slot_timeseries.csv',index=False,encoding='utf-8-sig')
    transitions.to_csv(out_dir/'selected_state_transitions.csv',index=False,encoding='utf-8-sig');ev['timeseries'].to_csv(out_dir/'selected_count_timeseries.csv',index=False,encoding='utf-8-sig')
    save_json(out_dir/'selected_state_params.json',best);pd.DataFrame([{'split':sp,**ev[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'metrics_dev_test.csv',index=False,encoding='utf-8-sig')
    errors=ev['timeseries'][(ev['timeseries']['split']=='TEST')&(ev['timeseries']['error']!=0)].copy();errors.to_csv(out_dir/'test_error_cases.csv',index=False,encoding='utf-8-sig')
    if slot_eval is not None:
        slot_eval['rows'].to_csv(out_dir/'slot_level_comparison.csv',index=False,encoding='utf-8-sig');slot_eval['by_slot'].to_csv(out_dir/'slot_level_by_slot.csv',index=False,encoding='utf-8-sig')
        slot_eval['rows'][(slot_eval['rows']['split']=='TEST')&(slot_eval['rows']['correct']==0)].to_csv(out_dir/'slot_level_test_errors.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{'split':sp,**slot_eval[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'slot_level_metrics.csv',index=False,encoding='utf-8-sig')
    with (out_dir/'REPORT.txt').open('w',encoding='utf-8') as f:
        f.write('Parking Slot State Engine v11 - FULL vs SLOT-CROP vs HYBRID + 10s causal temporal state\n\n')
        f.write(f'Warm-up excluded: 0 <= t < {warmup:.1f}s\nEvery decision uses only current/past frames.\nCandidate slots are advisory only and never auto-added.\n\nSELECTED PARAMETERS\n')
        for k,v in best.items():f.write(f'{k}={v}\n')
        f.write('\nDETECTION MODE COMPARISON (each mode selected on DEV; TEST shown only for audit)\n')
        for mr in mode_rows:
            f.write(f"{mr['evidence_mode']}: DEV exact={mr['dev_exact_rate']*100:.2f}% MAE={mr['dev_MAE']:.4f} | TEST exact={mr['test_exact_rate']*100:.2f}% MAE={mr['test_MAE']:.4f}\n")
        for sp in ['DEV','TEST']:
            m=ev[sp];f.write(f'\n[{sp} COUNT]\nN={m["N"]}\nExact match rate={m["exact_rate"]*100:.2f}%\nMAE={m["mae"]:.4f}\nUnder-count rate={m["false_empty_bias_rate"]*100:.2f}%\nOver-count rate={m["over_rate"]*100:.2f}%\nMax absolute error={m["max_abs_error"]}\n')
            if slot_eval is not None:
                sm=slot_eval[sp];f.write(f'[{sp} SLOT]\nSlot accuracy={sm["slot_accuracy"]*100:.2f}%\nFalse-empty rate={sm["false_empty_rate"]*100:.2f}%\nFalse-occupied rate={sm["false_occupied_rate"]*100:.2f}%\nAll-slots-exact time rate={sm["all_slots_exact_time_rate"]*100:.2f}%\n')
    return {'best_params':best,'evaluation':ev,'slot_evaluation':slot_eval,'leaderboard':board,'mode_comparison':pd.DataFrame(mode_rows),'output_dir':str(out_dir)}

# =============================================================================
# v12 additive multi-scale auxiliary crop override
# FULL-CCTV detection is the baseline and can NEVER be suppressed by crop YOLO.
# Crop YOLO is auxiliary-only: it may add missing evidence when sufficiently
# strong/consistent across wider multi-scale contexts.
# =============================================================================

def _slot_crop_geometry(slot: Dict, slots_same_cam: Sequence[Dict], img_shape, crop_cfg: Dict, scale: float = 1.0) -> Dict:
    h, w = int(img_shape[0]), int(img_shape[1])
    px, py = _manual_point(slot)
    neigh = []
    for other in slots_same_cam:
        if str(other.get('local_id')) == str(slot.get('local_id')):
            continue
        ox, oy = _manual_point(other)
        neigh.append(math.hypot(px-ox, py-oy))
    nearest = min(neigh) if neigh else float(crop_cfg.get('fallback_neighbor_px', 90.0))
    nearest = _clamp(nearest, float(crop_cfg.get('neighbor_min_px', 36.0)), float(crop_cfg.get('neighbor_max_px', 180.0)))

    # v12: the base crop is intentionally wider than v11 so the whole vehicle body
    # remains visible even when the manual point is near the ground footprint.
    base_w = _clamp(nearest * float(crop_cfg.get('width_neighbor_factor', 2.80)),
                    float(crop_cfg.get('min_width_px', 128.0)), float(crop_cfg.get('max_width_px', 520.0)))
    base_h = _clamp(nearest * float(crop_cfg.get('height_neighbor_factor', 3.35)),
                    float(crop_cfg.get('min_height_px', 144.0)), float(crop_cfg.get('max_height_px', 620.0)))
    sc = max(0.75, float(scale))
    cw = min(float(w), base_w * sc)
    ch = min(float(h), base_h * sc)
    cy = py - nearest * float(crop_cfg.get('vertical_center_offset_neighbor_factor', 0.38))
    x1, y1, x2, y2 = clip_box((px-cw/2, cy-ch/2, px+cw/2, cy+ch/2), w, h)
    core_r = _clamp(nearest * float(crop_cfg.get('core_radius_neighbor_factor', 0.66)),
                    float(crop_cfg.get('core_radius_min_px', 24.0)), float(crop_cfg.get('core_radius_max_px', 105.0)))
    return {
        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
        'manual_x': float(px), 'manual_y': float(py), 'nearest_slot_px': float(nearest),
        'core_radius_px': float(core_r), 'scale': float(sc),
        'crop_width_px': int(x2-x1), 'crop_height_px': int(y2-y1),
    }


def _v12_multiscale_crop_geometries(slot: Dict, slots_same_cam: Sequence[Dict], img_shape, crop_cfg: Dict) -> List[Dict]:
    scales = crop_cfg.get('scales', [1.0, 1.55])
    if not isinstance(scales, (list, tuple)) or not scales:
        scales = [1.0, 1.55]
    out = []
    seen = set()
    for raw in scales:
        try:
            sc = round(float(raw), 3)
        except Exception:
            continue
        if sc <= 0 or sc in seen:
            continue
        seen.add(sc)
        out.append(_slot_crop_geometry(slot, slots_same_cam, img_shape, crop_cfg, sc))
    return out or [_slot_crop_geometry(slot, slots_same_cam, img_shape, crop_cfg, 1.0)]


def extract_evidence(video_path: str, rois: Dict[str,Sequence[int]], slots_path: str, settings: Dict,
                     learn_dir: str, output_dir: str, progress=None) -> pd.DataFrame:
    """v12 evidence extraction: FULL baseline + wider multi-scale AUX crops.

    The FULL path is unchanged and remains the primary detector/tracker.
    AUX crop detections are recorded separately and are never allowed to erase a
    valid FULL detection.  Two crop scales are batched for each CCTV frame.
    """
    out_dir = ensure_dir(output_dir)
    slots = load_slots(slots_path)
    detector_cfgs = settings.get('detector_by_cctv', {})
    full_detectors = {c: VehicleDetector(detector_cfgs.get(c, settings['detector'])) for c in rois}
    crop_cfg = settings.get('slot_crop_detector', {})
    crop_detectors = {
        c: VehicleDetector(_crop_detector_config(detector_cfgs.get(c, settings['detector']), crop_cfg))
        for c in rois
    }
    matching_cfg = settings.get('matching', {})
    temporal_cfg = settings.get('temporal', {})
    trackers = {c: _OnlineCentroidTracker(temporal_cfg) for c in rois}
    end_sec = float(settings.get('eval_end_sec', 1230))
    sample_sec = float(settings.get('evidence_sample_sec', 1.0))
    times = np.arange(0.0, end_sec + 1e-6, sample_sec).tolist()
    by_cctv = defaultdict(list)
    for s in slots:
        by_cctv[s['cctv']].append(s)

    templates = {}
    learn_path = Path(learn_dir)
    for s in slots:
        p = _resolve_template_path(s, learn_path)
        templates[s['local_id']] = cv2.imread(str(p)) if p else None

    cap = open_video(video_path)
    rows, summary, track_rows, crop_geometry_rows = [], [], [], []
    crop_geoms: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    batch_size = int(crop_cfg.get('batch_size', 12))
    try:
        for ti, t in enumerate(times):
            frame = read_frame_at(cap, t)
            for cctv, roi in rois.items():
                img = crop_roi(frame, roi)
                cslots = by_cctv.get(cctv, [])
                if not crop_geoms.get(cctv):
                    for s in cslots:
                        lid = str(s['local_id'])
                        gs = _v12_multiscale_crop_geometries(s, cslots, img.shape, crop_cfg)
                        crop_geoms[cctv][lid] = gs
                        for gi, g in enumerate(gs):
                            crop_geometry_rows.append({
                                'cctv': cctv, 'local_id': lid, 'global_id': s.get('global_id', lid),
                                'scale_index': gi, **g,
                            })

                # Primary FULL detector and tracker.
                full_dets = full_detectors[cctv].detect(img)
                track_meta = trackers[cctv].update(full_dets, float(t))
                assigned, unmatched, ameta = assign_detections_to_slots(full_dets, cslots, matching_cfg)
                assigned_by_index = {
                    int(meta.get('det_index')): lid
                    for lid, meta in ameta.items() if meta.get('det_index') is not None
                }

                # Auxiliary multi-scale crops. Each crop is wide enough to contain
                # vehicle context, while acceptance is still tied to the manual slot core.
                crop_images, crop_keys = [], []
                for s in cslots:
                    lid = str(s['local_id'])
                    for gi, g in enumerate(crop_geoms[cctv][lid]):
                        crop_images.append(img[g['y1']:g['y2'], g['x1']:g['x2']].copy())
                        crop_keys.append((s, gi, g))
                crop_batches = _batched_crop_detect(crop_detectors[cctv], crop_images, batch_size) if crop_images else []

                per_slot_scale: Dict[str, List[Tuple[int, Dict, Optional[Detection], Dict]]] = defaultdict(list)
                for (s, gi, g), det_list in zip(crop_keys, crop_batches):
                    lid = str(s['local_id'])
                    d, meta = _best_slot_crop_detection(det_list, g, s, crop_cfg)
                    per_slot_scale[lid].append((gi, g, d, meta))

                crop_best: Dict[str, Optional[Detection]] = {}
                crop_meta: Dict[str, Dict] = {}
                for s in cslots:
                    lid = str(s['local_id'])
                    items = per_slot_scale.get(lid, [])
                    positives = [x for x in items if x[2] is not None]
                    support = len(positives)
                    if positives:
                        best_item = max(positives, key=lambda x: float(x[3].get('crop_score', 0.0)))
                        gi, g, d, meta = best_item
                        confs = [float(x[2].conf) for x in positives]
                        meta = dict(meta)
                        meta.update({
                            'aux_support_scales': support,
                            'aux_total_scales': len(items),
                            'aux_best_scale': float(g.get('scale', 1.0)),
                            'aux_best_scale_index': int(gi),
                            'aux_mean_conf_supported': float(np.mean(confs)) if confs else 0.0,
                            'aux_max_conf_supported': float(max(confs)) if confs else 0.0,
                            'best_geom': g,
                            'per_scale': items,
                        })
                        crop_best[lid] = d
                        crop_meta[lid] = meta
                    else:
                        fallback = items[0][1] if items else _slot_crop_geometry(s, cslots, img.shape, crop_cfg, 1.0)
                        crop_best[lid] = None
                        crop_meta[lid] = {
                            'crop_score': 0.0, 'crop_center_distance_px': np.nan,
                            'aux_support_scales': 0, 'aux_total_scales': len(items),
                            'aux_best_scale': float(fallback.get('scale', 1.0)),
                            'aux_best_scale_index': 0, 'aux_mean_conf_supported': 0.0,
                            'aux_max_conf_supported': 0.0, 'best_geom': fallback,
                            'per_scale': items,
                        }

                global_by_local = {str(s['local_id']): str(s.get('global_id', s['local_id'])) for s in cslots}
                summary.append({
                    'time_sec': round(float(t),3), 'timestamp':format_timestamp(t), 'cctv':cctv,
                    'full_deduplicated_detections':len(full_dets), 'full_assigned_detections':len(assigned),
                    'full_unmatched_detections':len(unmatched),
                    'aux_crop_positive_slots':sum(1 for d in crop_best.values() if d is not None),
                    'aux_crop_multi_scale_confirmed_slots':sum(1 for lid in crop_meta if int(crop_meta[lid].get('aux_support_scales',0)) >= 2),
                    'active_track_detections':len(track_meta),
                })
                for di, d in enumerate(full_dets):
                    tm = track_meta.get(di,{})
                    lid = assigned_by_index.get(di,'')
                    dc = box_center(d.box)
                    track_rows.append({
                        'time_sec':round(float(t),3),'timestamp':format_timestamp(t),'cctv':cctv,
                        'track_id':int(tm.get('track_id',-1)), 'track_age_sec':float(tm.get('track_age_sec',0.0)),
                        'track_speed_px_s':float(tm.get('track_speed_px_s',0.0)),
                        'track_motion_span_px':float(tm.get('track_motion_span_px',0.0)),
                        'det_center_x':float(dc[0]),'det_center_y':float(dc[1]),'det_conf':float(d.conf),
                        'assigned_local_id':lid,'assigned_global_id':global_by_local.get(lid,'') if lid else '',
                    })

                for s in cslots:
                    lid = str(s['local_id'])
                    fd = assigned.get(lid)
                    fmeta = ameta.get(lid,{})
                    cd = crop_best.get(lid)
                    cmeta = crop_meta.get(lid,{})
                    region = s.get('region') or robust_region([], s['point'], img.shape)
                    rx1,ry1,rx2,ry2 = clip_box(region, img.shape[1], img.shape[0])
                    current = img[ry1:ry2,rx1:rx2]
                    diff = normalized_visual_diff(current, templates.get(lid))

                    fdc = box_center(fd.box) if fd else (np.nan,np.nan)
                    fdet_index = int(fmeta.get('det_index',-1)) if fd else -1
                    tm = track_meta.get(fdet_index,{}) if fd else {}
                    cdc = box_center(cd.box) if cd else (np.nan,np.nan)
                    match_score = float(fmeta.get('match_score',0.0))
                    full_occ_score = (0.55 + 0.45*(0.60*float(fd.conf)+0.40*match_score)) if fd else 0.0
                    crop_occ_score = float(cd.conf) if cd else 0.0
                    bg = cmeta.get('best_geom', crop_geoms[cctv][lid][0])

                    row = {
                        'time_sec':round(float(t),3),'timestamp':format_timestamp(t),'cctv':cctv,
                        'local_id':lid,'global_id':s.get('global_id',lid),'initial_state':s.get('initial_state','UNKNOWN'),
                        # Legacy fields remain FULL.
                        'detected':int(fd is not None),'det_conf':float(fd.conf) if fd else 0.0,
                        'det_center_x':float(fdc[0]),'det_center_y':float(fdc[1]),
                        'track_id':int(tm.get('track_id',-1)) if fd else -1,
                        'track_age_sec':float(tm.get('track_age_sec',0.0)) if fd else 0.0,
                        'track_speed_px_s':float(tm.get('track_speed_px_s',0.0)) if fd else 0.0,
                        'track_motion_span_px':float(tm.get('track_motion_span_px',0.0)) if fd else 0.0,
                        # Primary FULL evidence.
                        'full_detected':int(fd is not None),'full_det_conf':float(fd.conf) if fd else 0.0,
                        'full_center_x':float(fdc[0]),'full_center_y':float(fdc[1]),
                        'full_track_id':int(tm.get('track_id',-1)) if fd else -1,
                        'full_track_age_sec':float(tm.get('track_age_sec',0.0)) if fd else 0.0,
                        'full_track_speed_px_s':float(tm.get('track_speed_px_s',0.0)) if fd else 0.0,
                        'full_track_motion_span_px':float(tm.get('track_motion_span_px',0.0)) if fd else 0.0,
                        'full_occupancy_score':float(full_occ_score),
                        # Auxiliary crop evidence. Legacy crop_* names are retained.
                        'crop_detected':int(cd is not None),'crop_det_conf':float(cd.conf) if cd else 0.0,
                        'crop_center_x':float(cdc[0]),'crop_center_y':float(cdc[1]),
                        'crop_score':float(cmeta.get('crop_score',0.0)),
                        'crop_center_distance_px':float(cmeta.get('crop_center_distance_px',np.nan)),
                        'crop_occupancy_score':float(crop_occ_score),
                        'aux_crop_detected':int(cd is not None),
                        'aux_crop_det_conf':float(cd.conf) if cd else 0.0,
                        'aux_support_scales':int(cmeta.get('aux_support_scales',0)),
                        'aux_total_scales':int(cmeta.get('aux_total_scales',0)),
                        'aux_best_scale':float(cmeta.get('aux_best_scale',1.0)),
                        'aux_mean_conf_supported':float(cmeta.get('aux_mean_conf_supported',0.0)),
                        'aux_max_conf_supported':float(cmeta.get('aux_max_conf_supported',0.0)),
                        'crop_x1':int(bg['x1']),'crop_y1':int(bg['y1']),'crop_x2':int(bg['x2']),'crop_y2':int(bg['y2']),
                        'crop_core_radius_px':float(bg['core_radius_px']),
                        # Spatial assignment/debug and appearance.
                        'assignment_cost':float(fmeta.get('assignment_cost',np.nan)),
                        'match_score':match_score,'match_distance_px':float(fmeta.get('distance_px',np.nan)),
                        'manual_distance_px':float(fmeta.get('manual_distance_px',np.nan)),
                        'anchor_distance_px':float(fmeta.get('anchor_distance_px',np.nan)),
                        'voronoi_best_distance_px':float(fmeta.get('voronoi_best_distance_px',np.nan)),
                        'voronoi_margin_px':float(fmeta.get('voronoi_margin_px',np.nan)),
                        'voronoi_allowed':int(fmeta.get('voronoi_allowed',0)) if fd else 0,
                        'anchor_trusted':int(bool(s.get('anchor_trusted',False))),
                        'anchor_shift_px':float(s.get('anchor_shift_px',0.0) or 0.0),
                        'region_iou':float(fmeta.get('region_iou',0.0)),
                        'det_occupancy_score':float(full_occ_score),'visual_diff_initial':float(diff),
                    }
                    # Per-scale audit columns for easy post-hoc analysis.
                    for gi, g, dscale, mscale in cmeta.get('per_scale', []):
                        row[f'aux_scale_{gi}_factor'] = float(g.get('scale',1.0))
                        row[f'aux_scale_{gi}_detected'] = int(dscale is not None)
                        row[f'aux_scale_{gi}_conf'] = float(dscale.conf) if dscale else 0.0
                    rows.append(row)
            if progress and (ti % max(1,int(10/sample_sec)) == 0 or ti == len(times)-1):
                progress((ti+1)/len(times), f'FULL + multi-scale AUX evidence {format_timestamp(t)} / {format_timestamp(end_sec)}')
    finally:
        cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir/'slot_evidence.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(out_dir/'frame_detection_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(track_rows).to_csv(out_dir/'vehicle_tracks.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(crop_geometry_rows).to_csv(out_dir/'slot_crop_geometry.csv',index=False,encoding='utf-8-sig')

    # No slot-level GT is required for this diagnostic; it simply shows where FULL
    # and AUX agree/disagree so weak slots can be inspected quickly.
    if not df.empty:
        diag = df.groupby(['cctv','local_id','global_id'],as_index=False).agg(
            samples=('time_sec','count'),
            full_hit_rate=('full_detected','mean'),
            aux_hit_rate=('aux_crop_detected','mean'),
            full_mean_conf=('full_det_conf','mean'),
            aux_mean_conf=('aux_crop_det_conf','mean'),
            aux_mean_scale_support=('aux_support_scales','mean'),
        )
        tmp = df.copy()
        tmp['both'] = ((tmp['full_detected']>0)&(tmp['aux_crop_detected']>0)).astype(int)
        tmp['full_only'] = ((tmp['full_detected']>0)&(tmp['aux_crop_detected']==0)).astype(int)
        tmp['aux_only'] = ((tmp['full_detected']==0)&(tmp['aux_crop_detected']>0)).astype(int)
        rates = tmp.groupby(['cctv','local_id','global_id'],as_index=False).agg(
            both_hit_rate=('both','mean'), full_only_rate=('full_only','mean'), aux_only_rate=('aux_only','mean'))
        diag = diag.merge(rates,on=['cctv','local_id','global_id'],how='left')
        diag.to_csv(out_dir/'slot_detector_diagnostics.csv',index=False,encoding='utf-8-sig')
    return df


def _effective_observation(r: pd.Series, params: Dict) -> Dict:
    """v12 evidence fusion.

    HYBRID_ADD is strictly additive: a FULL detection that passed the detector is
    never removed because AUX crop disagrees. AUX may only recover a missing FULL
    detection when it is strong enough or confirmed at multiple crop scales.
    """
    mode = str(params.get('evidence_mode','FULL')).upper()
    full_conf_min = float(params.get('full_min_conf',0.0))
    crop_conf_min = float(params.get('crop_min_conf',0.10))
    crop_add_min_conf = float(params.get('crop_add_min_conf',0.20))
    crop_add_high_conf = float(params.get('crop_add_high_conf',0.45))
    crop_add_min_scales = int(params.get('crop_add_min_scales',2))

    fd = int(r.get('full_detected',r.get('detected',0))) > 0
    fc = float(r.get('full_det_conf',r.get('det_conf',0.0)))
    cd = int(r.get('aux_crop_detected',r.get('crop_detected',0))) > 0
    cc = float(r.get('aux_crop_det_conf',r.get('crop_det_conf',0.0)))
    support = int(r.get('aux_support_scales',1 if cd else 0))
    full_ok = fd and fc >= full_conf_min
    crop_ok = cd and cc >= crop_conf_min
    crop_add_ok = cd and ((support >= crop_add_min_scales and cc >= crop_add_min_conf) or cc >= crop_add_high_conf)

    if mode in ('CROP','CROP_AUX','AUX'):
        detected = crop_ok
        source = 'AUX' if detected else 'NONE'
    elif mode in ('HYBRID','HYBRID_ADD','ADDITIVE'):
        detected = bool(full_ok or crop_add_ok)
        if full_ok:
            source = 'BOTH' if crop_ok else 'FULL'
        else:
            source = 'AUX_ADD' if crop_add_ok else 'NONE'
    else:
        detected = full_ok
        source = 'FULL' if detected else 'NONE'

    # Tracking always comes from FULL because per-slot crop inference has no
    # cross-frame object identity. If AUX alone recovers a miss, slot history still
    # provides the temporal context.
    if full_ok:
        track_id = int(r.get('full_track_id',r.get('track_id',-1)))
        speed = float(r.get('full_track_speed_px_s',r.get('track_speed_px_s',0.0)))
        motion = float(r.get('full_track_motion_span_px',r.get('track_motion_span_px',0.0)))
        cx = float(r.get('full_center_x',r.get('det_center_x',np.nan)))
        cy = float(r.get('full_center_y',r.get('det_center_y',np.nan)))
    else:
        track_id = -1; speed = 0.0; motion = 0.0
        cx = float(r.get('crop_center_x',np.nan)); cy = float(r.get('crop_center_y',np.nan))
    conf = max(fc if full_ok else 0.0, cc if crop_add_ok or mode in ('CROP','CROP_AUX','AUX') else 0.0)
    return {
        'detected':int(detected),'det_conf':float(conf if detected else 0.0),'track_id':int(track_id if detected else -1),
        'speed':float(speed if detected else 0.0),'motion_span':float(motion if detected else 0.0),
        'center_x':float(cx) if detected else np.nan,'center_y':float(cy) if detected else np.nan,
        'source':source,'aux_support_scales':support,
    }


def _state_param_combos(search: Dict) -> List[Dict]:
    """Generate only deployment-relevant combinations.

    CROP_AUX is audit-only in v12, so it is evaluated once later using the best
    FULL temporal settings instead of wasting hundreds of duplicate searches.
    Crop-add thresholds are varied only for HYBRID_ADD because they do not affect FULL.
    """
    combos=[];seen=set()
    requested=[str(x).upper() for x in search.get('evidence_mode',['FULL','CROP_AUX','HYBRID_ADD'])]
    modes=[m for m in requested if m in ('FULL','HYBRID_ADD')]
    if not modes: modes=['FULL']
    crop_add_values=list(search.get('crop_add_min_conf',[0.20,0.30]))
    for mode in modes:
        mode_crop_add = crop_add_values if mode=='HYBRID_ADD' else crop_add_values[:1]
        for entry_ratio,exit_ratio,stable_ratio,vthr,fusion,entry_conf,crop_add_conf in itertools.product(
            search.get('entry_hit_ratio',[0.35,0.50]), search.get('exit_hit_ratio',[0.10,0.20]),
            search.get('stable_track_ratio',[0.55,0.70]), search.get('occupied_visual_diff_threshold',[0.08,0.12,0.16]),
            search.get('global_merge',['ANY']), search.get('entry_mean_conf_min',[0.08,0.15]), mode_crop_add,
        ):
            fusion=str(fusion).upper()
            gvals=search.get('global_confidence_threshold',[0.60]) if fusion.startswith('CONF_') else [0.0]
            for gthr in gvals:
                effective_thr=float(gthr) if fusion.startswith('CONF_') else 0.0
                key=(mode,float(entry_ratio),float(exit_ratio),float(stable_ratio),float(vthr),fusion,effective_thr,float(entry_conf),float(crop_add_conf))
                if key in seen: continue
                seen.add(key)
                combos.append({
                    'evidence_mode':mode,'entry_hit_ratio':float(entry_ratio),'exit_hit_ratio':float(exit_ratio),
                    'stable_track_ratio':float(stable_ratio),'occupied_visual_diff_threshold':float(vthr),
                    'global_merge':fusion,'global_confidence_threshold':effective_thr,
                    'entry_mean_conf_min':float(entry_conf),
                    'full_min_conf':float(search.get('full_min_conf',[0.0])[0] if isinstance(search.get('full_min_conf',[0.0]),list) else search.get('full_min_conf',0.0)),
                    'crop_min_conf':float(search.get('crop_min_conf',[0.10])[0] if isinstance(search.get('crop_min_conf',[0.10]),list) else search.get('crop_min_conf',0.10)),
                    'crop_add_min_conf':float(crop_add_conf),
                    'crop_add_high_conf':float(search.get('crop_add_high_conf',[0.45])[0] if isinstance(search.get('crop_add_high_conf',[0.45]),list) else search.get('crop_add_high_conf',0.45)),
                    'crop_add_min_scales':int(search.get('crop_add_min_scales',[2])[0] if isinstance(search.get('crop_add_min_scales',[2]),list) else search.get('crop_add_min_scales',2)),
                })
    return combos


def _v12_eval_key(d: Dict, slot_eval: Optional[Dict]=None):
    if slot_eval is not None:
        sd=slot_eval['DEV']
        return (-sd['all_slots_exact_time_rate'],sd['false_empty_rate'],-sd['slot_accuracy'],-d['exact_rate'],d['mae'],d['max_abs_error'])
    return (-d['exact_rate'],d['mae'],d['false_empty_bias_rate'],d['max_abs_error'],d['over_rate'])


def search_state_parameters(evidence: pd.DataFrame, gt: pd.DataFrame, settings: Dict, output_dir: str,
                            progress=None, slot_gt_events_path: Optional[str]=None) -> Dict:
    """v14 compact DEV search with FULL deployment baseline.

    CROP_AUX remains in the leaderboard for research, but only FULL and
    HYBRID_ADD may be selected for deployment. HYBRID_ADD must clearly beat FULL
    on DEV without increasing over-count bias beyond a small allowance.
    """
    out_dir=ensure_dir(output_dir);search=settings['state_search'];combos=_state_param_combos(search);temporal=settings.get('temporal',{})
    fixed={'temporal_window_sec':float(temporal.get('window_sec',10.0)),'recent_window_sec':float(temporal.get('recent_window_sec',3.0)),
           'stationary_speed_px_s':float(temporal.get('stationary_speed_px_s',14.0)),'stationary_motion_span_px':float(temporal.get('stationary_motion_span_px',32.0)),
           'crop_stationary_motion_span_px':float(temporal.get('crop_stationary_motion_span_px',28.0)),
           'max_track_switches_for_entry':int(temporal.get('max_track_switches_for_entry',1)),
           'zone_motion_step_px':float(temporal.get('zone_motion_step_px',14.0)),
           'entry_zone_settle_sec':float(temporal.get('entry_zone_settle_sec',6.0)),
           'aux_recovery_conf':float(temporal.get('aux_recovery_conf',0.25)),
           'aux_recovery_min_scales':int(temporal.get('aux_recovery_min_scales',2))}
    dev_end=float(settings.get('dev_end_sec',900));warmup=float(settings.get('evaluation_warmup_sec',temporal.get('warmup_sec',10.0)))
    leaderboard=[];slot_events=None;best_by_mode={}
    if slot_gt_events_path and Path(slot_gt_events_path).exists():slot_events=load_slot_gt_events(slot_gt_events_path)
    gt_times=gt['time_sec'].tolist()
    for i,base in enumerate(combos):
        params={**base,**fixed};local_df,global_df,transitions=run_state_engine(evidence,params);ev=evaluate_state_output(global_df,gt,dev_end,warmup);d=ev['DEV'];te=ev['TEST']
        slot_eval=evaluate_slot_level(global_df,slot_events,gt_times,dev_end,warmup) if slot_events is not None else None
        key=_v12_eval_key(d,slot_eval)
        mode=str(params['evidence_mode']).upper()
        row={**params,'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],'dev_under_rate':d['false_empty_bias_rate'],
             'dev_over_rate':d['over_rate'],'dev_max_abs_error':d['max_abs_error'],
             'test_N':te['N'],'test_exact_rate':te['exact_rate'],'test_MAE':te['mae'],'test_under_rate':te['false_empty_bias_rate'],
             'test_over_rate':te['over_rate'],'test_max_abs_error':te['max_abs_error']}
        if slot_eval is not None:row.update({'dev_slot_accuracy':slot_eval['DEV']['slot_accuracy'],'dev_slot_false_empty_rate':slot_eval['DEV']['false_empty_rate'],'dev_all_slots_exact_time_rate':slot_eval['DEV']['all_slots_exact_time_rate']})
        leaderboard.append(row)
        if mode not in best_by_mode or key < best_by_mode[mode]['key']:
            best_by_mode[mode]={'key':key,'params':params,'outputs':(local_df,global_df,transitions,ev,slot_eval),'row':row}
        if progress and (i%max(1,len(combos)//20)==0 or i==len(combos)-1):progress((i+1)/len(combos),f'v14 slot-zone temporal search {i+1}/{len(combos)}')

    # Audit-only CROP_AUX: use the best FULL temporal settings so it is directly
    # comparable without spending hundreds of search iterations on a mode that can
    # never be deployed automatically.
    if 'CROP_AUX' in [str(x).upper() for x in search.get('evidence_mode',[])] and 'FULL' in best_by_mode:
        audit_params=dict(best_by_mode['FULL']['params']);audit_params['evidence_mode']='CROP_AUX'
        local_df,global_df,transitions=run_state_engine(evidence,audit_params);ev=evaluate_state_output(global_df,gt,dev_end,warmup);d=ev['DEV'];te=ev['TEST']
        slot_eval=evaluate_slot_level(global_df,slot_events,gt_times,dev_end,warmup) if slot_events is not None else None
        row={**audit_params,'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],'dev_under_rate':d['false_empty_bias_rate'],
             'dev_over_rate':d['over_rate'],'dev_max_abs_error':d['max_abs_error'],
             'test_N':te['N'],'test_exact_rate':te['exact_rate'],'test_MAE':te['mae'],'test_under_rate':te['false_empty_bias_rate'],
             'test_over_rate':te['over_rate'],'test_max_abs_error':te['max_abs_error']}
        if slot_eval is not None:row.update({'dev_slot_accuracy':slot_eval['DEV']['slot_accuracy'],'dev_slot_false_empty_rate':slot_eval['DEV']['false_empty_rate'],'dev_all_slots_exact_time_rate':slot_eval['DEV']['all_slots_exact_time_rate']})
        leaderboard.append(row);best_by_mode['CROP_AUX']={'key':_v12_eval_key(d,slot_eval),'params':audit_params,'outputs':(local_df,global_df,transitions,ev,slot_eval),'row':row}

    sort_cols=['dev_exact_rate','dev_MAE','dev_under_rate','dev_max_abs_error','dev_over_rate'];asc=[False,True,True,True,True]
    if slot_events is not None:sort_cols=['dev_all_slots_exact_time_rate','dev_slot_false_empty_rate','dev_slot_accuracy']+sort_cols;asc=[False,True,False]+asc
    board=pd.DataFrame(leaderboard).sort_values(sort_cols,ascending=asc).reset_index(drop=True);board.to_csv(out_dir/'state_search_leaderboard.csv',index=False,encoding='utf-8-sig')

    mode_rows=[]
    for mode in ['FULL','CROP_AUX','HYBRID_ADD']:
        if mode in best_by_mode: mode_rows.append(best_by_mode[mode]['row'])
    pd.DataFrame(mode_rows).to_csv(out_dir/'detection_mode_comparison.csv',index=False,encoding='utf-8-sig')

    eligible=[str(x).upper() for x in search.get('selection_eligible_modes',['FULL','HYBRID_ADD'])]
    full=best_by_mode.get('FULL')
    hybrid=best_by_mode.get('HYBRID_ADD')
    guard=settings.get('mode_selection_guardrail',{})
    min_exact_gain=float(guard.get('min_dev_exact_gain',0.08))
    min_mae_gain=float(guard.get('min_dev_mae_gain',0.05))
    max_over_increase=float(guard.get('max_dev_over_rate_increase',0.02))
    chosen=None;reason=''
    if full and 'FULL' in eligible:
        chosen=full;reason='FULL baseline selected by default.'
        if hybrid and 'HYBRID_ADD' in eligible:
            f=full['row'];h=hybrid['row']
            qualifies=(h['dev_exact_rate'] >= f['dev_exact_rate'] + min_exact_gain and
                       h['dev_MAE'] <= f['dev_MAE'] - min_mae_gain and
                       h['dev_over_rate'] <= f['dev_over_rate'] + max_over_increase)
            if qualifies:
                chosen=hybrid;reason='HYBRID_ADD passed FULL-first DEV guardrail.'
            else:
                reason='FULL retained: HYBRID_ADD did not clear the required DEV gain/MAE/over-count guardrail.'
    else:
        candidates=[best_by_mode[m] for m in eligible if m in best_by_mode]
        if candidates: chosen=min(candidates,key=lambda x:x['key']);reason='FULL unavailable; best eligible DEV mode selected.'
    if chosen is None:
        chosen=min(best_by_mode.values(),key=lambda x:x['key']);reason='Fallback: best available DEV mode selected.'

    best=chosen['params'];local_df,global_df,transitions,ev,slot_eval=chosen['outputs']
    local_df.to_csv(out_dir/'selected_slot_timeseries.csv',index=False,encoding='utf-8-sig');global_df.to_csv(out_dir/'selected_global_slot_timeseries.csv',index=False,encoding='utf-8-sig')
    transitions.to_csv(out_dir/'selected_state_transitions.csv',index=False,encoding='utf-8-sig');ev['timeseries'].to_csv(out_dir/'selected_count_timeseries.csv',index=False,encoding='utf-8-sig')
    save_json(out_dir/'selected_state_params.json',best);pd.DataFrame([{'split':sp,**ev[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'metrics_dev_test.csv',index=False,encoding='utf-8-sig')
    errors=ev['timeseries'][(ev['timeseries']['split']=='TEST')&(ev['timeseries']['error']!=0)].copy();errors.to_csv(out_dir/'test_error_cases.csv',index=False,encoding='utf-8-sig')
    selection_audit={
        'selected_mode':best.get('evidence_mode'),'reason':reason,
        'guardrail':{'min_dev_exact_gain':min_exact_gain,'min_dev_mae_gain':min_mae_gain,'max_dev_over_rate_increase':max_over_increase},
        'full_dev':full['row'] if full else None,'hybrid_add_dev':hybrid['row'] if hybrid else None,
        'crop_aux_is_audit_only':True,
    }
    save_json(out_dir/'mode_selection_audit.json',selection_audit)
    if slot_eval is not None:
        slot_eval['rows'].to_csv(out_dir/'slot_level_comparison.csv',index=False,encoding='utf-8-sig');slot_eval['by_slot'].to_csv(out_dir/'slot_level_by_slot.csv',index=False,encoding='utf-8-sig')
        slot_eval['rows'][(slot_eval['rows']['split']=='TEST')&(slot_eval['rows']['correct']==0)].to_csv(out_dir/'slot_level_test_errors.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{'split':sp,**slot_eval[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'slot_level_metrics.csv',index=False,encoding='utf-8-sig')
    with (out_dir/'REPORT.txt').open('w',encoding='utf-8') as f:
        f.write('Parking Slot State Engine v14 - FULL baseline + slot-zone memory + recovery-only AUX\n\n')
        f.write(f'Warm-up excluded: 0 <= t < {warmup:.1f}s\nEvery decision uses only current/past frames.\nCandidate slots are advisory only and never auto-added.\n')
        f.write('IMPORTANT: AUX crop is recovery insurance for occupied slots; NEW occupancy requires FULL evidence.\n\n')
        f.write('SELECTION GUARDRAIL\n'+reason+'\n')
        f.write(f'min_dev_exact_gain={min_exact_gain}\nmin_dev_mae_gain={min_mae_gain}\nmax_dev_over_rate_increase={max_over_increase}\n\nSELECTED PARAMETERS\n')
        for k,v in best.items():f.write(f'{k}={v}\n')
        f.write('\nDETECTION MODE COMPARISON (v14 deployment uses FULL; AUX is recovery-only insurance)\n')
        for mr in mode_rows:
            f.write(f"{mr['evidence_mode']}: DEV exact={mr['dev_exact_rate']*100:.2f}% MAE={mr['dev_MAE']:.4f} | TEST exact={mr['test_exact_rate']*100:.2f}% MAE={mr['test_MAE']:.4f}\n")
        for sp in ['DEV','TEST']:
            m=ev[sp];f.write(f'\n[{sp} COUNT]\nN={m["N"]}\nExact match rate={m["exact_rate"]*100:.2f}%\nMAE={m["mae"]:.4f}\nUnder-count rate={m["false_empty_bias_rate"]*100:.2f}%\nOver-count rate={m["over_rate"]*100:.2f}%\nMax absolute error={m["max_abs_error"]}\n')
            if slot_eval is not None:
                sm=slot_eval[sp];f.write(f'[{sp} SLOT]\nSlot accuracy={sm["slot_accuracy"]*100:.2f}%\nFalse-empty rate={sm["false_empty_rate"]*100:.2f}%\nFalse-occupied rate={sm["false_occupied_rate"]*100:.2f}%\nAll-slots-exact time rate={sm["all_slots_exact_time_rate"]*100:.2f}%\n')
    return {'best_params':best,'evaluation':ev,'slot_evaluation':slot_eval,'leaderboard':board,'mode_comparison':pd.DataFrame(mode_rows),'selection_audit':selection_audit,'output_dir':str(out_dir)}


def write_camera_detector_metrics(frame_summary_path: str | Path, gt: pd.DataFrame, settings: Dict, output_dir: str | Path) -> Dict:
    """Compare raw FULL-CCTV vehicle counts with the per-camera GT columns.

    This isolates detector quality from slot-state logic and makes it obvious which
    CCTV still needs better vehicle detection.
    """
    out_dir=ensure_dir(output_dir)
    p=Path(frame_summary_path)
    if not p.exists():
        return {'rows':pd.DataFrame(),'metrics':pd.DataFrame()}
    fs=pd.read_csv(p,encoding='utf-8-sig')
    rows=[];metrics=[]
    dev_end=float(settings.get('dev_end_sec',900));warmup=float(settings.get('evaluation_warmup_sec',10.0))
    for cctv in sorted(fs['cctv'].astype(str).unique()) if not fs.empty and 'cctv' in fs.columns else []:
        gt_col=f'{cctv}_count'
        if gt_col not in gt.columns:
            continue
        ss=fs[fs['cctv'].astype(str)==cctv][['time_sec','full_deduplicated_detections']].copy()
        gg=gt[['time_sec','timestamp',gt_col]].copy()
        mm=gg.merge(ss,on='time_sec',how='left')
        mm=mm[mm['time_sec']>=warmup-1e-6].copy()
        mm['pred_visible_vehicle_count']=mm['full_deduplicated_detections'].fillna(0).astype(int)
        mm['gt_visible_vehicle_count']=mm[gt_col].astype(int)
        mm['error']=mm['pred_visible_vehicle_count']-mm['gt_visible_vehicle_count']
        mm['abs_error']=mm['error'].abs();mm['cctv']=cctv
        mm['split']=np.where(mm['time_sec']<dev_end,'DEV','TEST')
        rows.append(mm[['time_sec','timestamp','cctv','split','gt_visible_vehicle_count','pred_visible_vehicle_count','error','abs_error']])
        for sp in ['DEV','TEST']:
            part=mm[mm['split']==sp]
            if part.empty:continue
            metrics.append({
                'cctv':cctv,'split':sp,'N':int(len(part)),
                'exact_rate':float((part['error']==0).mean()),'MAE':float(part['abs_error'].mean()),
                'under_rate':float((part['error']<0).mean()),'over_rate':float((part['error']>0).mean()),
                'max_abs_error':int(part['abs_error'].max()),
            })
    rows_df=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    metrics_df=pd.DataFrame(metrics)
    rows_df.to_csv(out_dir/'camera_detector_timeseries.csv',index=False,encoding='utf-8-sig')
    metrics_df.to_csv(out_dir/'camera_detector_metrics.csv',index=False,encoding='utf-8-sig')
    return {'rows':rows_df,'metrics':metrics_df}

# =============================================================================
# v13 selective auxiliary recheck + safer causal state override
# FULL remains the primary detector. Per-slot crop YOLO is called only when the
# FULL result is ambiguous, recently disappeared, or due for a sparse audit.
# =============================================================================

def _v13_interval_due(t: float, interval_sec: float, sample_sec: float) -> bool:
    interval = max(float(sample_sec), float(interval_sec))
    stride = max(1, int(round(interval / max(1e-6, float(sample_sec)))))
    idx = int(round(float(t) / max(1e-6, float(sample_sec))))
    return (idx % stride) == 0


def extract_evidence(video_path: str, rois: Dict[str,Sequence[int]], slots_path: str, settings: Dict,
                     learn_dir: str, output_dir: str, progress=None) -> pd.DataFrame:
    """v13 evidence extraction: FULL every sample + selective AUX only on ambiguity.

    FULL-CCTV YOLO is always evaluated and tracked. Slot-crop YOLO is expensive, so
    it is invoked only for slots that satisfy one of these causal triggers:
      * weak FULL confidence,
      * a sudden FULL miss after recent positive history,
      * an occupied-looking slot that FULL currently misses,
      * a sparse periodic audit of a FULL-negative slot.
    AUX is strictly additive and never deletes a FULL detection.
    """
    out_dir = ensure_dir(output_dir)
    slots = load_slots(slots_path)
    detector_cfgs = settings.get('detector_by_cctv', {})
    full_detectors = {c: VehicleDetector(detector_cfgs.get(c, settings['detector'])) for c in rois}
    crop_cfg = settings.get('slot_crop_detector', {})
    selective = settings.get('selective_aux', {})
    crop_detectors = {
        c: VehicleDetector(_crop_detector_config(detector_cfgs.get(c, settings['detector']), crop_cfg))
        for c in rois
    }
    matching_cfg = settings.get('matching', {})
    temporal_cfg = settings.get('temporal', {})
    trackers = {c: _OnlineCentroidTracker(temporal_cfg) for c in rois}
    end_sec = float(settings.get('eval_end_sec', 1230))
    sample_sec = float(settings.get('evidence_sample_sec', 1.0))
    times = np.arange(float(settings.get('inference_start_sec',0.0)), end_sec + 1e-6, sample_sec).tolist()
    by_cctv = defaultdict(list)
    for s in slots:
        by_cctv[s['cctv']].append(s)

    templates = {}
    learn_path = Path(learn_dir)
    missing_templates=[]
    invalid_regions=[]
    for s in slots:
        lid=str(s['local_id'])
        p = _resolve_template_path(s, learn_path)
        templates[lid] = cv2.imread(str(p)) if p else None
        if templates[lid] is None:
            missing_templates.append(lid)
        reg=s.get('region')
        if not (isinstance(reg,(list,tuple)) and len(reg)==4):
            invalid_regions.append(lid)
    # v16.2 guard: normalized_visual_diff() historically returned 1.0 for a missing
    # reference template. Allowing evidence extraction to continue would turn a cache/
    # metadata bug into a false visual-change signal for every frame. Fail loudly so
    # ALL-IN-ONE can repair/relearn instead of producing poisoned evidence.
    if missing_templates or invalid_regions:
        raise RuntimeError(
            'Slot-learning metadata is incomplete before evidence extraction. '
            f'missing_templates={missing_templates[:12]} invalid_regions={invalid_regions[:12]}. '
            'Re-run slot learning or ALL-IN-ONE cache repair.'
        )

    history_sec = float(selective.get('history_sec', 10.0))
    weak_full_conf = float(selective.get('weak_full_conf', 0.18))
    recent_miss_trigger_ratio = float(selective.get('recent_miss_trigger_ratio', 0.25))
    occupied_recheck_sec = float(selective.get('occupied_recheck_interval_sec', 2.0))
    appearance_recheck_sec = float(selective.get('appearance_recheck_interval_sec', 3.0))
    audit_interval_sec = float(selective.get('periodic_audit_interval_sec', 10.0))
    appearance_thr = float(selective.get('appearance_trigger_threshold', 0.08))
    always_audit_initial_occupied = bool(selective.get('always_audit_initial_occupied', True))

    cap = open_video(video_path)
    rows, summary, track_rows, crop_geometry_rows = [], [], [], []
    crop_geoms: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    full_hist: Dict[str, deque] = defaultdict(deque)
    last_full_positive_t: Dict[str, float] = defaultdict(lambda: -1e9)
    occupied_memory_sec = float(selective.get('occupied_memory_sec', 12.0))
    occupied_memory_recheck_sec = float(selective.get('occupied_memory_recheck_interval_sec', 2.0))
    batch_size = int(crop_cfg.get('batch_size', 12))
    try:
        for ti, t in enumerate(times):
            frame = read_frame_at(cap, t)
            for cctv, roi in rois.items():
                img = crop_roi(frame, roi)
                cslots = by_cctv.get(cctv, [])
                if not crop_geoms.get(cctv):
                    for s in cslots:
                        lid = str(s['local_id'])
                        gs = _v12_multiscale_crop_geometries(s, cslots, img.shape, crop_cfg)
                        crop_geoms[cctv][lid] = gs
                        for gi, g in enumerate(gs):
                            crop_geometry_rows.append({
                                'cctv': cctv, 'local_id': lid, 'global_id': s.get('global_id', lid),
                                'scale_index': gi, **g,
                            })

                # FULL detector is always on.
                full_dets = full_detectors[cctv].detect(img)
                track_meta = trackers[cctv].update(full_dets, float(t))
                assigned, unmatched, ameta = assign_detections_to_slots(full_dets, cslots, matching_cfg)
                assigned_by_index = {
                    int(meta.get('det_index')): lid
                    for lid, meta in ameta.items() if meta.get('det_index') is not None
                }

                # Precompute slot appearance and update causal FULL history.
                pre = {}
                for s in cslots:
                    lid = str(s['local_id'])
                    fd = assigned.get(lid)
                    fmeta = ameta.get(lid, {})
                    region = s.get('region') or robust_region([], s['point'], img.shape)
                    rx1, ry1, rx2, ry2 = clip_box(region, img.shape[1], img.shape[0])
                    current = img[ry1:ry2, rx1:rx2]
                    diff = normalized_visual_diff(current, templates.get(lid))
                    h = full_hist[lid]
                    if fd is not None:
                        last_full_positive_t[lid] = float(t)
                    h.append({'t': float(t), 'hit': int(fd is not None), 'conf': float(fd.conf) if fd else 0.0})
                    while h and float(t) - float(h[0]['t']) > history_sec + 1e-6:
                        h.popleft()
                    recent_hit_ratio = sum(int(x['hit']) for x in h) / max(1, len(h))
                    init = str(s.get('initial_state', 'UNKNOWN')).upper()
                    appearance_occ = False if settings.get('unlabeled_restart',False) else _appearance_occupied(init, diff, appearance_thr)
                    pre[lid] = {
                        'slot': s, 'fd': fd, 'fmeta': fmeta, 'diff': diff,
                        'recent_full_hit_ratio': float(recent_hit_ratio),
                        'appearance_occ': bool(appearance_occ),
                    }

                # Decide which slots deserve an expensive AUX recheck.
                requested = []
                trigger_reason: Dict[str, str] = {}
                for s in cslots:
                    lid = str(s['local_id'])
                    item = pre[lid]
                    fd = item['fd']
                    init = str(s.get('initial_state', 'UNKNOWN')).upper()
                    reason = ''
                    if fd is not None and float(fd.conf) < weak_full_conf:
                        reason = 'WEAK_FULL'
                    elif fd is None:
                        if item['recent_full_hit_ratio'] >= recent_miss_trigger_ratio:
                            reason = 'RECENT_FULL_MISS'
                        elif (float(t) - float(last_full_positive_t.get(lid, -1e9)) <= occupied_memory_sec
                              and _v13_interval_due(t, occupied_memory_recheck_sec, sample_sec)):
                            reason = 'OCCUPIED_MEMORY_RECHECK'
                        elif item['appearance_occ'] and _v13_interval_due(t, appearance_recheck_sec, sample_sec):
                            reason = 'APPEARANCE_OCCUPIED'
                        elif always_audit_initial_occupied and init == 'OCCUPIED' and _v13_interval_due(t, occupied_recheck_sec, sample_sec):
                            reason = 'INITIAL_OCCUPIED_RECHECK'
                        elif _v13_interval_due(t, audit_interval_sec, sample_sec):
                            reason = 'PERIODIC_AUDIT'
                    if reason:
                        requested.append(s)
                        trigger_reason[lid] = reason

                crop_images, crop_keys = [], []
                for s in requested:
                    lid = str(s['local_id'])
                    for gi, g in enumerate(crop_geoms[cctv][lid]):
                        crop_images.append(img[g['y1']:g['y2'], g['x1']:g['x2']].copy())
                        crop_keys.append((s, gi, g))
                crop_batches = _batched_crop_detect(crop_detectors[cctv], crop_images, batch_size) if crop_images else []

                per_slot_scale: Dict[str, List[Tuple[int, Dict, Optional[Detection], Dict]]] = defaultdict(list)
                for (s, gi, g), det_list in zip(crop_keys, crop_batches):
                    lid = str(s['local_id'])
                    d, meta = _best_slot_crop_detection(det_list, g, s, crop_cfg)
                    per_slot_scale[lid].append((gi, g, d, meta))

                crop_best: Dict[str, Optional[Detection]] = {}
                crop_meta: Dict[str, Dict] = {}
                for s in cslots:
                    lid = str(s['local_id'])
                    items = per_slot_scale.get(lid, [])
                    positives = [x for x in items if x[2] is not None]
                    support = len(positives)
                    if positives:
                        gi, g, d, meta = max(positives, key=lambda x: float(x[3].get('crop_score', 0.0)))
                        confs = [float(x[2].conf) for x in positives]
                        meta = dict(meta)
                        meta.update({
                            'aux_support_scales': support, 'aux_total_scales': len(items),
                            'aux_best_scale': float(g.get('scale', 1.0)), 'aux_best_scale_index': int(gi),
                            'aux_mean_conf_supported': float(np.mean(confs)) if confs else 0.0,
                            'aux_max_conf_supported': float(max(confs)) if confs else 0.0,
                            'best_geom': g, 'per_scale': items,
                        })
                        crop_best[lid] = d; crop_meta[lid] = meta
                    else:
                        fallback = crop_geoms[cctv][lid][0]
                        crop_best[lid] = None
                        crop_meta[lid] = {
                            'crop_score': 0.0, 'crop_center_distance_px': np.nan,
                            'aux_support_scales': 0, 'aux_total_scales': len(items),
                            'aux_best_scale': float(fallback.get('scale', 1.0)), 'aux_best_scale_index': 0,
                            'aux_mean_conf_supported': 0.0, 'aux_max_conf_supported': 0.0,
                            'best_geom': fallback, 'per_scale': items,
                        }

                global_by_local = {str(s['local_id']): str(s.get('global_id', s['local_id'])) for s in cslots}
                reason_counts = Counter(trigger_reason.values())
                summary.append({
                    'time_sec': round(float(t),3), 'timestamp': format_timestamp(t), 'cctv': cctv,
                    'full_deduplicated_detections': len(full_dets), 'full_assigned_detections': len(assigned),
                    'full_unmatched_detections': len(unmatched),
                    'aux_requested_slots': len(requested), 'aux_crop_inferences': len(crop_images),
                    'aux_crop_positive_slots': sum(1 for d in crop_best.values() if d is not None),
                    'aux_crop_multi_scale_confirmed_slots': sum(1 for lid in crop_meta if int(crop_meta[lid].get('aux_support_scales',0)) >= 2),
                    'aux_trigger_weak_full': int(reason_counts.get('WEAK_FULL',0)),
                    'aux_trigger_recent_miss': int(reason_counts.get('RECENT_FULL_MISS',0)),
                    'aux_trigger_occupied_memory': int(reason_counts.get('OCCUPIED_MEMORY_RECHECK',0)),
                    'aux_trigger_appearance': int(reason_counts.get('APPEARANCE_OCCUPIED',0)),
                    'aux_trigger_periodic': int(reason_counts.get('PERIODIC_AUDIT',0)),
                    'active_track_detections': len(track_meta),
                })

                for di, d in enumerate(full_dets):
                    tm = track_meta.get(di,{})
                    lid = assigned_by_index.get(di,'')
                    dc = box_center(d.box)
                    track_rows.append({
                        'time_sec':round(float(t),3),'timestamp':format_timestamp(t),'cctv':cctv,
                        'track_id':int(tm.get('track_id',-1)), 'track_age_sec':float(tm.get('track_age_sec',0.0)),
                        'track_speed_px_s':float(tm.get('track_speed_px_s',0.0)),
                        'track_motion_span_px':float(tm.get('track_motion_span_px',0.0)),
                        'det_center_x':float(dc[0]),'det_center_y':float(dc[1]),'det_conf':float(d.conf),
                        'assigned_local_id':lid,'assigned_global_id':global_by_local.get(lid,'') if lid else '',
                    })

                for s in cslots:
                    lid = str(s['local_id'])
                    item = pre[lid]
                    fd = item['fd']; fmeta = item['fmeta']; diff = item['diff']
                    cd = crop_best.get(lid); cmeta = crop_meta.get(lid,{})
                    fdc = box_center(fd.box) if fd else (np.nan,np.nan)
                    fdet_index = int(fmeta.get('det_index',-1)) if fd else -1
                    tm = track_meta.get(fdet_index,{}) if fd else {}
                    cdc = box_center(cd.box) if cd else (np.nan,np.nan)
                    match_score = float(fmeta.get('match_score',0.0))
                    full_occ_score = (0.55 + 0.45*(0.60*float(fd.conf)+0.40*match_score)) if fd else 0.0
                    crop_occ_score = float(cd.conf) if cd else 0.0
                    bg = cmeta.get('best_geom', crop_geoms[cctv][lid][0])
                    requested_flag = int(lid in trigger_reason)
                    row = {
                        'time_sec':round(float(t),3),'timestamp':format_timestamp(t),'cctv':cctv,
                        'local_id':lid,'global_id':s.get('global_id',lid),'initial_state':s.get('initial_state','UNKNOWN'),
                        'detected':int(fd is not None),'det_conf':float(fd.conf) if fd else 0.0,
                        'det_center_x':float(fdc[0]),'det_center_y':float(fdc[1]),
                        'track_id':int(tm.get('track_id',-1)) if fd else -1,
                        'track_age_sec':float(tm.get('track_age_sec',0.0)) if fd else 0.0,
                        'track_speed_px_s':float(tm.get('track_speed_px_s',0.0)) if fd else 0.0,
                        'track_motion_span_px':float(tm.get('track_motion_span_px',0.0)) if fd else 0.0,
                        'full_detected':int(fd is not None),'full_det_conf':float(fd.conf) if fd else 0.0,
                        'full_center_x':float(fdc[0]),'full_center_y':float(fdc[1]),
                        'full_track_id':int(tm.get('track_id',-1)) if fd else -1,
                        'full_track_age_sec':float(tm.get('track_age_sec',0.0)) if fd else 0.0,
                        'full_track_speed_px_s':float(tm.get('track_speed_px_s',0.0)) if fd else 0.0,
                        'full_track_motion_span_px':float(tm.get('track_motion_span_px',0.0)) if fd else 0.0,
                        'full_match_score':match_score,'full_occupancy_score':float(full_occ_score),
                        'aux_requested':requested_flag,'aux_trigger_reason':trigger_reason.get(lid,'SKIPPED'),
                        'recent_full_hit_ratio':float(item['recent_full_hit_ratio']),
                        'crop_detected':int(cd is not None),'crop_det_conf':float(cd.conf) if cd else 0.0,
                        'crop_center_x':float(cdc[0]),'crop_center_y':float(cdc[1]),
                        'crop_score':float(cmeta.get('crop_score',0.0)),
                        'crop_center_distance_px':float(cmeta.get('crop_center_distance_px',np.nan)),
                        'crop_occupancy_score':float(crop_occ_score),
                        'aux_crop_detected':int(cd is not None),'aux_crop_det_conf':float(cd.conf) if cd else 0.0,
                        'aux_support_scales':int(cmeta.get('aux_support_scales',0)),
                        'aux_total_scales':int(cmeta.get('aux_total_scales',0)),
                        'aux_best_scale':float(cmeta.get('aux_best_scale',1.0)),
                        'aux_mean_conf_supported':float(cmeta.get('aux_mean_conf_supported',0.0)),
                        'aux_max_conf_supported':float(cmeta.get('aux_max_conf_supported',0.0)),
                        'crop_x1':int(bg['x1']),'crop_y1':int(bg['y1']),'crop_x2':int(bg['x2']),'crop_y2':int(bg['y2']),
                        'crop_core_radius_px':float(bg['core_radius_px']),
                        'visual_diff_initial':float(diff),
                    }
                    rows.append(row)
            if progress and (ti % max(1,len(times)//100)==0 or ti==len(times)-1):
                total_req = sum(int(x.get('aux_requested_slots',0)) for x in summary)
                total_slots_seen = max(1, len(summary) * max(1, int(np.mean([len(by_cctv.get(c,[])) for c in rois]))))
                progress((ti+1)/len(times), f'v13 FULL + selective AUX {ti+1}/{len(times)} | AUX requested {total_req} slot-frames')
    finally:
        cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir/'slot_evidence.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(out_dir/'frame_detection_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(track_rows).to_csv(out_dir/'vehicle_tracks.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(crop_geometry_rows).drop_duplicates(['cctv','local_id','scale_index']).to_csv(out_dir/'slot_crop_geometry.csv',index=False,encoding='utf-8-sig')

    if not df.empty:
        diag = df.groupby(['cctv','local_id','global_id'],as_index=False).agg(
            full_hit_rate=('full_detected','mean'), full_mean_conf=('full_det_conf','mean'),
            aux_request_rate=('aux_requested','mean'), aux_hit_rate=('aux_crop_detected','mean'),
            aux_mean_conf=('aux_crop_det_conf','mean'), aux_mean_support=('aux_support_scales','mean'),
            recent_full_hit_ratio_mean=('recent_full_hit_ratio','mean'),
        )
        diag['aux_hit_given_requested'] = 0.0
        for i,r in diag.iterrows():
            sub=df[(df['cctv']==r['cctv'])&(df['local_id']==r['local_id'])&(df['aux_requested']>0)]
            diag.at[i,'aux_hit_given_requested']=float(sub['aux_crop_detected'].mean()) if not sub.empty else 0.0
        diag.to_csv(out_dir/'slot_detector_diagnostics.csv',index=False,encoding='utf-8-sig')
    return df


def run_state_engine(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """v13 causal state engine.

    Differences from v12:
    * a newly dominant Track cannot inherit maneuver-history hits from another Track;
      it must settle on the same slot for a minimum span before entry confirmation,
    * weak low-confidence repeated detections need stronger appearance support,
    * an occupied slot may clear on a short recent no-detection run when appearance
      also supports EMPTY, instead of waiting for the full 10-second window,
    * a recently cleared slot can recover quickly when a strong Track reappears.
    """
    window_sec = float(params.get('temporal_window_sec',10.0))
    recent_sec = float(params.get('recent_window_sec',3.0))
    entry_ratio = float(params.get('entry_hit_ratio',0.50))
    exit_ratio = float(params.get('exit_hit_ratio',0.10))
    stable_track_ratio_req = float(params.get('stable_track_ratio',0.65))
    visual_thr = float(params.get('occupied_visual_diff_threshold',0.12))
    stationary_speed = float(params.get('stationary_speed_px_s',14.0))
    stationary_span = float(params.get('stationary_motion_span_px',32.0))
    crop_stationary_span = float(params.get('crop_stationary_motion_span_px',28.0))
    max_switches = int(params.get('max_track_switches_for_entry',1))
    entry_mean_conf_min = float(params.get('entry_mean_conf_min',0.12))
    entry_high_conf = float(params.get('entry_high_conf_override',0.30))
    entry_settle_sec = float(params.get('entry_track_settle_sec',4.0))
    exit_clear_sec = float(params.get('exit_recent_clear_sec',3.0))
    recovery_window_sec = float(params.get('recovery_window_sec',8.0))
    fusion = str(params.get('global_merge','ANY')).upper()
    global_thr = float(params.get('global_confidence_threshold',0.60))
    evidence_mode = str(params.get('evidence_mode','FULL')).upper()

    evidence = evidence.sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    state={}; phase={}; owner_track={}; prev_local={}; prev_global={}; last_change={}
    slot_hist: Dict[str,deque] = defaultdict(deque); track_hist: Dict[Tuple[str,int],deque] = defaultdict(deque)
    slot_meta = evidence.groupby('local_id',sort=False).first()
    for lid,r in slot_meta.iterrows():
        init=str(r.get('initial_state','UNKNOWN')).upper(); st='OCCUPIED' if init=='OCCUPIED' else 'EMPTY'
        lid=str(lid); state[lid]=st; phase[lid]=st; owner_track[lid]=None; prev_local[lid]=st; last_change[lid]=-1e9

    local_rows=[]; global_rows=[]; transitions=[]
    for t,group in evidence.groupby('time_sec',sort=True):
        t=float(t)
        for _,r in group.iterrows():
            lid=str(r['local_id']); obs=_effective_observation(r,params)
            rec={'t':t,**obs,'visual_diff':float(r.get('visual_diff_initial',0.0)),'cctv':str(r.get('cctv',''))}
            slot_hist[lid].append(rec)
            while slot_hist[lid] and t-float(slot_hist[lid][0]['t'])>window_sec+1e-6: slot_hist[lid].popleft()
            if rec['detected'] and rec['track_id']>=0:
                key=(rec['cctv'],rec['track_id']); track_hist[key].append({'t':t,'lid':lid,'speed':rec['speed'],'motion_span':rec['motion_span'],'conf':rec['det_conf']})
                while track_hist[key] and t-float(track_hist[key][0]['t'])>window_sec+1e-6: track_hist[key].popleft()
        for key in list(track_hist.keys()):
            if not track_hist[key] or t-float(track_hist[key][-1]['t'])>window_sec+1e-6: del track_hist[key]

        owned_by_track={}
        for lid,tid in owner_track.items():
            if tid is None or state.get(lid)!='OCCUPIED': continue
            try:cctv=str(slot_meta.loc[lid].get('cctv',''))
            except Exception:cctv=''
            owned_by_track[(cctv,int(tid))]=lid

        now=[]
        for _,r in group.iterrows():
            lid=str(r['local_id']); cctv=str(r['cctv']); hist=list(slot_hist[lid]); hits=[x for x in hist if x['detected']]
            recent=[x for x in hist if t-float(x['t'])<=recent_sec+1e-6]; recent_hits=[x for x in recent if x['detected']]
            hit_ratio=len(hits)/max(1,len(hist)); recent_hit_ratio=len(recent_hits)/max(1,len(recent))
            mean_conf=float(np.mean([x['det_conf'] for x in hits])) if hits else 0.0
            recent_mean_conf=float(np.mean([x['det_conf'] for x in recent_hits])) if recent_hits else 0.0
            track_counts=Counter(int(x['track_id']) for x in hits if int(x['track_id'])>=0)
            dom_track=track_counts.most_common(1)[0][0] if track_counts else -1
            dom_ratio=(track_counts.get(dom_track,0)/max(1,len(hits))) if dom_track>=0 else 0.0
            track_slot_ratio=0.0; track_switches=0; track_speed=0.0; track_motion=0.0; track_settle_span=0.0
            if dom_track>=0:
                th=list(track_hist.get((cctv,dom_track),[]))
                if th:
                    same=[x for x in th if x['lid']==lid]
                    track_slot_ratio=len(same)/max(1,len(th)); seq=[x['lid'] for x in th]
                    track_switches=sum(1 for a,b in zip(seq,seq[1:]) if a!=b)
                    tr_recent=[x for x in th if t-float(x['t'])<=recent_sec+1e-6]; src=tr_recent if tr_recent else th
                    track_speed=float(np.median([x['speed'] for x in src])) if src else 0.0
                    track_motion=float(max([x['motion_span'] for x in src],default=0.0))
                    if len(same)>=2: track_settle_span=float(max(x['t'] for x in same)-min(x['t'] for x in same))
            centers=[(float(x['center_x']),float(x['center_y'])) for x in hits if np.isfinite(x['center_x']) and np.isfinite(x['center_y'])]
            local_motion=0.0
            if len(centers)>=2:
                mx=float(np.median([x for x,_ in centers])); my=float(np.median([y for _,y in centers]))
                local_motion=max(math.hypot(x-mx,y-my) for x,y in centers)
            init=str(r.get('initial_state','UNKNOWN')).upper(); vis=float(r.get('visual_diff_initial',0.0)); appearance_occ=_appearance_occupied(init,vis,visual_thr)
            stable_track=(dom_track>=0 and track_slot_ratio>=stable_track_ratio_req and dom_ratio>=0.50 and track_switches<=max_switches and track_speed<=stationary_speed and track_motion<=stationary_span and recent_hit_ratio>=0.34)
            stable_slot=(hit_ratio>=max(entry_ratio,0.55) and recent_hit_ratio>=0.50 and local_motion<=crop_stationary_span)
            maneuvering=(dom_track>=0 and (track_switches>max_switches or track_speed>stationary_speed or track_motion>stationary_span)) or (local_motion>crop_stationary_span and hit_ratio>0)
            last_hit_t=max([float(x['t']) for x in hits],default=-1e9)
            no_hit_for=max(0.0,t-last_hit_t) if last_hit_t>-1e8 else window_sec+recent_sec

            entry_gate=False; exit_gate=False
            if state[lid]=='EMPTY':
                recently_cleared=(t-float(last_change.get(lid,-1e9)))<=recovery_window_sec
                reliable_track=(stable_track and track_settle_span>=entry_settle_sec)
                strong_recent=(recent_hit_ratio>=0.66 and recent_mean_conf>=max(entry_mean_conf_min,0.12) and local_motion<=crop_stationary_span)
                confidence_ok=(mean_conf>=entry_mean_conf_min)
                # Low-confidence repetitive ghosts must also look occupied. High-confidence
                # tracks may enter without appearance support because lighting can corrupt it.
                appearance_or_strong=(appearance_occ or recent_mean_conf>=entry_high_conf)
                if dom_track>=0:
                    evidence_stable=reliable_track
                else:
                    evidence_stable=stable_slot and strong_recent
                if recently_cleared and dom_track>=0 and stable_track and recent_mean_conf>=entry_mean_conf_min:
                    # Fast recovery from a detector dropout, but still require the same Track
                    # to be stable in the slot for at least half the normal settle time.
                    evidence_stable = evidence_stable or (track_settle_span>=max(1.0,entry_settle_sec*0.5))
                entry_gate=bool(evidence_stable and hit_ratio>=entry_ratio and confidence_ok and appearance_or_strong)
                conflict=owned_by_track.get((cctv,dom_track)) if dom_track>=0 else None
                if entry_gate and (not conflict or conflict==lid):
                    state[lid]='OCCUPIED'; phase[lid]='OCCUPIED'; owner_track[lid]=dom_track if dom_track>=0 else None; last_change[lid]=t
                    if dom_track>=0: owned_by_track[(cctv,dom_track)]=lid
                elif maneuvering or hit_ratio>0: phase[lid]='MANEUVERING'
                else: phase[lid]='EMPTY'
            else:
                if dom_track>=0 and recent_hit_ratio>0 and (owner_track[lid] is None or stable_track):
                    owner_track[lid]=dom_track; owned_by_track[(cctv,dom_track)]=lid
                conservative_clear=(hit_ratio<=exit_ratio and recent_hit_ratio<=exit_ratio and not appearance_occ)
                fast_clear=(no_hit_for>=exit_clear_sec and recent_hit_ratio<=0.01 and not appearance_occ)
                exit_gate=bool(conservative_clear or fast_clear)
                if exit_gate:
                    old=owner_track[lid]; state[lid]='EMPTY'; phase[lid]='EMPTY'; owner_track[lid]=None; last_change[lid]=t
                    if old is not None and owned_by_track.get((cctv,int(old)))==lid: owned_by_track.pop((cctv,int(old)),None)
                elif maneuvering and recent_hit_ratio>0: phase[lid]='LEAVING' if owner_track[lid]==dom_track else 'OCCUPIED'
                else: phase[lid]='OCCUPIED'

            temporal_score=0.38*hit_ratio+0.18*recent_hit_ratio+0.12*track_slot_ratio+0.15*(1.0 if appearance_occ else 0.0)+0.17*min(1.0,mean_conf)
            occ_score=max(0.51,min(0.99,temporal_score)) if state[lid]=='OCCUPIED' else min(0.49,max(0.01,temporal_score))
            obs_now=_effective_observation(r,params)
            row={'time_sec':t,'timestamp':r['timestamp'],'cctv':cctv,'local_id':lid,'global_id':r['global_id'],
                 'evidence_mode':evidence_mode,'evidence_source':obs_now['source'],'detected':obs_now['detected'],'det_conf':obs_now['det_conf'],
                 'track_id':obs_now['track_id'],'dominant_track_id':int(dom_track),'state':state[lid],'phase':phase[lid],
                 'occupied_score':float(occ_score),'window_hit_ratio':float(hit_ratio),'recent_hit_ratio':float(recent_hit_ratio),
                 'mean_detection_confidence':float(mean_conf),'recent_mean_confidence':float(recent_mean_conf),
                 'dominant_track_ratio_in_slot':float(dom_ratio),'track_slot_ratio':float(track_slot_ratio),
                 'track_switches_window':int(track_switches),'track_speed_px_s':float(track_speed),'track_motion_span_px':float(track_motion),
                 'dominant_track_settle_sec':float(track_settle_span),'time_since_last_detection_sec':float(no_hit_for),
                 'slot_detection_motion_span_px':float(local_motion),'appearance_occupied':int(bool(appearance_occ)),'visual_diff_initial':vis,
                 'entry_gate_ok':int(entry_gate),'exit_gate_ok':int(exit_gate),
                 'owner_track_id':int(owner_track[lid]) if owner_track[lid] is not None else -1,'warmup':int(t<window_sec-1e-6)}
            local_rows.append(row); now.append(row)
            if prev_local.get(lid)!=state[lid]:
                transitions.append({'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,'global_id':r['global_id'],
                    'from_state':prev_local.get(lid,''),'to_state':state[lid],'cctv':cctv,'phase':phase[lid],
                    'dominant_track_id':int(dom_track),'evidence_mode':evidence_mode})
                prev_local[lid]=state[lid]

        by_global=defaultdict(list)
        for x in now: by_global[x['global_id']].append(x)
        for gid,items in by_global.items():
            occ=sum(1 for x in items if x['state']=='OCCUPIED'); scores=[float(x['occupied_score']) for x in items]
            if fusion=='CONF_MAX': gscore=max(scores) if scores else 0.0; global_occ=gscore>=global_thr
            elif fusion=='CONF_MEAN': gscore=float(np.mean(scores)) if scores else 0.0; global_occ=gscore>=global_thr
            elif fusion=='MAJORITY': gscore=float(np.mean(scores)) if scores else 0.0; global_occ=occ>=math.ceil(len(items)/2)
            else: gscore=max(scores) if scores else 0.0; global_occ=occ>0
            gstate='OCCUPIED' if global_occ else 'EMPTY'; phases=[str(x.get('phase','')) for x in items]
            gphase='MANEUVERING' if 'MANEUVERING' in phases else ('LEAVING' if 'LEAVING' in phases else gstate)
            grow={'time_sec':t,'timestamp':items[0]['timestamp'],'global_id':gid,'state':gstate,'phase':gphase,'global_score':gscore,
                  'evidence_mode':evidence_mode,'source_local_slots':';'.join(x['local_id'] for x in items),'occupied_votes':occ,'total_votes':len(items),
                  'warmup':int(t<window_sec-1e-6)}
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid]!=gstate:
                transitions.append({'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,'global_id':gid,
                    'from_state':prev_global[gid],'to_state':gstate,'cctv':'MULTI','phase':gphase,'dominant_track_id':-1,'evidence_mode':evidence_mode})
            prev_global[gid]=gstate
    return pd.DataFrame(local_rows),pd.DataFrame(global_rows),pd.DataFrame(transitions)


def _state_param_combos(search: Dict) -> List[Dict]:
    """v13 compact search: FULL + additive selective AUX only.

    CROP-only was experimentally poor and is no longer searched. The smaller grid
    makes cached temporal re-runs substantially faster.
    """
    combos=[]; seen=set()
    requested=[str(x).upper() for x in search.get('evidence_mode',['FULL','HYBRID_ADD'])]
    modes=[m for m in requested if m in ('FULL','HYBRID_ADD')]
    if not modes: modes=['FULL']
    crop_add_values=list(search.get('crop_add_min_conf',[0.25]))
    for mode in modes:
        cvals=crop_add_values if mode=='HYBRID_ADD' else crop_add_values[:1]
        for entry_ratio,exit_ratio,stable_ratio,vthr,fusion,entry_conf,settle,exit_clear,crop_add_conf in itertools.product(
            search.get('entry_hit_ratio',[0.35,0.50]), search.get('exit_hit_ratio',[0.10,0.20]),
            search.get('stable_track_ratio',[0.55,0.70]), search.get('occupied_visual_diff_threshold',[0.08,0.12]),
            search.get('global_merge',['ANY','CONF_MAX']), search.get('entry_mean_conf_min',[0.12,0.18]),
            search.get('entry_track_settle_sec',[3.0,5.0]), search.get('exit_recent_clear_sec',[2.0,4.0]), cvals,
        ):
            gvals=search.get('global_confidence_threshold',[0.60]) if str(fusion).upper().startswith('CONF_') else [0.0]
            for gthr in gvals:
                key=(mode,entry_ratio,exit_ratio,stable_ratio,vthr,str(fusion).upper(),gthr,entry_conf,settle,exit_clear,crop_add_conf)
                if key in seen: continue
                seen.add(key)
                combos.append({
                    'evidence_mode':mode,'entry_hit_ratio':float(entry_ratio),'exit_hit_ratio':float(exit_ratio),
                    'stable_track_ratio':float(stable_ratio),'occupied_visual_diff_threshold':float(vthr),
                    'global_merge':str(fusion).upper(),'global_confidence_threshold':float(gthr),
                    'entry_mean_conf_min':float(entry_conf),'entry_track_settle_sec':float(settle),
                    'exit_recent_clear_sec':float(exit_clear),
                    'entry_high_conf_override':float(search.get('entry_high_conf_override',[0.30])[0] if isinstance(search.get('entry_high_conf_override',[0.30]),list) else search.get('entry_high_conf_override',0.30)),
                    'recovery_window_sec':float(search.get('recovery_window_sec',[8.0])[0] if isinstance(search.get('recovery_window_sec',[8.0]),list) else search.get('recovery_window_sec',8.0)),
                    'full_min_conf':float(search.get('full_min_conf',[0.0])[0] if isinstance(search.get('full_min_conf',[0.0]),list) else search.get('full_min_conf',0.0)),
                    'crop_min_conf':float(search.get('crop_min_conf',[0.10])[0] if isinstance(search.get('crop_min_conf',[0.10]),list) else search.get('crop_min_conf',0.10)),
                    'crop_add_min_conf':float(crop_add_conf),
                    'crop_add_high_conf':float(search.get('crop_add_high_conf',[0.50])[0] if isinstance(search.get('crop_add_high_conf',[0.50]),list) else search.get('crop_add_high_conf',0.50)),
                    'crop_add_min_scales':int(search.get('crop_add_min_scales',[2])[0] if isinstance(search.get('crop_add_min_scales',[2]),list) else search.get('crop_add_min_scales',2)),
                })
    return combos


# Keep v12 FULL-first selection/reporting logic, but the globally-resolved
# run_state_engine and _state_param_combos above are now the v13 implementations.

# =============================================================================
# v13.1 conservative temporal override
# Preserve the proven v12 state timing, but block low-confidence stationary ghosts
# from becoming occupied when the image appearance does not support occupancy.
# =============================================================================

def run_state_engine(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    window_sec=float(params.get('temporal_window_sec',10.0)); recent_sec=float(params.get('recent_window_sec',3.0))
    entry_ratio=float(params.get('entry_hit_ratio',0.50)); exit_ratio=float(params.get('exit_hit_ratio',0.10))
    stable_track_ratio_req=float(params.get('stable_track_ratio',0.65)); visual_thr=float(params.get('occupied_visual_diff_threshold',0.12))
    stationary_speed=float(params.get('stationary_speed_px_s',14.0)); stationary_span=float(params.get('stationary_motion_span_px',32.0))
    crop_stationary_span=float(params.get('crop_stationary_motion_span_px',28.0)); max_switches=int(params.get('max_track_switches_for_entry',1))
    entry_mean_conf_min=float(params.get('entry_mean_conf_min',0.10)); entry_no_appearance_conf=float(params.get('entry_no_appearance_conf',0.25))
    fusion=str(params.get('global_merge','ANY')).upper(); global_thr=float(params.get('global_confidence_threshold',0.60))
    evidence_mode=str(params.get('evidence_mode','FULL')).upper()

    evidence=evidence.sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    state={}; phase={}; owner_track={}; prev_local={}; prev_global={}
    slot_hist: Dict[str,deque]=defaultdict(deque); track_hist: Dict[Tuple[str,int],deque]=defaultdict(deque)
    slot_meta=evidence.groupby('local_id',sort=False).first()
    for lid,r in slot_meta.iterrows():
        init=str(r.get('initial_state','UNKNOWN')).upper(); st='OCCUPIED' if init=='OCCUPIED' else 'EMPTY'
        lid=str(lid); state[lid]=st; phase[lid]=st; owner_track[lid]=None; prev_local[lid]=st

    local_rows=[]; global_rows=[]; transitions=[]
    for t,group in evidence.groupby('time_sec',sort=True):
        t=float(t)
        for _,r in group.iterrows():
            lid=str(r['local_id']); obs=_effective_observation(r,params)
            rec={'t':t,**obs,'visual_diff':float(r.get('visual_diff_initial',0.0)),'cctv':str(r.get('cctv',''))}
            slot_hist[lid].append(rec)
            while slot_hist[lid] and t-float(slot_hist[lid][0]['t'])>window_sec+1e-6: slot_hist[lid].popleft()
            if rec['detected'] and rec['track_id']>=0:
                key=(rec['cctv'],rec['track_id']); track_hist[key].append({'t':t,'lid':lid,'speed':rec['speed'],'motion_span':rec['motion_span']})
                while track_hist[key] and t-float(track_hist[key][0]['t'])>window_sec+1e-6: track_hist[key].popleft()
        for key in list(track_hist.keys()):
            if not track_hist[key] or t-float(track_hist[key][-1]['t'])>window_sec+1e-6: del track_hist[key]

        owned_by_track={}
        for lid,tid in owner_track.items():
            if tid is None or state.get(lid)!='OCCUPIED': continue
            try:cctv=str(slot_meta.loc[lid].get('cctv',''))
            except Exception:cctv=''
            owned_by_track[(cctv,int(tid))]=lid

        now=[]
        for _,r in group.iterrows():
            lid=str(r['local_id']); cctv=str(r['cctv']); hist=list(slot_hist[lid]); hits=[x for x in hist if x['detected']]
            recent=[x for x in hist if t-float(x['t'])<=recent_sec+1e-6]; recent_hits=[x for x in recent if x['detected']]
            hit_ratio=len(hits)/max(1,len(hist)); recent_hit_ratio=len(recent_hits)/max(1,len(recent))
            mean_conf=float(np.mean([x['det_conf'] for x in hits])) if hits else 0.0
            track_counts=Counter(int(x['track_id']) for x in hits if int(x['track_id'])>=0)
            dom_track=track_counts.most_common(1)[0][0] if track_counts else -1
            dom_ratio=(track_counts.get(dom_track,0)/max(1,len(hits))) if dom_track>=0 else 0.0
            track_slot_ratio=0.0; track_switches=0; track_speed=0.0; track_motion=0.0
            if dom_track>=0:
                th=list(track_hist.get((cctv,dom_track),[]))
                if th:
                    track_slot_ratio=sum(1 for x in th if x['lid']==lid)/max(1,len(th)); seq=[x['lid'] for x in th]
                    track_switches=sum(1 for a,b in zip(seq,seq[1:]) if a!=b)
                    tr_recent=[x for x in th if t-float(x['t'])<=recent_sec+1e-6]; src=tr_recent if tr_recent else th
                    track_speed=float(np.median([x['speed'] for x in src])) if src else 0.0
                    track_motion=float(max([x['motion_span'] for x in src],default=0.0))
            centers=[(float(x['center_x']),float(x['center_y'])) for x in hits if np.isfinite(x['center_x']) and np.isfinite(x['center_y'])]
            local_motion=0.0
            if len(centers)>=2:
                mx=float(np.median([x for x,_ in centers])); my=float(np.median([y for _,y in centers]))
                local_motion=max(math.hypot(x-mx,y-my) for x,y in centers)
            init=str(r.get('initial_state','UNKNOWN')).upper(); vis=float(r.get('visual_diff_initial',0.0)); appearance_occ=_appearance_occupied(init,vis,visual_thr)
            stable_track=(dom_track>=0 and track_slot_ratio>=stable_track_ratio_req and dom_ratio>=0.50 and track_switches<=max_switches and track_speed<=stationary_speed and track_motion<=stationary_span and recent_hit_ratio>=0.34)
            stable_slot=(hit_ratio>=max(entry_ratio,0.55) and recent_hit_ratio>=0.50 and local_motion<=crop_stationary_span)
            maneuvering=(dom_track>=0 and (track_switches>max_switches or track_speed>stationary_speed or track_motion>stationary_span)) or (local_motion>crop_stationary_span and hit_ratio>0)
            ghost_guard_ok=bool(appearance_occ or mean_conf>=entry_no_appearance_conf)
            entry_gate=False; exit_gate=False

            if state[lid]=='EMPTY':
                strong_stationary=(hit_ratio>=max(entry_ratio,0.65) and recent_hit_ratio>=0.66 and local_motion<=crop_stationary_span)
                evidence_stable=stable_track if dom_track>=0 else (stable_slot or strong_stationary)
                entry_gate=bool(evidence_stable and hit_ratio>=entry_ratio and mean_conf>=entry_mean_conf_min and ghost_guard_ok)
                conflict=owned_by_track.get((cctv,dom_track)) if dom_track>=0 else None
                if entry_gate and (not conflict or conflict==lid):
                    state[lid]='OCCUPIED'; phase[lid]='OCCUPIED'; owner_track[lid]=dom_track if dom_track>=0 else None
                    if dom_track>=0: owned_by_track[(cctv,dom_track)]=lid
                elif maneuvering or hit_ratio>0: phase[lid]='MANEUVERING'
                else: phase[lid]='EMPTY'
            else:
                if dom_track>=0 and recent_hit_ratio>0 and (owner_track[lid] is None or stable_track):
                    owner_track[lid]=dom_track; owned_by_track[(cctv,dom_track)]=lid
                clear_empty=(hit_ratio<=exit_ratio and recent_hit_ratio<=exit_ratio and not appearance_occ)
                exit_gate=bool(clear_empty)
                if clear_empty:
                    old=owner_track[lid]; state[lid]='EMPTY'; phase[lid]='EMPTY'; owner_track[lid]=None
                    if old is not None and owned_by_track.get((cctv,int(old)))==lid: owned_by_track.pop((cctv,int(old)),None)
                elif maneuvering and recent_hit_ratio>0: phase[lid]='LEAVING' if owner_track[lid]==dom_track else 'OCCUPIED'
                else: phase[lid]='OCCUPIED'

            temporal_score=0.40*hit_ratio+0.18*recent_hit_ratio+0.12*track_slot_ratio+0.15*(1.0 if appearance_occ else 0.0)+0.15*min(1.0,mean_conf)
            occ_score=max(0.51,min(0.99,temporal_score)) if state[lid]=='OCCUPIED' else min(0.49,max(0.01,temporal_score))
            obs_now=_effective_observation(r,params)
            row={'time_sec':t,'timestamp':r['timestamp'],'cctv':cctv,'local_id':lid,'global_id':r['global_id'],
                 'evidence_mode':evidence_mode,'evidence_source':obs_now['source'],'detected':obs_now['detected'],'det_conf':obs_now['det_conf'],
                 'track_id':obs_now['track_id'],'dominant_track_id':int(dom_track),'state':state[lid],'phase':phase[lid],
                 'occupied_score':float(occ_score),'window_hit_ratio':float(hit_ratio),'recent_hit_ratio':float(recent_hit_ratio),
                 'mean_detection_confidence':float(mean_conf),'dominant_track_ratio_in_slot':float(dom_ratio),'track_slot_ratio':float(track_slot_ratio),
                 'track_switches_window':int(track_switches),'track_speed_px_s':float(track_speed),'track_motion_span_px':float(track_motion),
                 'slot_detection_motion_span_px':float(local_motion),'appearance_occupied':int(bool(appearance_occ)),'visual_diff_initial':vis,
                 'ghost_guard_ok':int(ghost_guard_ok),'entry_gate_ok':int(entry_gate),'exit_gate_ok':int(exit_gate),
                 'owner_track_id':int(owner_track[lid]) if owner_track[lid] is not None else -1,'warmup':int(t<window_sec-1e-6)}
            local_rows.append(row); now.append(row)
            if prev_local.get(lid)!=state[lid]:
                transitions.append({'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,'global_id':r['global_id'],
                    'from_state':prev_local.get(lid,''),'to_state':state[lid],'cctv':cctv,'phase':phase[lid],
                    'dominant_track_id':int(dom_track),'evidence_mode':evidence_mode})
                prev_local[lid]=state[lid]

        by_global=defaultdict(list)
        for x in now: by_global[x['global_id']].append(x)
        for gid,items in by_global.items():
            occ=sum(1 for x in items if x['state']=='OCCUPIED'); scores=[float(x['occupied_score']) for x in items]
            if fusion=='CONF_MAX': gscore=max(scores) if scores else 0.0; global_occ=gscore>=global_thr
            elif fusion=='CONF_MEAN': gscore=float(np.mean(scores)) if scores else 0.0; global_occ=gscore>=global_thr
            elif fusion=='MAJORITY': gscore=float(np.mean(scores)) if scores else 0.0; global_occ=occ>=math.ceil(len(items)/2)
            else: gscore=max(scores) if scores else 0.0; global_occ=occ>0
            gstate='OCCUPIED' if global_occ else 'EMPTY'; phases=[str(x.get('phase','')) for x in items]
            gphase='MANEUVERING' if 'MANEUVERING' in phases else ('LEAVING' if 'LEAVING' in phases else gstate)
            grow={'time_sec':t,'timestamp':items[0]['timestamp'],'global_id':gid,'state':gstate,'phase':gphase,'global_score':gscore,
                  'evidence_mode':evidence_mode,'source_local_slots':';'.join(x['local_id'] for x in items),'occupied_votes':occ,'total_votes':len(items),
                  'warmup':int(t<window_sec-1e-6)}
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid]!=gstate:
                transitions.append({'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,'global_id':gid,
                    'from_state':prev_global[gid],'to_state':gstate,'cctv':'MULTI','phase':gphase,'dominant_track_id':-1,'evidence_mode':evidence_mode})
            prev_global[gid]=gstate
    return pd.DataFrame(local_rows),pd.DataFrame(global_rows),pd.DataFrame(transitions)


def _state_param_combos(search: Dict) -> List[Dict]:
    """v13 compact DEV search. CROP-only is removed; FULL is the baseline."""
    combos=[]; seen=set()
    modes=[m for m in [str(x).upper() for x in search.get('evidence_mode',['FULL','HYBRID_ADD'])] if m in ('FULL','HYBRID_ADD')]
    if not modes: modes=['FULL']
    crop_vals=list(search.get('crop_add_min_conf',[0.25]))
    for mode in modes:
        cv=crop_vals if mode=='HYBRID_ADD' else crop_vals[:1]
        for entry_ratio,exit_ratio,stable_ratio,vthr,entry_conf,crop_add_conf in itertools.product(
            search.get('entry_hit_ratio',[0.35,0.50]), search.get('exit_hit_ratio',[0.10,0.20]),
            search.get('stable_track_ratio',[0.55,0.70]), search.get('occupied_visual_diff_threshold',[0.08,0.12]),
            search.get('entry_mean_conf_min',[0.08,0.12,0.15]), cv,
        ):
            key=(mode,float(entry_ratio),float(exit_ratio),float(stable_ratio),float(vthr),float(entry_conf),float(crop_add_conf))
            if key in seen: continue
            seen.add(key)
            combos.append({
                'evidence_mode':mode,'entry_hit_ratio':float(entry_ratio),'exit_hit_ratio':float(exit_ratio),
                'stable_track_ratio':float(stable_ratio),'occupied_visual_diff_threshold':float(vthr),
                'global_merge':'ANY','global_confidence_threshold':0.0,'entry_mean_conf_min':float(entry_conf),
                'entry_no_appearance_conf':float(search.get('entry_no_appearance_conf',[0.25])[0] if isinstance(search.get('entry_no_appearance_conf',[0.25]),list) else search.get('entry_no_appearance_conf',0.25)),
                'full_min_conf':float(search.get('full_min_conf',[0.0])[0] if isinstance(search.get('full_min_conf',[0.0]),list) else search.get('full_min_conf',0.0)),
                'crop_min_conf':float(search.get('crop_min_conf',[0.10])[0] if isinstance(search.get('crop_min_conf',[0.10]),list) else search.get('crop_min_conf',0.10)),
                'crop_add_min_conf':float(crop_add_conf),
                'crop_add_high_conf':float(search.get('crop_add_high_conf',[0.50])[0] if isinstance(search.get('crop_add_high_conf',[0.50]),list) else search.get('crop_add_high_conf',0.50)),
                'crop_add_min_scales':int(search.get('crop_add_min_scales',[2])[0] if isinstance(search.get('crop_add_min_scales',[2]),list) else search.get('crop_add_min_scales',2)),
            })
    return combos

# =============================================================================
# v14 slot-zone temporal memory override
# Track IDs are advisory. Slot-zone motion history survives tracker fragmentation.
# FULL evidence alone can create a NEW occupied state. AUX crop is used only as
# insurance against false EMPTY transitions for an already-occupied slot.
# =============================================================================

def _v14_motion_span(records):
    pts=[(float(x.get('center_x',np.nan)),float(x.get('center_y',np.nan))) for x in records
         if bool(x.get('full_hit',False)) and np.isfinite(float(x.get('center_x',np.nan))) and np.isfinite(float(x.get('center_y',np.nan)))]
    if len(pts)<2:
        return 0.0
    mx=float(np.median([x for x,_ in pts])); my=float(np.median([y for _,y in pts]))
    return float(max(math.hypot(x-mx,y-my) for x,y in pts))


def run_state_engine(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """v14 causal state engine with Track-ID-independent slot-zone memory."""
    window_sec=float(params.get('temporal_window_sec',10.0))
    recent_sec=float(params.get('recent_window_sec',3.0))
    zone_recent_sec=float(params.get('zone_recent_sec',4.0))
    entry_ratio=float(params.get('entry_hit_ratio',0.50))
    exit_ratio=float(params.get('exit_hit_ratio',0.10))
    stable_track_ratio_req=float(params.get('stable_track_ratio',0.55))
    visual_thr=float(params.get('occupied_visual_diff_threshold',0.12))
    stationary_speed=float(params.get('stationary_speed_px_s',14.0))
    stationary_span=float(params.get('stationary_motion_span_px',32.0))
    zone_motion_step=float(params.get('zone_motion_step_px',14.0))
    zone_motion_max=float(params.get('zone_recent_motion_max_px',24.0))
    zone_settle_required=float(params.get('entry_zone_settle_sec',8.0))
    entry_mean_conf_min=float(params.get('entry_mean_conf_min',0.10))
    entry_no_appearance_conf=float(params.get('entry_no_appearance_conf',0.25))
    exit_no_full_sec=float(params.get('exit_no_full_sec',4.0))
    exit_force_no_full_sec=float(params.get('exit_force_no_full_sec',9.0))
    exit_aux_window_sec=float(params.get('exit_aux_window_sec',5.0))
    exit_aux_min_checks=int(params.get('exit_aux_min_checks',2))
    aux_recovery_conf=float(params.get('aux_recovery_conf',0.12))
    aux_recovery_min_scales=int(params.get('aux_recovery_min_scales',1))
    fusion=str(params.get('global_merge','ANY')).upper()
    global_thr=float(params.get('global_confidence_threshold',0.60))
    evidence_mode='FULL'  # deployment baseline; AUX is handled separately as insurance.

    evidence=evidence.sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    state={}; phase={}; owner_track={}; prev_local={}; prev_global={}
    slot_hist: Dict[str,deque]=defaultdict(deque)
    track_hist: Dict[Tuple[str,int],deque]=defaultdict(deque)
    last_center: Dict[str,Tuple[float,float,float]]={}
    last_zone_motion_t: Dict[str,float]=defaultdict(lambda:-1e9)
    last_full_hit_t: Dict[str,float]=defaultdict(lambda:-1e9)
    last_change_t: Dict[str,float]=defaultdict(lambda:-1e9)
    slot_meta=evidence.groupby('local_id',sort=False).first()
    for lid,r in slot_meta.iterrows():
        init=str(r.get('initial_state','UNKNOWN')).upper(); st='OCCUPIED' if init=='OCCUPIED' else 'EMPTY'
        lid=str(lid); state[lid]=st; phase[lid]=st; owner_track[lid]=None; prev_local[lid]=st

    local_rows=[]; global_rows=[]; transitions=[]
    for t,group in evidence.groupby('time_sec',sort=True):
        t=float(t)
        # First pass: accumulate FULL-centric slot history and raw AUX audit results.
        for _,r in group.iterrows():
            lid=str(r['local_id']); cctv=str(r.get('cctv',''))
            fd=int(r.get('full_detected',r.get('detected',0)))>0
            fc=float(r.get('full_det_conf',r.get('det_conf',0.0))) if fd else 0.0
            tid=int(r.get('full_track_id',r.get('track_id',-1))) if fd else -1
            cx=float(r.get('full_center_x',r.get('det_center_x',np.nan))) if fd else np.nan
            cy=float(r.get('full_center_y',r.get('det_center_y',np.nan))) if fd else np.nan
            speed=float(r.get('full_track_speed_px_s',r.get('track_speed_px_s',0.0))) if fd else 0.0
            motion=float(r.get('full_track_motion_span_px',r.get('track_motion_span_px',0.0))) if fd else 0.0
            aux_checked=int(r.get('aux_requested',0))>0
            aux_conf=float(r.get('aux_crop_det_conf',r.get('crop_det_conf',0.0)))
            aux_support=int(r.get('aux_support_scales',0))
            aux_hit=bool(aux_checked and int(r.get('aux_crop_detected',r.get('crop_detected',0)))>0 and
                         aux_conf>=aux_recovery_conf and aux_support>=aux_recovery_min_scales)
            if fd:
                last_full_hit_t[lid]=t
                prev=last_center.get(lid)
                if prev is not None and t-float(prev[2])<=max(3.0,recent_sec+1.0):
                    if math.hypot(cx-float(prev[0]),cy-float(prev[1]))>=zone_motion_step:
                        last_zone_motion_t[lid]=t
                if speed>stationary_speed or motion>stationary_span:
                    last_zone_motion_t[lid]=t
                if np.isfinite(cx) and np.isfinite(cy): last_center[lid]=(cx,cy,t)
            rec={'t':t,'full_hit':int(fd),'detected':int(fd),'det_conf':fc,'track_id':tid,
                 'speed':speed,'motion_span':motion,'center_x':cx,'center_y':cy,'cctv':cctv,
                 'visual_diff':float(r.get('visual_diff_initial',0.0)),
                 'aux_checked':int(aux_checked),'aux_hit':int(aux_hit),'aux_conf':aux_conf,'aux_support':aux_support}
            slot_hist[lid].append(rec)
            while slot_hist[lid] and t-float(slot_hist[lid][0]['t'])>window_sec+1e-6: slot_hist[lid].popleft()
            if fd and tid>=0:
                key=(cctv,tid); track_hist[key].append({'t':t,'lid':lid,'speed':speed,'motion_span':motion})
                while track_hist[key] and t-float(track_hist[key][0]['t'])>window_sec+1e-6: track_hist[key].popleft()
        for key in list(track_hist.keys()):
            if not track_hist[key] or t-float(track_hist[key][-1]['t'])>window_sec+1e-6: del track_hist[key]

        owned_by_track={}
        for lid,tid in owner_track.items():
            if tid is None or state.get(lid)!='OCCUPIED': continue
            try:cctv=str(slot_meta.loc[lid].get('cctv',''))
            except Exception:cctv=''
            owned_by_track[(cctv,int(tid))]=lid

        now=[]
        for _,r in group.iterrows():
            lid=str(r['local_id']); cctv=str(r['cctv']); hist=list(slot_hist[lid])
            hits=[x for x in hist if x['full_hit']]
            recent=[x for x in hist if t-float(x['t'])<=recent_sec+1e-6]
            recent_hits=[x for x in recent if x['full_hit']]
            zone_recent=[x for x in hist if t-float(x['t'])<=zone_recent_sec+1e-6]
            hit_ratio=len(hits)/max(1,len(hist)); recent_hit_ratio=len(recent_hits)/max(1,len(recent))
            mean_conf=float(np.mean([x['det_conf'] for x in hits])) if hits else 0.0
            recent_mean_conf=float(np.mean([x['det_conf'] for x in recent_hits])) if recent_hits else 0.0
            track_counts=Counter(int(x['track_id']) for x in hits if int(x['track_id'])>=0)
            dom_track=track_counts.most_common(1)[0][0] if track_counts else -1
            dom_ratio=(track_counts.get(dom_track,0)/max(1,len(hits))) if dom_track>=0 else 0.0
            track_slot_ratio=0.0; track_switches=0; track_speed=0.0; track_motion=0.0
            if dom_track>=0:
                th=list(track_hist.get((cctv,dom_track),[]))
                if th:
                    track_slot_ratio=sum(1 for x in th if x['lid']==lid)/max(1,len(th)); seq=[x['lid'] for x in th]
                    track_switches=sum(1 for a,b in zip(seq,seq[1:]) if a!=b)
                    tr_recent=[x for x in th if t-float(x['t'])<=recent_sec+1e-6]; src=tr_recent if tr_recent else th
                    track_speed=float(np.median([x['speed'] for x in src])) if src else 0.0
                    track_motion=float(max([x['motion_span'] for x in src],default=0.0))
            zone_motion_10s=_v14_motion_span(hist)
            zone_motion_recent=_v14_motion_span(zone_recent)
            zone_settle_sec=(t-float(last_zone_motion_t[lid])) if last_zone_motion_t[lid]>-1e8 else window_sec+zone_recent_sec
            no_full_for=(t-float(last_full_hit_t[lid])) if last_full_hit_t[lid]>-1e8 else window_sec+zone_recent_sec
            aux_recent=[x for x in hist if t-float(x['t'])<=exit_aux_window_sec+1e-6 and x.get('aux_checked',0)]
            aux_checks=len(aux_recent); aux_hits=sum(int(x.get('aux_hit',0)) for x in aux_recent)
            aux_protect=aux_hits>0
            init=str(r.get('initial_state','UNKNOWN')).upper(); vis=float(r.get('visual_diff_initial',0.0)); appearance_occ=_appearance_occupied(init,vis,visual_thr)
            zone_stable=(zone_motion_recent<=zone_motion_max and zone_settle_sec>=zone_settle_required and recent_hit_ratio>=0.50)
            stable_track=(dom_track>=0 and track_slot_ratio>=stable_track_ratio_req and dom_ratio>=0.35 and track_speed<=stationary_speed and track_motion<=stationary_span)
            maneuvering=(zone_motion_recent>zone_motion_max or zone_settle_sec<zone_settle_required or track_speed>stationary_speed or track_motion>stationary_span)
            ghost_guard_ok=bool(appearance_occ or recent_mean_conf>=entry_no_appearance_conf or mean_conf>=entry_no_appearance_conf)
            entry_gate=False; exit_gate=False; exit_reason=''

            if state[lid]=='EMPTY':
                # NEW occupancy is FULL-only. Track continuity helps, but a broken Track ID
                # cannot erase slot-zone motion history.
                evidence_stable=zone_stable and (stable_track or recent_hit_ratio>=0.66)
                entry_gate=bool(evidence_stable and hit_ratio>=entry_ratio and recent_mean_conf>=entry_mean_conf_min and ghost_guard_ok)
                conflict=owned_by_track.get((cctv,dom_track)) if dom_track>=0 else None
                if entry_gate and (not conflict or conflict==lid):
                    state[lid]='OCCUPIED'; phase[lid]='OCCUPIED'; owner_track[lid]=dom_track if dom_track>=0 else None; last_change_t[lid]=t
                    if dom_track>=0: owned_by_track[(cctv,dom_track)]=lid
                elif maneuvering or hit_ratio>0: phase[lid]='MANEUVERING'
                else: phase[lid]='EMPTY'
            else:
                if dom_track>=0 and recent_hit_ratio>0 and (owner_track[lid] is None or stable_track):
                    owner_track[lid]=dom_track; owned_by_track[(cctv,dom_track)]=lid
                # AUX cannot create occupancy, but it can protect an existing occupied slot
                # from a false EMPTY while FULL temporarily misses the parked vehicle.
                normal_clear=(no_full_for>=exit_no_full_sec and recent_hit_ratio<=max(exit_ratio,0.25) and not appearance_occ and not aux_protect)
                forced_clear=(no_full_for>=exit_force_no_full_sec and recent_hit_ratio<=0.01 and not aux_protect and aux_checks>=exit_aux_min_checks)
                exit_gate=bool(normal_clear or forced_clear)
                if exit_gate:
                    exit_reason='FORCED_NO_FULL_PLUS_AUX_MISS' if forced_clear and not normal_clear else 'FULL_MISS_APPEARANCE_EMPTY'
                    old=owner_track[lid]; state[lid]='EMPTY'; phase[lid]='EMPTY'; owner_track[lid]=None; last_change_t[lid]=t
                    if old is not None and owned_by_track.get((cctv,int(old)))==lid: owned_by_track.pop((cctv,int(old)),None)
                elif aux_protect and recent_hit_ratio<=0.25:
                    phase[lid]='OCCUPIED'
                elif maneuvering and recent_hit_ratio>0:
                    phase[lid]='LEAVING'
                else: phase[lid]='OCCUPIED'

            temporal_score=0.34*hit_ratio+0.16*recent_hit_ratio+0.10*track_slot_ratio+0.15*(1.0 if appearance_occ else 0.0)+0.15*min(1.0,mean_conf)+0.10*(1.0 if aux_protect else 0.0)
            occ_score=max(0.51,min(0.99,temporal_score)) if state[lid]=='OCCUPIED' else min(0.49,max(0.01,temporal_score))
            row={'time_sec':t,'timestamp':r['timestamp'],'cctv':cctv,'local_id':lid,'global_id':r['global_id'],
                 'evidence_mode':'FULL+AUX_RECOVERY','evidence_source':'FULL' if int(r.get('full_detected',r.get('detected',0))) else ('AUX_RECOVERY' if aux_protect else 'NONE'),
                 'detected':int(r.get('full_detected',r.get('detected',0))),'det_conf':float(r.get('full_det_conf',r.get('det_conf',0.0))),
                 'track_id':int(r.get('full_track_id',r.get('track_id',-1))),'dominant_track_id':int(dom_track),'state':state[lid],'phase':phase[lid],
                 'occupied_score':float(occ_score),'window_hit_ratio':float(hit_ratio),'recent_hit_ratio':float(recent_hit_ratio),
                 'mean_detection_confidence':float(mean_conf),'recent_mean_confidence':float(recent_mean_conf),
                 'dominant_track_ratio_in_slot':float(dom_ratio),'track_slot_ratio':float(track_slot_ratio),'track_switches_window':int(track_switches),
                 'track_speed_px_s':float(track_speed),'track_motion_span_px':float(track_motion),
                 'slot_zone_motion_10s_px':float(zone_motion_10s),'slot_zone_motion_recent_px':float(zone_motion_recent),
                 'slot_zone_settle_sec':float(zone_settle_sec),'time_since_last_full_detection_sec':float(no_full_for),
                 'aux_recovery_checks_recent':int(aux_checks),'aux_recovery_hits_recent':int(aux_hits),'aux_recovery_protect':int(aux_protect),
                 'appearance_occupied':int(bool(appearance_occ)),'visual_diff_initial':vis,'ghost_guard_ok':int(ghost_guard_ok),
                 'entry_gate_ok':int(entry_gate),'exit_gate_ok':int(exit_gate),'exit_reason':exit_reason,
                 'owner_track_id':int(owner_track[lid]) if owner_track[lid] is not None else -1,'warmup':int(t<window_sec-1e-6)}
            local_rows.append(row); now.append(row)
            if prev_local.get(lid)!=state[lid]:
                transitions.append({'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,'global_id':r['global_id'],
                    'from_state':prev_local.get(lid,''),'to_state':state[lid],'cctv':cctv,'phase':phase[lid],
                    'dominant_track_id':int(dom_track),'evidence_mode':'FULL+AUX_RECOVERY','exit_reason':exit_reason})
                prev_local[lid]=state[lid]

        by_global=defaultdict(list)
        for x in now: by_global[x['global_id']].append(x)
        for gid,items in by_global.items():
            occ=sum(1 for x in items if x['state']=='OCCUPIED'); scores=[float(x['occupied_score']) for x in items]
            if fusion=='CONF_MAX': gscore=max(scores) if scores else 0.0; global_occ=gscore>=global_thr
            elif fusion=='CONF_MEAN': gscore=float(np.mean(scores)) if scores else 0.0; global_occ=gscore>=global_thr
            elif fusion=='MAJORITY': gscore=float(np.mean(scores)) if scores else 0.0; global_occ=occ>=math.ceil(len(items)/2)
            else: gscore=max(scores) if scores else 0.0; global_occ=occ>0
            gstate='OCCUPIED' if global_occ else 'EMPTY'; phases=[str(x.get('phase','')) for x in items]
            gphase='MANEUVERING' if 'MANEUVERING' in phases else ('LEAVING' if 'LEAVING' in phases else gstate)
            grow={'time_sec':t,'timestamp':items[0]['timestamp'],'global_id':gid,'state':gstate,'phase':gphase,'global_score':gscore,
                  'evidence_mode':'FULL+AUX_RECOVERY','source_local_slots':';'.join(x['local_id'] for x in items),'occupied_votes':occ,'total_votes':len(items),
                  'warmup':int(t<window_sec-1e-6)}
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid]!=gstate:
                transitions.append({'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,'global_id':gid,
                    'from_state':prev_global[gid],'to_state':gstate,'cctv':'MULTI','phase':gphase,'dominant_track_id':-1,'evidence_mode':'FULL+AUX_RECOVERY','exit_reason':''})
            prev_global[gid]=gstate
    return pd.DataFrame(local_rows),pd.DataFrame(global_rows),pd.DataFrame(transitions)


def _state_param_combos(search: Dict) -> List[Dict]:
    """v14 compact FULL-only DEV grid. AUX never creates new occupancy."""
    combos=[]; seen=set()
    for entry_ratio,vthr,entry_conf in itertools.product(
        search.get('entry_hit_ratio',[0.35,0.50]),
        search.get('occupied_visual_diff_threshold',[0.08,0.12]),
        search.get('entry_mean_conf_min',[0.08,0.12]),
    ):
        key=(float(entry_ratio),float(vthr),float(entry_conf))
        if key in seen: continue
        seen.add(key)
        combos.append({
            'evidence_mode':'FULL','entry_hit_ratio':float(entry_ratio),'exit_hit_ratio':float(search.get('exit_hit_ratio',[0.10])[0]),
            'stable_track_ratio':float(search.get('stable_track_ratio',[0.55])[0]),'occupied_visual_diff_threshold':float(vthr),
            'global_merge':'ANY','global_confidence_threshold':0.0,'entry_mean_conf_min':float(entry_conf),
            'entry_no_appearance_conf':float(search.get('entry_no_appearance_conf',[0.25])[0]),
            'full_min_conf':0.0,'crop_min_conf':0.10,'crop_add_min_conf':0.25,'crop_add_high_conf':0.50,'crop_add_min_scales':2,
        })
    return combos

# =============================================================================
# v14.1 conservative slot-zone cooldown override
# This deliberately preserves the proven v13.1 state machine and adds only two
# safeguards: Track-ID-independent maneuver cooldown for NEW occupancy, and AUX
# protection against false EMPTY. It avoids aggressive force-clears.
# =============================================================================

def run_state_engine(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    window_sec=float(params.get('temporal_window_sec',10.0)); recent_sec=float(params.get('recent_window_sec',3.0))
    entry_ratio=float(params.get('entry_hit_ratio',0.50)); exit_ratio=float(params.get('exit_hit_ratio',0.10))
    stable_track_ratio_req=float(params.get('stable_track_ratio',0.65)); visual_thr=float(params.get('occupied_visual_diff_threshold',0.12))
    stationary_speed=float(params.get('stationary_speed_px_s',14.0)); stationary_span=float(params.get('stationary_motion_span_px',32.0))
    crop_stationary_span=float(params.get('crop_stationary_motion_span_px',28.0)); max_switches=int(params.get('max_track_switches_for_entry',1))
    entry_mean_conf_min=float(params.get('entry_mean_conf_min',0.10)); entry_no_appearance_conf=float(params.get('entry_no_appearance_conf',0.25))
    zone_step=float(params.get('zone_motion_step_px',14.0)); zone_settle_req=float(params.get('entry_zone_settle_sec',6.0))
    aux_recovery_conf=float(params.get('aux_recovery_conf',0.12)); aux_recovery_min_scales=int(params.get('aux_recovery_min_scales',1))
    fusion=str(params.get('global_merge','ANY')).upper(); global_thr=float(params.get('global_confidence_threshold',0.60))
    evidence_mode='FULL'

    evidence=evidence.sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    state={};phase={};owner_track={};prev_local={};prev_global={}
    slot_hist: Dict[str,deque]=defaultdict(deque);track_hist: Dict[Tuple[str,int],deque]=defaultdict(deque)
    last_center={};last_zone_motion_t=defaultdict(lambda:-1e9)
    slot_meta=evidence.groupby('local_id',sort=False).first()
    for lid,r in slot_meta.iterrows():
        init=str(r.get('initial_state','UNKNOWN')).upper();st='OCCUPIED' if init=='OCCUPIED' else 'EMPTY'
        lid=str(lid);state[lid]=st;phase[lid]=st;owner_track[lid]=None;prev_local[lid]=st

    local_rows=[];global_rows=[];transitions=[]
    for t,group in evidence.groupby('time_sec',sort=True):
        t=float(t)
        for _,r in group.iterrows():
            lid=str(r['local_id']);obs=_effective_observation(r,{**params,'evidence_mode':'FULL'})
            fd=bool(obs['detected']);cx=float(obs['center_x']) if fd else np.nan;cy=float(obs['center_y']) if fd else np.nan
            if fd and np.isfinite(cx) and np.isfinite(cy):
                prev=last_center.get(lid)
                if prev is not None and t-float(prev[2])<=max(3.0,recent_sec+1.0) and math.hypot(cx-float(prev[0]),cy-float(prev[1]))>=zone_step:
                    last_zone_motion_t[lid]=t
                if float(obs['speed'])>stationary_speed or float(obs['motion_span'])>stationary_span:
                    last_zone_motion_t[lid]=t
                last_center[lid]=(cx,cy,t)
            aux_checked=int(r.get('aux_requested',0))>0
            aux_conf=float(r.get('aux_crop_det_conf',r.get('crop_det_conf',0.0)))
            aux_support=int(r.get('aux_support_scales',0))
            aux_hit=bool(aux_checked and int(r.get('aux_crop_detected',r.get('crop_detected',0)))>0 and aux_conf>=aux_recovery_conf and aux_support>=aux_recovery_min_scales)
            rec={'t':t,**obs,'visual_diff':float(r.get('visual_diff_initial',0.0)),'cctv':str(r.get('cctv','')),
                 'aux_checked':int(aux_checked),'aux_hit':int(aux_hit),'aux_conf':aux_conf}
            slot_hist[lid].append(rec)
            while slot_hist[lid] and t-float(slot_hist[lid][0]['t'])>window_sec+1e-6:slot_hist[lid].popleft()
            if rec['detected'] and rec['track_id']>=0:
                key=(rec['cctv'],rec['track_id']);track_hist[key].append({'t':t,'lid':lid,'speed':rec['speed'],'motion_span':rec['motion_span']})
                while track_hist[key] and t-float(track_hist[key][0]['t'])>window_sec+1e-6:track_hist[key].popleft()
        for key in list(track_hist.keys()):
            if not track_hist[key] or t-float(track_hist[key][-1]['t'])>window_sec+1e-6:del track_hist[key]

        owned_by_track={}
        for lid,tid in owner_track.items():
            if tid is None or state.get(lid)!='OCCUPIED':continue
            try:cctv=str(slot_meta.loc[lid].get('cctv',''))
            except Exception:cctv=''
            owned_by_track[(cctv,int(tid))]=lid
        now=[]
        for _,r in group.iterrows():
            lid=str(r['local_id']);cctv=str(r['cctv']);hist=list(slot_hist[lid]);hits=[x for x in hist if x['detected']]
            recent=[x for x in hist if t-float(x['t'])<=recent_sec+1e-6];recent_hits=[x for x in recent if x['detected']]
            hit_ratio=len(hits)/max(1,len(hist));recent_hit_ratio=len(recent_hits)/max(1,len(recent))
            mean_conf=float(np.mean([x['det_conf'] for x in hits])) if hits else 0.0
            recent_mean_conf=float(np.mean([x['det_conf'] for x in recent_hits])) if recent_hits else 0.0
            track_counts=Counter(int(x['track_id']) for x in hits if int(x['track_id'])>=0)
            dom_track=track_counts.most_common(1)[0][0] if track_counts else -1;dom_ratio=(track_counts.get(dom_track,0)/max(1,len(hits))) if dom_track>=0 else 0.0
            track_slot_ratio=0.0;track_switches=0;track_speed=0.0;track_motion=0.0
            if dom_track>=0:
                th=list(track_hist.get((cctv,dom_track),[]))
                if th:
                    track_slot_ratio=sum(1 for x in th if x['lid']==lid)/max(1,len(th));seq=[x['lid'] for x in th]
                    track_switches=sum(1 for a,b in zip(seq,seq[1:]) if a!=b)
                    tr_recent=[x for x in th if t-float(x['t'])<=recent_sec+1e-6];src=tr_recent if tr_recent else th
                    track_speed=float(np.median([x['speed'] for x in src])) if src else 0.0;track_motion=float(max([x['motion_span'] for x in src],default=0.0))
            centers=[(float(x['center_x']),float(x['center_y'])) for x in hits if np.isfinite(x['center_x']) and np.isfinite(x['center_y'])]
            local_motion=0.0
            if len(centers)>=2:
                mx=float(np.median([x for x,_ in centers]));my=float(np.median([y for _,y in centers]));local_motion=max(math.hypot(x-mx,y-my) for x,y in centers)
            init=str(r.get('initial_state','UNKNOWN')).upper();vis=float(r.get('visual_diff_initial',0.0));appearance_occ=_appearance_occupied(init,vis,visual_thr)
            stable_track=(dom_track>=0 and track_slot_ratio>=stable_track_ratio_req and dom_ratio>=0.50 and track_switches<=max_switches and track_speed<=stationary_speed and track_motion<=stationary_span and recent_hit_ratio>=0.34)
            stable_slot=(hit_ratio>=max(entry_ratio,0.55) and recent_hit_ratio>=0.50 and local_motion<=crop_stationary_span)
            zone_settle_sec=(t-float(last_zone_motion_t[lid])) if last_zone_motion_t[lid]>-1e8 else window_sec+recent_sec
            zone_cooldown_ok=zone_settle_sec>=zone_settle_req
            maneuvering=(dom_track>=0 and (track_switches>max_switches or track_speed>stationary_speed or track_motion>stationary_span)) or (local_motion>crop_stationary_span and hit_ratio>0) or (not zone_cooldown_ok and recent_hit_ratio>0)
            ghost_guard_ok=bool(appearance_occ or mean_conf>=entry_no_appearance_conf)
            aux_recent=[x for x in hist if t-float(x['t'])<=recent_sec+1e-6 and x.get('aux_checked',0)]
            aux_protect=any(bool(x.get('aux_hit',0)) for x in aux_recent)
            entry_gate=False;exit_gate=False
            if state[lid]=='EMPTY':
                strong_stationary=(hit_ratio>=max(entry_ratio,0.65) and recent_hit_ratio>=0.66 and local_motion<=crop_stationary_span)
                evidence_stable=stable_track if dom_track>=0 else (stable_slot or strong_stationary)
                entry_gate=bool(evidence_stable and zone_cooldown_ok and hit_ratio>=entry_ratio and mean_conf>=entry_mean_conf_min and ghost_guard_ok)
                conflict=owned_by_track.get((cctv,dom_track)) if dom_track>=0 else None
                if entry_gate and (not conflict or conflict==lid):
                    state[lid]='OCCUPIED';phase[lid]='OCCUPIED';owner_track[lid]=dom_track if dom_track>=0 else None
                    if dom_track>=0:owned_by_track[(cctv,dom_track)]=lid
                elif maneuvering or hit_ratio>0:phase[lid]='MANEUVERING'
                else:phase[lid]='EMPTY'
            else:
                if dom_track>=0 and recent_hit_ratio>0 and (owner_track[lid] is None or stable_track):
                    owner_track[lid]=dom_track;owned_by_track[(cctv,dom_track)]=lid
                clear_empty=(hit_ratio<=exit_ratio and recent_hit_ratio<=exit_ratio and not appearance_occ and not aux_protect)
                exit_gate=bool(clear_empty)
                if clear_empty:
                    old=owner_track[lid];state[lid]='EMPTY';phase[lid]='EMPTY';owner_track[lid]=None
                    if old is not None and owned_by_track.get((cctv,int(old)))==lid:owned_by_track.pop((cctv,int(old)),None)
                elif aux_protect and recent_hit_ratio<=exit_ratio:phase[lid]='OCCUPIED'
                elif maneuvering and recent_hit_ratio>0:phase[lid]='LEAVING' if owner_track[lid]==dom_track else 'OCCUPIED'
                else:phase[lid]='OCCUPIED'
            temporal_score=0.40*hit_ratio+0.18*recent_hit_ratio+0.12*track_slot_ratio+0.15*(1.0 if appearance_occ else 0.0)+0.15*min(1.0,mean_conf)
            occ_score=max(0.51,min(0.99,temporal_score)) if state[lid]=='OCCUPIED' else min(0.49,max(0.01,temporal_score))
            row={'time_sec':t,'timestamp':r['timestamp'],'cctv':cctv,'local_id':lid,'global_id':r['global_id'],'evidence_mode':'FULL+AUX_RECOVERY',
                 'evidence_source':'FULL' if int(r.get('full_detected',r.get('detected',0))) else ('AUX_RECOVERY' if aux_protect else 'NONE'),
                 'detected':int(r.get('full_detected',r.get('detected',0))),'det_conf':float(r.get('full_det_conf',r.get('det_conf',0.0))),
                 'track_id':int(r.get('full_track_id',r.get('track_id',-1))),'dominant_track_id':int(dom_track),'state':state[lid],'phase':phase[lid],
                 'occupied_score':float(occ_score),'window_hit_ratio':float(hit_ratio),'recent_hit_ratio':float(recent_hit_ratio),'mean_detection_confidence':float(mean_conf),
                 'recent_mean_confidence':float(recent_mean_conf),'dominant_track_ratio_in_slot':float(dom_ratio),'track_slot_ratio':float(track_slot_ratio),
                 'track_switches_window':int(track_switches),'track_speed_px_s':float(track_speed),'track_motion_span_px':float(track_motion),
                 'slot_detection_motion_span_px':float(local_motion),'slot_zone_settle_sec':float(zone_settle_sec),'zone_cooldown_ok':int(zone_cooldown_ok),
                 'appearance_occupied':int(bool(appearance_occ)),'visual_diff_initial':vis,'aux_recovery_protect':int(aux_protect),
                 'ghost_guard_ok':int(ghost_guard_ok),'entry_gate_ok':int(entry_gate),'exit_gate_ok':int(exit_gate),
                 'owner_track_id':int(owner_track[lid]) if owner_track[lid] is not None else -1,'warmup':int(t<window_sec-1e-6)}
            local_rows.append(row);now.append(row)
            if prev_local.get(lid)!=state[lid]:
                transitions.append({'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,'global_id':r['global_id'],'from_state':prev_local.get(lid,''),'to_state':state[lid],'cctv':cctv,'phase':phase[lid],'dominant_track_id':int(dom_track),'evidence_mode':'FULL+AUX_RECOVERY'})
                prev_local[lid]=state[lid]
        by_global=defaultdict(list)
        for x in now:by_global[x['global_id']].append(x)
        for gid,items in by_global.items():
            occ=sum(1 for x in items if x['state']=='OCCUPIED');scores=[float(x['occupied_score']) for x in items]
            if fusion=='CONF_MAX':gscore=max(scores) if scores else 0.0;global_occ=gscore>=global_thr
            elif fusion=='CONF_MEAN':gscore=float(np.mean(scores)) if scores else 0.0;global_occ=gscore>=global_thr
            elif fusion=='MAJORITY':gscore=float(np.mean(scores)) if scores else 0.0;global_occ=occ>=math.ceil(len(items)/2)
            else:gscore=max(scores) if scores else 0.0;global_occ=occ>0
            gstate='OCCUPIED' if global_occ else 'EMPTY';phases=[str(x.get('phase','')) for x in items]
            gphase='MANEUVERING' if 'MANEUVERING' in phases else ('LEAVING' if 'LEAVING' in phases else gstate)
            grow={'time_sec':t,'timestamp':items[0]['timestamp'],'global_id':gid,'state':gstate,'phase':gphase,'global_score':gscore,'evidence_mode':'FULL+AUX_RECOVERY','source_local_slots':';'.join(x['local_id'] for x in items),'occupied_votes':occ,'total_votes':len(items),'warmup':int(t<window_sec-1e-6)}
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid]!=gstate:
                transitions.append({'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,'global_id':gid,'from_state':prev_global[gid],'to_state':gstate,'cctv':'MULTI','phase':gphase,'dominant_track_id':-1,'evidence_mode':'FULL+AUX_RECOVERY'})
            prev_global[gid]=gstate
    return pd.DataFrame(local_rows),pd.DataFrame(global_rows),pd.DataFrame(transitions)


# v14 safe baseline: exact v13.1 temporal engine retained for regression protection.
def _run_state_engine_v13_safe(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    window_sec=float(params.get('temporal_window_sec',10.0)); recent_sec=float(params.get('recent_window_sec',3.0))
    entry_ratio=float(params.get('entry_hit_ratio',0.50)); exit_ratio=float(params.get('exit_hit_ratio',0.10))
    stable_track_ratio_req=float(params.get('stable_track_ratio',0.65)); visual_thr=float(params.get('occupied_visual_diff_threshold',0.12))
    stationary_speed=float(params.get('stationary_speed_px_s',14.0)); stationary_span=float(params.get('stationary_motion_span_px',32.0))
    crop_stationary_span=float(params.get('crop_stationary_motion_span_px',28.0)); max_switches=int(params.get('max_track_switches_for_entry',1))
    entry_mean_conf_min=float(params.get('entry_mean_conf_min',0.10)); entry_no_appearance_conf=float(params.get('entry_no_appearance_conf',0.25))
    fusion=str(params.get('global_merge','ANY')).upper(); global_thr=float(params.get('global_confidence_threshold',0.60))
    evidence_mode=str(params.get('evidence_mode','FULL')).upper()

    evidence=evidence.sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    state={}; phase={}; owner_track={}; prev_local={}; prev_global={}
    slot_hist: Dict[str,deque]=defaultdict(deque); track_hist: Dict[Tuple[str,int],deque]=defaultdict(deque)
    slot_meta=evidence.groupby('local_id',sort=False).first()
    for lid,r in slot_meta.iterrows():
        init=str(r.get('initial_state','UNKNOWN')).upper()
        st=init if params.get('preserve_unknown',False) and init in ('OCCUPIED','EMPTY','UNKNOWN') else ('OCCUPIED' if init=='OCCUPIED' else 'EMPTY')
        lid=str(lid); state[lid]=st; phase[lid]=st; owner_track[lid]=None; prev_local[lid]=st

    recovery_cfg=params.get('startup_recovery',{})
    recovery_start=float(evidence.time_sec.min()) if not evidence.empty else 0.0
    recovery_hist=defaultdict(list)
    local_rows=[]; global_rows=[]; transitions=[]
    for t,group in evidence.groupby('time_sec',sort=True):
        t=float(t)
        for _,r in group.iterrows():
            lid=str(r['local_id']); obs=_effective_observation(r,params)
            rec={'t':t,**obs,'visual_diff':float(r.get('visual_diff_initial',0.0)),'cctv':str(r.get('cctv',''))}
            slot_hist[lid].append(rec)
            while slot_hist[lid] and t-float(slot_hist[lid][0]['t'])>window_sec+1e-6: slot_hist[lid].popleft()
            if rec['detected'] and rec['track_id']>=0:
                key=(rec['cctv'],rec['track_id']); track_hist[key].append({'t':t,'lid':lid,'speed':rec['speed'],'motion_span':rec['motion_span']})
                while track_hist[key] and t-float(track_hist[key][0]['t'])>window_sec+1e-6: track_hist[key].popleft()
        for key in list(track_hist.keys()):
            if not track_hist[key] or t-float(track_hist[key][-1]['t'])>window_sec+1e-6: del track_hist[key]

        recovery_ready=set()
        if recovery_cfg:
            from restart_evaluation import recovery_gate
            for _,r in group.iterrows():
                lid=str(r['local_id'])
                recovery_hist[lid].append(r.to_dict())
                recovery_hist[lid]=[x for x in recovery_hist[lid] if t-float(x['time_sec'])<=float(recovery_cfg.get('window_sec',10))]
            recovery_ready={lid for lid,h in recovery_hist.items() if recovery_gate(h,t-recovery_start,recovery_cfg)}
        owned_by_track={}
        for lid,tid in owner_track.items():
            if tid is None or state.get(lid)!='OCCUPIED': continue
            try:cctv=str(slot_meta.loc[lid].get('cctv',''))
            except Exception:cctv=''
            owned_by_track[(cctv,int(tid))]=lid

        now=[]
        for _,r in group.iterrows():
            lid=str(r['local_id']); cctv=str(r['cctv']); hist=list(slot_hist[lid]); hits=[x for x in hist if x['detected']]
            recent=[x for x in hist if t-float(x['t'])<=recent_sec+1e-6]; recent_hits=[x for x in recent if x['detected']]
            hit_ratio=len(hits)/max(1,len(hist)); recent_hit_ratio=len(recent_hits)/max(1,len(recent))
            mean_conf=float(np.mean([x['det_conf'] for x in hits])) if hits else 0.0
            track_counts=Counter(int(x['track_id']) for x in hits if int(x['track_id'])>=0)
            dom_track=track_counts.most_common(1)[0][0] if track_counts else -1
            dom_ratio=(track_counts.get(dom_track,0)/max(1,len(hits))) if dom_track>=0 else 0.0
            track_slot_ratio=0.0; track_switches=0; track_speed=0.0; track_motion=0.0
            if dom_track>=0:
                th=list(track_hist.get((cctv,dom_track),[]))
                if th:
                    track_slot_ratio=sum(1 for x in th if x['lid']==lid)/max(1,len(th)); seq=[x['lid'] for x in th]
                    track_switches=sum(1 for a,b in zip(seq,seq[1:]) if a!=b)
                    tr_recent=[x for x in th if t-float(x['t'])<=recent_sec+1e-6]; src=tr_recent if tr_recent else th
                    track_speed=float(np.median([x['speed'] for x in src])) if src else 0.0
                    track_motion=float(max([x['motion_span'] for x in src],default=0.0))
            centers=[(float(x['center_x']),float(x['center_y'])) for x in hits if np.isfinite(x['center_x']) and np.isfinite(x['center_y'])]
            local_motion=0.0
            if len(centers)>=2:
                mx=float(np.median([x for x,_ in centers])); my=float(np.median([y for _,y in centers]))
                local_motion=max(math.hypot(x-mx,y-my) for x,y in centers)
            init=str(r.get('initial_state','UNKNOWN')).upper(); vis=float(r.get('visual_diff_initial',0.0)); appearance_occ=False if params.get('unlabeled_restart',False) else _appearance_occupied(init,vis,visual_thr)
            stable_track=(dom_track>=0 and track_slot_ratio>=stable_track_ratio_req and dom_ratio>=0.50 and track_switches<=max_switches and track_speed<=stationary_speed and track_motion<=stationary_span and recent_hit_ratio>=0.34)
            stable_slot=(hit_ratio>=max(entry_ratio,0.55) and recent_hit_ratio>=0.50 and local_motion<=crop_stationary_span)
            maneuvering=(dom_track>=0 and (track_switches>max_switches or track_speed>stationary_speed or track_motion>stationary_span)) or (local_motion>crop_stationary_span and hit_ratio>0)
            ghost_guard_ok=bool(appearance_occ or mean_conf>=entry_no_appearance_conf)
            entry_gate=False; exit_gate=False

            if state[lid] in ('EMPTY','UNKNOWN'):
                strong_stationary=(hit_ratio>=max(entry_ratio,0.65) and recent_hit_ratio>=0.66 and local_motion<=crop_stationary_span)
                evidence_stable=stable_track if dom_track>=0 else (stable_slot or strong_stationary)
                entry_gate=bool(evidence_stable and hit_ratio>=entry_ratio and mean_conf>=entry_mean_conf_min and ghost_guard_ok)
                if recovery_cfg and t-recovery_start<=float(recovery_cfg.get('duration_sec',30)):
                    elapsed=t-recovery_start
                    # v16.5.6: recovery adds corroborated entry; normal SAFE entry
                    # remains available while startup observations accumulate.
                    # The explicit legacy flag retains the archived v16.5.5 comparator.
                    if not recovery_cfg.get('preserve_safe_entry',True):
                        entry_gate=bool(entry_gate and not maneuvering and elapsed>=float(recovery_cfg.get('min_observation_sec',10)))
                    if elapsed>=float(recovery_cfg.get('min_observation_sec',10)) and lid in recovery_ready and (not maneuvering or not recovery_cfg.get('preserve_safe_entry',True)):
                        peers=[str(x) for x in recovery_ready if str(slot_meta.loc[x,'global_id'])==str(r['global_id']) and str(slot_meta.loc[x,'cctv'])!=cctv]
                        confidences=[float(x.get('full_det_conf',0)) for x in recovery_hist[lid] if int(x.get('full_detected',0))]
                        strong_single=bool(confidences and np.mean(confidences)>=float(recovery_cfg.get('single_camera_conf',0.25)))
                        entry_gate=bool(entry_gate or peers or strong_single)
                conflict=owned_by_track.get((cctv,dom_track)) if dom_track>=0 else None
                if entry_gate and (not conflict or conflict==lid):
                    state[lid]='OCCUPIED'; phase[lid]='OCCUPIED'; owner_track[lid]=dom_track if dom_track>=0 else None
                    if dom_track>=0: owned_by_track[(cctv,dom_track)]=lid
                elif maneuvering or hit_ratio>0: phase[lid]='MANEUVERING'
                else: phase[lid]=state[lid]
            else:
                if dom_track>=0 and recent_hit_ratio>0 and (owner_track[lid] is None or stable_track):
                    owner_track[lid]=dom_track; owned_by_track[(cctv,dom_track)]=lid
                clear_empty=(hit_ratio<=exit_ratio and recent_hit_ratio<=exit_ratio and not appearance_occ)
                if params.get('manual_initialization',False) and init=='OCCUPIED' and t-recovery_start<window_sec:
                    clear_empty=False
                exit_gate=bool(clear_empty)
                if clear_empty:
                    old=owner_track[lid]; state[lid]='EMPTY'; phase[lid]='EMPTY'; owner_track[lid]=None
                    if old is not None and owned_by_track.get((cctv,int(old)))==lid: owned_by_track.pop((cctv,int(old)),None)
                elif maneuvering and recent_hit_ratio>0: phase[lid]='LEAVING' if owner_track[lid]==dom_track else 'OCCUPIED'
                else: phase[lid]='OCCUPIED'

            if params.get('manual_initialization',False) and t==recovery_start:
                state[lid]=init; phase[lid]=init; entry_gate=False; exit_gate=False; owner_track[lid]=None

            temporal_score=0.40*hit_ratio+0.18*recent_hit_ratio+0.12*track_slot_ratio+0.15*(1.0 if appearance_occ else 0.0)+0.15*min(1.0,mean_conf)
            occ_score=max(0.51,min(0.99,temporal_score)) if state[lid]=='OCCUPIED' else min(0.49,max(0.01,temporal_score))
            obs_now=_effective_observation(r,params)
            row={'time_sec':t,'timestamp':r['timestamp'],'cctv':cctv,'local_id':lid,'global_id':r['global_id'],
                 'evidence_mode':evidence_mode,'evidence_source':obs_now['source'],'detected':obs_now['detected'],'det_conf':obs_now['det_conf'],
                 'track_id':obs_now['track_id'],'dominant_track_id':int(dom_track),'state':state[lid],'phase':phase[lid],
                 'occupied_score':float(occ_score),'window_hit_ratio':float(hit_ratio),'recent_hit_ratio':float(recent_hit_ratio),
                 'mean_detection_confidence':float(mean_conf),'dominant_track_ratio_in_slot':float(dom_ratio),'track_slot_ratio':float(track_slot_ratio),
                 'track_switches_window':int(track_switches),'track_speed_px_s':float(track_speed),'track_motion_span_px':float(track_motion),
                 'slot_detection_motion_span_px':float(local_motion),'appearance_occupied':int(bool(appearance_occ)),'visual_diff_initial':vis,
                 'ghost_guard_ok':int(ghost_guard_ok),'entry_gate_ok':int(entry_gate),'exit_gate_ok':int(exit_gate),
                 'startup_phase':('RECOVERY' if recovery_cfg and t-recovery_start<=float(recovery_cfg.get('duration_sec',30)) else 'NORMAL'),'owner_track_id':int(owner_track[lid]) if owner_track[lid] is not None else -1,'warmup':int((t-recovery_start if params.get('unlabeled_restart',False) else t)<window_sec-1e-6)}
            local_rows.append(row); now.append(row)
            if prev_local.get(lid)!=state[lid]:
                transitions.append({'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,'global_id':r['global_id'],
                    'from_state':prev_local.get(lid,''),'to_state':state[lid],'cctv':cctv,'phase':phase[lid],
                    'dominant_track_id':int(dom_track),'evidence_mode':evidence_mode})
                prev_local[lid]=state[lid]

        by_global=defaultdict(list)
        for x in now: by_global[x['global_id']].append(x)
        for gid,items in by_global.items():
            occ=sum(1 for x in items if x['state']=='OCCUPIED'); scores=[float(x['occupied_score']) for x in items]
            if fusion=='CONF_MAX': gscore=max(scores) if scores else 0.0; global_occ=gscore>=global_thr
            elif fusion=='CONF_MEAN': gscore=float(np.mean(scores)) if scores else 0.0; global_occ=gscore>=global_thr
            elif fusion=='MAJORITY': gscore=float(np.mean(scores)) if scores else 0.0; global_occ=occ>=math.ceil(len(items)/2)
            else: gscore=max(scores) if scores else 0.0; global_occ=occ>0
            gstate='OCCUPIED' if global_occ else ('UNKNOWN' if params.get('preserve_unknown',False) and any(x['state']=='UNKNOWN' for x in items) else 'EMPTY'); phases=[str(x.get('phase','')) for x in items]
            gphase='MANEUVERING' if 'MANEUVERING' in phases else ('LEAVING' if 'LEAVING' in phases else gstate)
            grow={'time_sec':t,'timestamp':items[0]['timestamp'],'global_id':gid,'state':gstate,'phase':gphase,'global_score':gscore,
                  'evidence_mode':evidence_mode,'source_local_slots':';'.join(x['local_id'] for x in items),'occupied_votes':occ,'total_votes':len(items),
                  'warmup':int((t-recovery_start if params.get('unlabeled_restart',False) else t)<window_sec-1e-6)}
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid]!=gstate:
                transitions.append({'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,'global_id':gid,
                    'from_state':prev_global[gid],'to_state':gstate,'cctv':'MULTI','phase':gphase,'dominant_track_id':-1,'evidence_mode':evidence_mode})
            prev_global[gid]=gstate
    return pd.DataFrame(local_rows),pd.DataFrame(global_rows),pd.DataFrame(transitions)


def search_state_parameters(evidence: pd.DataFrame, gt: pd.DataFrame, settings: Dict, output_dir: str,
                            progress=None, slot_gt_events_path: Optional[str]=None) -> Dict:
    """v14 baseline-first temporal comparison.

    The proven v13.1 engine is preserved as SAFE_BASELINE. The new Track-ID-independent
    slot-zone memory is evaluated in parallel as ZONE_MEMORY. Selection uses DEV only.
    When DEV is clearly under-count-confounded (as in the current video with advisory
    candidate slots), v14 refuses to let that biased DEV segment replace the safe baseline.
    """
    out_dir=ensure_dir(output_dir)
    temporal=settings.get('temporal',{})
    search=settings.get('state_search',{})
    base={
        'evidence_mode':'FULL',
        'entry_hit_ratio':float(search.get('entry_hit_ratio',[0.35])[0]),
        'exit_hit_ratio':float(search.get('exit_hit_ratio',[0.10])[0]),
        'stable_track_ratio':float(search.get('stable_track_ratio',[0.55])[0]),
        'occupied_visual_diff_threshold':float(search.get('occupied_visual_diff_threshold',[0.08])[0]),
        'global_merge':'ANY','global_confidence_threshold':0.0,
        'entry_mean_conf_min':float(search.get('entry_mean_conf_min',[0.08])[0]),
        'entry_no_appearance_conf':float(search.get('entry_no_appearance_conf',[0.25])[0]),
        'full_min_conf':0.0,'crop_min_conf':0.10,'crop_add_min_conf':0.25,'crop_add_high_conf':0.50,'crop_add_min_scales':2,
        'temporal_window_sec':float(temporal.get('window_sec',10.0)),
        'recent_window_sec':float(temporal.get('recent_window_sec',3.0)),
        'stationary_speed_px_s':float(temporal.get('stationary_speed_px_s',14.0)),
        'stationary_motion_span_px':float(temporal.get('stationary_motion_span_px',32.0)),
        'crop_stationary_motion_span_px':float(temporal.get('crop_stationary_motion_span_px',28.0)),
        'max_track_switches_for_entry':int(temporal.get('max_track_switches_for_entry',1)),
    }
    zone={**base,
        'zone_motion_step_px':float(temporal.get('zone_motion_step_px',14.0)),
        'entry_zone_settle_sec':float(temporal.get('entry_zone_settle_sec',6.0)),
        'aux_recovery_conf':float(temporal.get('aux_recovery_conf',0.25)),
        'aux_recovery_min_scales':int(temporal.get('aux_recovery_min_scales',2)),
    }
    dev_end=float(settings.get('dev_end_sec',900)); warmup=float(settings.get('evaluation_warmup_sec',10.0))
    slot_events=load_slot_gt_events(slot_gt_events_path) if slot_gt_events_path and Path(slot_gt_events_path).exists() else None
    gt_times=gt['time_sec'].tolist()

    variants=[]
    for idx,(name,runner,params) in enumerate([
        ('SAFE_BASELINE',_run_state_engine_v13_safe,base),
        ('ZONE_MEMORY',run_state_engine,zone),
    ]):
        local_df,global_df,transitions=runner(evidence,params)
        ev=evaluate_state_output(global_df,gt,dev_end,warmup)
        slot_eval=evaluate_slot_level(global_df,slot_events,gt_times,dev_end,warmup) if slot_events is not None else None
        variants.append({'name':name,'params':params,'local':local_df,'global':global_df,'transitions':transitions,'ev':ev,'slot_eval':slot_eval})
        if progress: progress((idx+1)/2.0,f'v14 temporal comparison {idx+1}/2 | {name}')

    b,z=variants
    bd=b['ev']['DEV']; zd=z['ev']['DEV']
    dev_confounded=bool(bd['false_empty_bias_rate']>=0.35 and bd['exact_rate']<0.60)
    choose_zone=(not dev_confounded and
                 zd['exact_rate']>=bd['exact_rate']+0.03 and
                 zd['mae']<=bd['mae']-0.02 and
                 zd['over_rate']<=bd['over_rate']+0.02)
    chosen=z if choose_zone else b
    reason=(
        'ZONE_MEMORY selected: it cleared the baseline-first DEV improvement guardrail.' if choose_zone else
        'SAFE_BASELINE retained because DEV is strongly under-count-confounded; candidate-slot bias makes temporal replacement unsafe.' if dev_confounded else
        'SAFE_BASELINE retained because ZONE_MEMORY did not clearly improve DEV without extra bias.'
    )

    rows=[]
    for v in variants:
        d=v['ev']['DEV']; te=v['ev']['TEST']
        row={'temporal_variant':v['name'],**v['params'],
             'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],'dev_under_rate':d['false_empty_bias_rate'],'dev_over_rate':d['over_rate'],'dev_max_abs_error':d['max_abs_error'],
             'test_N':te['N'],'test_exact_rate':te['exact_rate'],'test_MAE':te['mae'],'test_under_rate':te['false_empty_bias_rate'],'test_over_rate':te['over_rate'],'test_max_abs_error':te['max_abs_error']}
        if v['slot_eval'] is not None:
            row.update({'dev_slot_accuracy':v['slot_eval']['DEV']['slot_accuracy'],'dev_slot_false_empty_rate':v['slot_eval']['DEV']['false_empty_rate'],'dev_all_slots_exact_time_rate':v['slot_eval']['DEV']['all_slots_exact_time_rate']})
        rows.append(row)
    board=pd.DataFrame(rows)
    board.to_csv(out_dir/'temporal_variant_comparison.csv',index=False,encoding='utf-8-sig')
    board.to_csv(out_dir/'state_search_leaderboard.csv',index=False,encoding='utf-8-sig')
    board.rename(columns={'temporal_variant':'evidence_mode'}).to_csv(out_dir/'detection_mode_comparison.csv',index=False,encoding='utf-8-sig')

    # Save both variants for diagnosis; selected_* remains deployment-safe.
    for v in variants:
        tag='baseline' if v['name']=='SAFE_BASELINE' else 'zone_memory'
        v['ev']['timeseries'].to_csv(out_dir/f'{tag}_count_timeseries.csv',index=False,encoding='utf-8-sig')
        v['global'].to_csv(out_dir/f'{tag}_global_slot_timeseries.csv',index=False,encoding='utf-8-sig')
        v['local'].to_csv(out_dir/f'{tag}_slot_timeseries.csv',index=False,encoding='utf-8-sig')

    local_df=chosen['local']; global_df=chosen['global']; transitions=chosen['transitions']; ev=chosen['ev']; slot_eval=chosen['slot_eval']
    local_df.to_csv(out_dir/'selected_slot_timeseries.csv',index=False,encoding='utf-8-sig')
    global_df.to_csv(out_dir/'selected_global_slot_timeseries.csv',index=False,encoding='utf-8-sig')
    transitions.to_csv(out_dir/'selected_state_transitions.csv',index=False,encoding='utf-8-sig')
    ev['timeseries'].to_csv(out_dir/'selected_count_timeseries.csv',index=False,encoding='utf-8-sig')
    selected_params={**chosen['params'],'temporal_variant':chosen['name']}
    save_json(out_dir/'selected_state_params.json',selected_params)
    pd.DataFrame([{'split':sp,**ev[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'metrics_dev_test.csv',index=False,encoding='utf-8-sig')
    ev['timeseries'][(ev['timeseries']['split']=='TEST')&(ev['timeseries']['error']!=0)].to_csv(out_dir/'test_error_cases.csv',index=False,encoding='utf-8-sig')
    save_json(out_dir/'mode_selection_audit.json',{
        'selected_temporal_variant':chosen['name'],'reason':reason,'dev_confounded':dev_confounded,
        'selection_uses_test':False,'baseline_first':True,
    })
    if slot_eval is not None:
        slot_eval['rows'].to_csv(out_dir/'slot_level_comparison.csv',index=False,encoding='utf-8-sig')
        slot_eval['by_slot'].to_csv(out_dir/'slot_level_by_slot.csv',index=False,encoding='utf-8-sig')
        slot_eval['rows'][(slot_eval['rows']['split']=='TEST')&(slot_eval['rows']['correct']==0)].to_csv(out_dir/'slot_level_test_errors.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{'split':sp,**slot_eval[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'slot_level_metrics.csv',index=False,encoding='utf-8-sig')

    with (out_dir/'REPORT.txt').open('w',encoding='utf-8') as f:
        f.write('Parking Slot State Engine v14 - baseline-safe slot-zone experiment\n\n')
        f.write(f'Warm-up excluded: 0 <= t < {warmup:.1f}s\nEvery decision uses only current/past frames.\n')
        f.write('NEW occupancy is FULL-only. AUX crop is recovery insurance only.\n')
        f.write('Candidate slots remain advisory and are never auto-added.\n\n')
        f.write('SELECTION\n'+reason+'\n')
        f.write(f'dev_confounded={dev_confounded}\nselection_uses_TEST=False\n\n')
        for v in variants:
            d=v['ev']['DEV'];te=v['ev']['TEST']
            f.write(f"[{v['name']}] DEV exact={d['exact_rate']*100:.2f}% MAE={d['mae']:.4f} under={d['false_empty_bias_rate']*100:.2f}% | TEST exact={te['exact_rate']*100:.2f}% MAE={te['mae']:.4f} max={te['max_abs_error']}\n")
        f.write(f'\nSELECTED={chosen["name"]}\n')
        for sp in ['DEV','TEST']:
            m=ev[sp];f.write(f'[{sp}] N={m["N"]} Exact={m["exact_rate"]*100:.2f}% MAE={m["mae"]:.4f} Under={m["false_empty_bias_rate"]*100:.2f}% Over={m["over_rate"]*100:.2f}% MaxAbs={m["max_abs_error"]}\n')
    return {'best_params':selected_params,'evaluation':ev,'slot_evaluation':slot_eval,'leaderboard':board,'mode_comparison':board,'selection_audit':{'reason':reason,'dev_confounded':dev_confounded},'output_dir':str(out_dir)}

# =============================================================================
# v15 transition-guard engine
# Keep the proven v13 state engine as the source of proposed state changes.
# Ten-second slot-zone memory is consulted ONLY when a state would change.
# =============================================================================

def _v15_motion_span(records: Sequence[Dict]) -> float:
    pts=[]
    for x in records:
        try:
            cx=float(x.get('center_x',np.nan)); cy=float(x.get('center_y',np.nan))
        except Exception:
            continue
        if np.isfinite(cx) and np.isfinite(cy): pts.append((cx,cy))
    if len(pts)<2:return 0.0
    xs=np.asarray([p[0] for p in pts],dtype=np.float32);ys=np.asarray([p[1] for p in pts],dtype=np.float32)
    # Robust bounding span: resistant to one bad center while still catching parking maneuvers.
    qx=np.percentile(xs,[10,90]);qy=np.percentile(ys,[10,90])
    return float(math.hypot(float(qx[1]-qx[0]),float(qy[1]-qy[0])))


def _run_state_engine_v15_transition_guard(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """v13 baseline with a causal guard on EMPTY<->OCCUPIED transitions only.

    Stable OCCUPIED/EMPTY states are left alone. This avoids the v14 failure where
    weak/intermittent YOLO evidence could erase a long-established occupied slot.
    """
    base_local,_,_= _run_state_engine_v13_safe(evidence,params)
    if base_local.empty:
        return base_local,pd.DataFrame(),pd.DataFrame()
    base_idx={(round(float(r.time_sec),6),str(r.local_id)):r for r in base_local.itertuples()}
    ev=evidence.sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    slot_meta=ev.groupby('local_id',sort=False).first()
    window=float(params.get('transition_guard_window_sec',params.get('temporal_window_sec',10.0)))
    recent_sec=float(params.get('transition_guard_recent_sec',3.0))
    entry_motion_max=float(params.get('entry_guard_motion_span_px',45.0))
    entry_recent_motion_max=float(params.get('entry_guard_recent_motion_span_px',22.0))
    entry_settle=float(params.get('entry_guard_settle_sec',7.0))
    entry_recent_ratio=float(params.get('entry_guard_min_recent_hit_ratio',0.50))
    entry_mean_conf=float(params.get('entry_guard_min_mean_conf',0.08))
    motion_step=float(params.get('zone_motion_step_px',14.0))
    stationary_speed=float(params.get('stationary_speed_px_s',14.0))
    stationary_span=float(params.get('stationary_motion_span_px',32.0))
    visual_thr=float(params.get('occupied_visual_diff_threshold',0.08))
    no_full_req=float(params.get('exit_guard_no_full_sec',4.0))
    aux_window=float(params.get('exit_guard_aux_window_sec',5.0))
    aux_min_checks=int(params.get('exit_guard_aux_min_checks',2))
    aux_conf_req=float(params.get('exit_guard_aux_conf',0.25))
    aux_scales_req=int(params.get('exit_guard_aux_min_scales',2))
    fusion=str(params.get('global_merge','ANY')).upper();global_thr=float(params.get('global_confidence_threshold',0.0))

    hist:Dict[str,deque]=defaultdict(deque);last_center={};last_motion_t=defaultdict(lambda:-1e9);last_full_t=defaultdict(lambda:-1e9)
    state={};phase={};prev_local={};prev_global={};local_rows=[];global_rows=[];transitions=[]
    for lid,r in slot_meta.iterrows():
        lid=str(lid);st='OCCUPIED' if str(r.get('initial_state','UNKNOWN')).upper()=='OCCUPIED' else 'EMPTY'
        state[lid]=st;phase[lid]=st;prev_local[lid]=st

    for t,group in ev.groupby('time_sec',sort=True):
        t=float(t)
        # Update slot-zone history before considering baseline's proposed transition.
        for _,r in group.iterrows():
            lid=str(r['local_id']);fd=int(r.get('full_detected',r.get('detected',0)))>0
            fc=float(r.get('full_det_conf',r.get('det_conf',0.0))) if fd else 0.0
            cx=float(r.get('full_center_x',r.get('det_center_x',np.nan))) if fd else np.nan
            cy=float(r.get('full_center_y',r.get('det_center_y',np.nan))) if fd else np.nan
            speed=float(r.get('full_track_speed_px_s',r.get('track_speed_px_s',0.0))) if fd else 0.0
            mot=float(r.get('full_track_motion_span_px',r.get('track_motion_span_px',0.0))) if fd else 0.0
            if fd:
                last_full_t[lid]=t
                prev=last_center.get(lid)
                if prev is not None and t-float(prev[2])<=3.5 and math.hypot(cx-float(prev[0]),cy-float(prev[1]))>=motion_step:
                    last_motion_t[lid]=t
                if speed>stationary_speed or mot>stationary_span:last_motion_t[lid]=t
                if np.isfinite(cx) and np.isfinite(cy):last_center[lid]=(cx,cy,t)
            aux_checked=int(r.get('aux_requested',0))>0
            aux_conf=float(r.get('aux_crop_det_conf',r.get('crop_det_conf',0.0)))
            aux_support=int(r.get('aux_support_scales',0))
            aux_hit=bool(aux_checked and int(r.get('aux_crop_detected',r.get('crop_detected',0)))>0 and aux_conf>=aux_conf_req and aux_support>=aux_scales_req)
            rec={'t':t,'full_hit':int(fd),'det_conf':fc,'center_x':cx,'center_y':cy,'speed':speed,'motion_span':mot,
                 'visual_diff':float(r.get('visual_diff_initial',0.0)),'aux_checked':int(aux_checked),'aux_hit':int(aux_hit),
                 'aux_conf':aux_conf,'aux_support':aux_support}
            hist[lid].append(rec)
            while hist[lid] and t-float(hist[lid][0]['t'])>window+1e-6:hist[lid].popleft()

        now=[]
        for _,r in group.iterrows():
            lid=str(r['local_id']);cctv=str(r['cctv']);key=(round(t,6),lid);br=base_idx.get(key)
            desired=str(getattr(br,'state',state[lid])).upper() if br is not None else state[lid]
            bphase=str(getattr(br,'phase',desired)).upper() if br is not None else desired
            hh=list(hist[lid]);hits=[x for x in hh if x['full_hit']]
            recent=[x for x in hh if t-float(x['t'])<=recent_sec+1e-6];recent_hits=[x for x in recent if x['full_hit']]
            hit_ratio=len(hits)/max(1,len(hh));recent_ratio=len(recent_hits)/max(1,len(recent));mean_conf=float(np.mean([x['det_conf'] for x in hits])) if hits else 0.0
            motion10=_v15_motion_span(hh);motion_recent=_v15_motion_span(recent)
            settle=(t-float(last_motion_t[lid])) if last_motion_t[lid]>-1e8 else window+recent_sec
            no_full=(t-float(last_full_t[lid])) if last_full_t[lid]>-1e8 else window+recent_sec
            init=str(r.get('initial_state','UNKNOWN')).upper();vis=float(r.get('visual_diff_initial',0.0));appearance_occ=_appearance_occupied(init,vis,visual_thr)
            aux_recent=[x for x in hh if t-float(x['t'])<=aux_window+1e-6 and x.get('aux_checked',0)]
            aux_checks=len(aux_recent);aux_hits=sum(int(x.get('aux_hit',0)) for x in aux_recent)
            entry_ok=exit_ok=False;guard_reason='STABLE_NO_GUARD'

            if desired!=state[lid]:
                if state[lid]=='EMPTY' and desired=='OCCUPIED':
                    # Baseline says "new car". Delay only when the slot-zone still looks like a maneuver.
                    ghost_ok=bool(appearance_occ or mean_conf>=float(params.get('entry_no_appearance_conf',0.25)))
                    entry_ok=bool(hit_ratio>=float(params.get('entry_hit_ratio',0.35)) and
                                  recent_ratio>=entry_recent_ratio and mean_conf>=entry_mean_conf and ghost_ok and
                                  motion10<=entry_motion_max and motion_recent<=entry_recent_motion_max and settle>=entry_settle)
                    if entry_ok:
                        state[lid]='OCCUPIED';phase[lid]='OCCUPIED';guard_reason='ENTRY_CONFIRMED'
                    else:
                        phase[lid]='MANEUVERING';guard_reason='ENTRY_HELD_FOR_10S_MOTION'
                elif state[lid]=='OCCUPIED' and desired=='EMPTY':
                    # Baseline says "empty". Require three independent cues: FULL miss, AUX miss, appearance change.
                    exit_ok=bool(no_full>=no_full_req and len(recent_hits)==0 and aux_checks>=aux_min_checks and aux_hits==0 and not appearance_occ)
                    if exit_ok:
                        state[lid]='EMPTY';phase[lid]='EMPTY';guard_reason='EXIT_CONFIRMED_FULL_AUX_APPEARANCE'
                    else:
                        phase[lid]='LEAVING' if no_full>=no_full_req else 'OCCUPIED';guard_reason='EXIT_HELD_FOR_CONFIRMATION'
            else:
                # Crucial v15 rule: ten-second memory does NOT reclassify a stable state.
                phase[lid]=bphase if bphase in ('EMPTY','OCCUPIED','MANEUVERING','LEAVING') else state[lid]

            occ_score=float(getattr(br,'occupied_score',0.8 if state[lid]=='OCCUPIED' else 0.2)) if br is not None else (0.8 if state[lid]=='OCCUPIED' else 0.2)
            if state[lid]=='OCCUPIED':occ_score=max(0.51,occ_score)
            else:occ_score=min(0.49,occ_score)
            row={'time_sec':t,'timestamp':r['timestamp'],'cctv':cctv,'local_id':lid,'global_id':r['global_id'],
                 'evidence_mode':'FULL_TRANSITION_GUARD','evidence_source':str(getattr(br,'evidence_source','FULL')) if br is not None else 'FULL',
                 'state':state[lid],'phase':phase[lid],'occupied_score':occ_score,'baseline_desired_state':desired,
                 'window_hit_ratio':hit_ratio,'recent_hit_ratio':recent_ratio,'mean_detection_confidence':mean_conf,
                 'slot_zone_motion_10s_px':motion10,'slot_zone_motion_recent_px':motion_recent,'slot_zone_settle_sec':settle,
                 'time_since_full_detection_sec':no_full,'appearance_occupied':int(bool(appearance_occ)),
                 'aux_recent_checks':aux_checks,'aux_recent_hits':aux_hits,'entry_guard_ok':int(entry_ok),'exit_guard_ok':int(exit_ok),
                 'transition_guard_reason':guard_reason,'warmup':int(t<window-1e-6)}
            local_rows.append(row);now.append(row)
            if prev_local.get(lid)!=state[lid]:
                transitions.append({'scope':'LOCAL','time_sec':t,'timestamp':r['timestamp'],'id':lid,'global_id':r['global_id'],'from_state':prev_local.get(lid,''),'to_state':state[lid],'cctv':cctv,'phase':phase[lid],'dominant_track_id':-1,'evidence_mode':'FULL_TRANSITION_GUARD','guard_reason':guard_reason})
                prev_local[lid]=state[lid]

        by_global=defaultdict(list)
        for x in now:by_global[str(x['global_id'])].append(x)
        for gid,items in by_global.items():
            occ=sum(1 for x in items if x['state']=='OCCUPIED');scores=[float(x['occupied_score']) for x in items]
            if fusion=='CONF_MAX':gscore=max(scores) if scores else 0.0;global_occ=gscore>=global_thr
            elif fusion=='CONF_MEAN':gscore=float(np.mean(scores)) if scores else 0.0;global_occ=gscore>=global_thr
            elif fusion=='MAJORITY':gscore=float(np.mean(scores)) if scores else 0.0;global_occ=occ>=math.ceil(len(items)/2)
            else:gscore=max(scores) if scores else 0.0;global_occ=occ>0
            gstate='OCCUPIED' if global_occ else 'EMPTY';phases=[x['phase'] for x in items]
            gphase='MANEUVERING' if 'MANEUVERING' in phases else ('LEAVING' if 'LEAVING' in phases else gstate)
            grow={'time_sec':t,'timestamp':items[0]['timestamp'],'global_id':gid,'state':gstate,'phase':gphase,'global_score':gscore,
                  'evidence_mode':'FULL_TRANSITION_GUARD','source_local_slots':';'.join(x['local_id'] for x in items),'occupied_votes':occ,'total_votes':len(items),'warmup':int(t<window-1e-6)}
            global_rows.append(grow)
            if gid in prev_global and prev_global[gid]!=gstate:
                transitions.append({'scope':'GLOBAL','time_sec':t,'timestamp':items[0]['timestamp'],'id':gid,'global_id':gid,'from_state':prev_global[gid],'to_state':gstate,'cctv':'MULTI','phase':gphase,'dominant_track_id':-1,'evidence_mode':'FULL_TRANSITION_GUARD','guard_reason':'GLOBAL_FUSION'})
            prev_global[gid]=gstate
    return pd.DataFrame(local_rows),pd.DataFrame(global_rows),pd.DataFrame(transitions)


def search_state_parameters(evidence: pd.DataFrame, gt: pd.DataFrame, settings: Dict, output_dir: str,
                            progress=None, slot_gt_events_path: Optional[str]=None) -> Dict:
    """v15: v13 SAFE_BASELINE plus transition-only ten-second guard.

    TEST is never used for selection. The guard is allowed to replace baseline only when
    DEV safety metrics show no material regression; otherwise baseline remains deployable.
    """
    out_dir=ensure_dir(output_dir);temporal=settings.get('temporal',{});search=settings.get('state_search',{})
    base={
        'evidence_mode':'FULL','entry_hit_ratio':float(search.get('entry_hit_ratio',[0.35])[0]),'exit_hit_ratio':float(search.get('exit_hit_ratio',[0.10])[0]),
        'stable_track_ratio':float(search.get('stable_track_ratio',[0.55])[0]),'occupied_visual_diff_threshold':float(search.get('occupied_visual_diff_threshold',[0.08])[0]),
        'global_merge':'ANY','global_confidence_threshold':0.0,'entry_mean_conf_min':float(search.get('entry_mean_conf_min',[0.08])[0]),
        'entry_no_appearance_conf':float(search.get('entry_no_appearance_conf',[0.25])[0]),'full_min_conf':0.0,
        'temporal_window_sec':float(temporal.get('window_sec',10.0)),'recent_window_sec':float(temporal.get('recent_window_sec',3.0)),
        'stationary_speed_px_s':float(temporal.get('stationary_speed_px_s',14.0)),'stationary_motion_span_px':float(temporal.get('stationary_motion_span_px',32.0)),
        'crop_stationary_motion_span_px':float(temporal.get('crop_stationary_motion_span_px',28.0)),'max_track_switches_for_entry':int(temporal.get('max_track_switches_for_entry',1)),
    }
    guard={**base,
        'transition_guard_window_sec':float(temporal.get('transition_guard_window_sec',10.0)),'transition_guard_recent_sec':float(temporal.get('transition_guard_recent_sec',3.0)),
        'entry_guard_motion_span_px':float(temporal.get('entry_guard_motion_span_px',45.0)),'entry_guard_recent_motion_span_px':float(temporal.get('entry_guard_recent_motion_span_px',22.0)),
        'entry_guard_settle_sec':float(temporal.get('entry_guard_settle_sec',7.0)),'entry_guard_min_recent_hit_ratio':float(temporal.get('entry_guard_min_recent_hit_ratio',0.50)),
        'entry_guard_min_mean_conf':float(temporal.get('entry_guard_min_mean_conf',0.08)),'zone_motion_step_px':float(temporal.get('zone_motion_step_px',14.0)),
        'exit_guard_no_full_sec':float(temporal.get('exit_guard_no_full_sec',4.0)),'exit_guard_aux_window_sec':float(temporal.get('exit_guard_aux_window_sec',5.0)),
        'exit_guard_aux_min_checks':int(temporal.get('exit_guard_aux_min_checks',2)),'exit_guard_aux_conf':float(temporal.get('exit_guard_aux_conf',0.25)),
        'exit_guard_aux_min_scales':int(temporal.get('exit_guard_aux_min_scales',2)),
    }
    dev_end=float(settings.get('dev_end_sec',900));warmup=float(settings.get('evaluation_warmup_sec',10.0));gt_times=gt['time_sec'].tolist()
    slot_events=load_slot_gt_events(slot_gt_events_path) if slot_gt_events_path and Path(slot_gt_events_path).exists() else None
    variants=[]
    for idx,(name,runner,params) in enumerate([('SAFE_BASELINE',_run_state_engine_v13_safe,base),('TRANSITION_GUARD',_run_state_engine_v15_transition_guard,guard)]):
        local_df,global_df,transitions=runner(evidence,params);ev=evaluate_state_output(global_df,gt,dev_end,warmup)
        slot_eval=evaluate_slot_level(global_df,slot_events,gt_times,dev_end,warmup) if slot_events is not None else None
        variants.append({'name':name,'params':params,'local':local_df,'global':global_df,'transitions':transitions,'ev':ev,'slot_eval':slot_eval})
        if progress:progress((idx+1)/2.0,f'v15 temporal comparison {idx+1}/2 | {name}')
    b,g=variants;bd=b['ev']['DEV'];gd=g['ev']['DEV'];dev_confounded=bool(bd['false_empty_bias_rate']>=0.35 and bd['exact_rate']<0.60)
    # Conservative adoption: guard may be selected even on confounded DEV, but only if it does not
    # materially worsen the already-biased count metrics. This prevents another v14/G029 collapse.
    safe_guard=(gd['mae']<=bd['mae']+0.08 and gd['exact_rate']>=bd['exact_rate']-0.04 and gd['max_abs_error']<=bd['max_abs_error']+1 and gd['over_rate']<=bd['over_rate']+0.02)
    clearly_better=(gd['mae']<=bd['mae']-0.02 or gd['over_rate']<bd['over_rate']-0.02 or gd['exact_rate']>=bd['exact_rate']+0.03)
    slot_guard_better=False
    if b['slot_eval'] is not None and g['slot_eval'] is not None:
        slot_guard_better=(g['slot_eval']['DEV']['slot_accuracy']>=b['slot_eval']['DEV']['slot_accuracy']+0.01 and
                           g['slot_eval']['DEV']['false_empty_rate']<=b['slot_eval']['DEV']['false_empty_rate'])
    # v15 principle: v13 remains the deployment mainline. If count DEV is known-confounded by
    # advisory/missing slots, it cannot promote the experimental guard. A future slot-GT event
    # file can provide a trustworthy promotion signal instead.
    choose_guard=bool(safe_guard and ((not dev_confounded and clearly_better) or slot_guard_better))
    chosen=g if choose_guard else b
    reason=('TRANSITION_GUARD selected: it passed the DEV safety envelope with a trustworthy improvement signal.' if choose_guard else
            'SAFE_BASELINE retained: DEV count is confounded by candidate/missing-slot bias, or the guard lacked a trustworthy improvement signal.')
    rows=[]
    for v in variants:
        d=v['ev']['DEV'];te=v['ev']['TEST'];row={'temporal_variant':v['name'],**v['params'],'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],'dev_under_rate':d['false_empty_bias_rate'],'dev_over_rate':d['over_rate'],'dev_max_abs_error':d['max_abs_error'],'test_N':te['N'],'test_exact_rate':te['exact_rate'],'test_MAE':te['mae'],'test_under_rate':te['false_empty_bias_rate'],'test_over_rate':te['over_rate'],'test_max_abs_error':te['max_abs_error']}
        if v['slot_eval'] is not None:row.update({'dev_slot_accuracy':v['slot_eval']['DEV']['slot_accuracy'],'dev_slot_false_empty_rate':v['slot_eval']['DEV']['false_empty_rate'],'dev_all_slots_exact_time_rate':v['slot_eval']['DEV']['all_slots_exact_time_rate']})
        rows.append(row)
    board=pd.DataFrame(rows);board.to_csv(out_dir/'temporal_variant_comparison.csv',index=False,encoding='utf-8-sig');board.to_csv(out_dir/'state_search_leaderboard.csv',index=False,encoding='utf-8-sig');board.rename(columns={'temporal_variant':'evidence_mode'}).to_csv(out_dir/'detection_mode_comparison.csv',index=False,encoding='utf-8-sig')
    for v in variants:
        tag='baseline' if v['name']=='SAFE_BASELINE' else 'transition_guard';v['ev']['timeseries'].to_csv(out_dir/f'{tag}_count_timeseries.csv',index=False,encoding='utf-8-sig');v['global'].to_csv(out_dir/f'{tag}_global_slot_timeseries.csv',index=False,encoding='utf-8-sig');v['local'].to_csv(out_dir/f'{tag}_slot_timeseries.csv',index=False,encoding='utf-8-sig')
    local_df=chosen['local'];global_df=chosen['global'];transitions=chosen['transitions'];ev=chosen['ev'];slot_eval=chosen['slot_eval']
    local_df.to_csv(out_dir/'selected_slot_timeseries.csv',index=False,encoding='utf-8-sig');global_df.to_csv(out_dir/'selected_global_slot_timeseries.csv',index=False,encoding='utf-8-sig');transitions.to_csv(out_dir/'selected_state_transitions.csv',index=False,encoding='utf-8-sig');ev['timeseries'].to_csv(out_dir/'selected_count_timeseries.csv',index=False,encoding='utf-8-sig')
    selected_params={**chosen['params'],'temporal_variant':chosen['name']};save_json(out_dir/'selected_state_params.json',selected_params);pd.DataFrame([{'split':sp,**ev[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'metrics_dev_test.csv',index=False,encoding='utf-8-sig');ev['timeseries'][(ev['timeseries']['split']=='TEST')&(ev['timeseries']['error']!=0)].to_csv(out_dir/'test_error_cases.csv',index=False,encoding='utf-8-sig')
    audit={'selected_temporal_variant':chosen['name'],'reason':reason,'dev_confounded':dev_confounded,'guard_safe':safe_guard,'guard_clearly_better':clearly_better,'slot_guard_better':slot_guard_better,'selection_uses_test':False,'baseline_first':True};save_json(out_dir/'mode_selection_audit.json',audit)
    if slot_eval is not None:
        slot_eval['rows'].to_csv(out_dir/'slot_level_comparison.csv',index=False,encoding='utf-8-sig');slot_eval['by_slot'].to_csv(out_dir/'slot_level_by_slot.csv',index=False,encoding='utf-8-sig');slot_eval['rows'][(slot_eval['rows']['split']=='TEST')&(slot_eval['rows']['correct']==0)].to_csv(out_dir/'slot_level_test_errors.csv',index=False,encoding='utf-8-sig');pd.DataFrame([{'split':sp,**slot_eval[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'slot_level_metrics.csv',index=False,encoding='utf-8-sig')
    with (out_dir/'REPORT.txt').open('w',encoding='utf-8') as f:
        f.write('Parking Slot State Engine v15 - stable baseline + transition-only 10s guard\n\n');f.write(f'Warm-up excluded: 0 <= t < {warmup:.1f}s\nEvery decision uses only current/past frames.\n');f.write('Stable states are never reclassified by the 10s guard. The guard runs only when baseline proposes EMPTY<->OCCUPIED.\n');f.write('EMPTY->OCCUPIED: requires settled 10s slot-zone motion. OCCUPIED->EMPTY: requires FULL miss + AUX miss + appearance change.\n');f.write('Ground truth is never auto-edited from candidate detections.\n\n');f.write('SELECTION\n'+reason+f'\ndev_confounded={dev_confounded}\nselection_uses_TEST=False\n\n')
        for v in variants:
            d=v['ev']['DEV'];te=v['ev']['TEST'];f.write(f"[{v['name']}] DEV exact={d['exact_rate']*100:.2f}% MAE={d['mae']:.4f} under={d['false_empty_bias_rate']*100:.2f}% | TEST exact={te['exact_rate']*100:.2f}% MAE={te['mae']:.4f} max={te['max_abs_error']}\n")
        f.write(f'\nSELECTED={chosen["name"]}\n')
    return {'best_params':selected_params,'evaluation':ev,'slot_evaluation':slot_eval,'leaderboard':board,'mode_comparison':board,'selection_audit':audit,'output_dir':str(out_dir)}

# =============================================================================
# v15.2 RAW+ENHANCED selective segmentation assist
# The stable FULL detector and transition guard stay primary. Segmentation is an
# independent, selectively-invoked confirmation sensor for ambiguous slot crops.
# =============================================================================

def _seg_request_reason(r: pd.Series, cfg: Dict) -> str:
    """v15.2 lazy SEG trigger.

    Operational segmentation is deliberately much narrower than v15.1. Robustness
    against glare/moire/low-resolution is tested independently by the bounded
    degradation experiment; the live state engine calls SEG only where a second
    visual opinion can change a transition decision.
    """
    if not bool(cfg.get('enabled', True)) or int(r.get('aux_requested', 0)) <= 0:
        return ''
    reason = str(r.get('aux_trigger_reason', '')).strip().upper()
    allowed = {str(x).strip().upper() for x in cfg.get('lazy_trigger_reasons', ['RECENT_FULL_MISS','WEAK_FULL'])}
    if reason not in allowed:
        return ''
    if reason == 'WEAK_FULL':
        conf = float(r.get('full_det_conf', 0.0) or 0.0)
        if conf > float(cfg.get('weak_full_seg_conf_max', 0.14)):
            return ''
    if reason == 'RECENT_FULL_MISS':
        ratio = float(r.get('recent_full_hit_ratio', 0.0) or 0.0)
        if ratio > float(cfg.get('recent_full_hit_ratio_max', 0.55)):
            return ''
    return f'LAZY_{reason}'

def _seg_boxes_overlap(a: Sequence[float], b: Sequence[float], iou_thr: float = 0.35, center_thr: float = 28.0) -> bool:
    if box_iou(a, b) >= float(iou_thr):
        return True
    ac = box_center(a); bc = box_center(b)
    return math.hypot(float(ac[0])-float(bc[0]), float(ac[1])-float(bc[1])) <= float(center_thr)



def _v16_augment_empty_reference(base: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Calibrate an EMPTY-reference threshold from the causal warm-up of slots that began EMPTY."""
    out=base.copy(); cfg=dict(settings.get('empty_reference',{}) or {})
    for c,default in [('empty_ref_threshold',0.0),('empty_ref_margin',0.0),('empty_ref_positive',0)]:
        if c not in out.columns: out[c]=default
    if not bool(cfg.get('enabled',True)) or out.empty or 'visual_diff_initial' not in out.columns:
        return out
    cal_sec=float(cfg.get('calibration_sec',10.0));min_margin=float(cfg.get('min_margin',0.018));mad_mult=float(cfg.get('mad_multiplier',3.0))
    out['visual_diff_initial']=pd.to_numeric(out['visual_diff_initial'],errors='coerce').fillna(0.0)
    for lid,g in out.groupby('local_id',sort=False):
        init=str(g.iloc[0].get('initial_state','')).upper()
        if init!='EMPTY': continue
        gg=g.sort_values('time_sec');t0=float(gg['time_sec'].min());cal=gg[gg['time_sec']<=t0+cal_sec]['visual_diff_initial'].astype(float)
        if cal.empty: continue
        med=float(cal.median());mad=float(np.median(np.abs(cal.to_numpy()-med))) if len(cal) else 0.0
        thr=max(med+min_margin, med+mad_mult*1.4826*mad)
        idx=gg.index;vals=out.loc[idx,'visual_diff_initial'].astype(float)
        out.loc[idx,'empty_ref_threshold']=thr;out.loc[idx,'empty_ref_margin']=vals-thr;out.loc[idx,'empty_ref_positive']=(vals>=thr).astype(int)
    return out


def _v16_baseline_params(settings: Dict) -> Dict:
    temporal=settings.get('temporal',{}) or {};search=settings.get('state_search',{}) or {}
    first=lambda k,d: float(search.get(k,[d])[0] if isinstance(search.get(k,[d]),list) else search.get(k,d))
    return {
        'evidence_mode':'FULL','entry_hit_ratio':first('entry_hit_ratio',0.35),'exit_hit_ratio':first('exit_hit_ratio',0.10),
        'stable_track_ratio':first('stable_track_ratio',0.55),'occupied_visual_diff_threshold':first('occupied_visual_diff_threshold',0.08),
        'global_merge':'ANY','global_confidence_threshold':0.0,'entry_mean_conf_min':first('entry_mean_conf_min',0.08),
        'entry_no_appearance_conf':first('entry_no_appearance_conf',0.25),'full_min_conf':0.0,
        'temporal_window_sec':float(temporal.get('window_sec',10.0)),'recent_window_sec':float(temporal.get('recent_window_sec',3.0)),
        'stationary_speed_px_s':float(temporal.get('stationary_speed_px_s',14.0)),'stationary_motion_span_px':float(temporal.get('stationary_motion_span_px',32.0)),
        'crop_stationary_motion_span_px':float(temporal.get('crop_stationary_motion_span_px',28.0)),'max_track_switches_for_entry':int(temporal.get('max_track_switches_for_entry',1)),
    }


def _v16_transition_seg_requests(base: pd.DataFrame, settings: Dict) -> Dict[Tuple[float,str],str]:
    """Request expensive SEG at proposed state transitions, plus sparse sensor-disagreement checks."""
    reasons={}
    if base.empty: return reasons
    try:
        local,_,_=_run_state_engine_v13_safe(base,{**_v16_baseline_params(settings),'unlabeled_restart':bool(settings.get('unlabeled_restart',False))})
        for lid,g in local.sort_values(['local_id','time_sec']).groupby('local_id',sort=False):
            prev=None
            for r in g.itertuples():
                state=str(getattr(r,'state','EMPTY')).upper();t=round(float(getattr(r,'time_sec')),6)
                if prev is not None and state!=prev:
                    reasons[(t,str(lid))]='BASELINE_ENTRY_TRANSITION' if state=='OCCUPIED' else 'BASELINE_EXIT_TRANSITION'
                prev=state
    except Exception:
        pass
    cfg=dict(settings.get('segmentation_assist',{}) or {});interval=max(float(settings.get('evidence_sample_sec',1.0)),float(cfg.get('empty_ref_disagreement_interval_sec',5.0)))
    stride=max(1,int(round(interval/max(1e-6,float(settings.get('evidence_sample_sec',1.0))))))
    # If the empty-reference says "appearance changed" while FULL misses but crop evidence exists,
    # ask SEG sparsely. This is aimed at slots such as G029 that the baseline may never transition into.
    for lid,g in base.sort_values(['local_id','time_sec']).groupby('local_id',sort=False):
        for j,r in enumerate(g.itertuples()):
            if j%stride: continue
            if str(getattr(r,'initial_state','')).upper()!='EMPTY': continue
            eref=int(getattr(r,'empty_ref_positive',0));full=int(getattr(r,'full_detected',0));crop=int(getattr(r,'crop_detected',0))
            if eref and not full and crop:
                reasons.setdefault((round(float(getattr(r,'time_sec')),6),str(lid)),'EMPTY_REF_CROP_DISAGREEMENT')
    return reasons

def extract_segmentation_assist(video_path: str, rois: Dict[str,Sequence[int]], slots_path: str, settings: Dict,
                                evidence: pd.DataFrame, output_dir: str, progress=None) -> pd.DataFrame:
    """v16 transition-time multi-sensor recovery assist.

    The real CCTV distribution calibrates per-camera glare/stripe/low-resolution tails.
    Each requested slot is evaluated as RAW SEG plus a condition-specific ADAPTIVE SEG.
    For glare/stripe tails only, a small recovery detector also sees the model-specific
    adaptive image. Causal temporal median is used for stripe recovery. Recovery evidence
    may protect an established OCCUPIED state but never creates occupancy by itself.
    """
    from segmentation import (
        VehicleSegmenter, choose_best_segment, conditioned_parking_crop, image_quality_metrics,
        build_relative_quality_thresholds, classify_quality_condition,
    )
    from detector import VehicleDetector

    out_dir = ensure_dir(output_dir)
    cfg = dict(settings.get('segmentation_assist', {}) or {})
    cfg['sample_sec'] = float(settings.get('evidence_sample_sec', 1.0))
    base = evidence.copy()
    text_cols = {'seg_request_reason','seg_status','seg_source','raw_seg_status','enh_seg_status','seg_adaptive_condition','adaptive_det_status'}
    seg_cols = [
        'seg_available','seg_requested','seg_request_reason','seg_detected','seg_conf','seg_cls',
        'seg_mask_ratio','seg_core_overlap_ratio','seg_point_covered','seg_box_x1','seg_box_y1','seg_box_x2','seg_box_y2',
        'seg_one_mask_one_slot','seg_status','seg_source',
        'raw_seg_available','raw_seg_detected','raw_seg_conf','raw_seg_mask_ratio','raw_seg_core_overlap_ratio','raw_seg_point_covered','raw_seg_status',
        'enh_seg_available','enh_seg_detected','enh_seg_conf','enh_seg_mask_ratio','enh_seg_core_overlap_ratio','enh_seg_point_covered','enh_seg_status',
        'seg_input_highlight_ratio','seg_input_dark_ratio','seg_input_sharpness','seg_input_stripe_energy_ratio','seg_adaptive_condition','seg_condition_score',
        'adaptive_det_requested','adaptive_det_detected','adaptive_det_conf','adaptive_det_status'
    ]
    for c in seg_cols:
        if c not in base.columns:
            base[c] = '' if c in text_cols else 0.0
    if not bool(cfg.get('enabled', True)) or base.empty:
        base['seg_status'] = 'DISABLED'
        base.to_csv(out_dir/'segmentation_evidence.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame([{'enabled':False,'requested_rows':0,'positive_rows':0,'request_ratio':0.0}]).to_csv(out_dir/'segmentation_summary.csv',index=False,encoding='utf-8-sig')
        return base

    slots = load_slots(slots_path)
    slot_lookup = {str(s['local_id']): s for s in slots}
    by_cctv = defaultdict(list)
    for s in slots:
        by_cctv[str(s['cctv'])].append(s)

    base=_v16_augment_empty_reference(base,settings)
    transition_reasons=_v16_transition_seg_requests(base,settings)
    reasons=[]; req=[]
    for _,r in base.iterrows():
        reason=transition_reasons.get((round(float(r.get('time_sec',0.0)),6),str(r.get('local_id',''))),'')
        reasons.append(reason); req.append(bool(reason))
    base['seg_requested']=np.asarray(req,dtype=np.int32); base['seg_request_reason']=reasons
    requested_total=int(base['seg_requested'].sum())
    request_indices=base.index[base['seg_requested'].astype(int)>0].tolist(); grouped=defaultdict(list)
    for idx in request_indices:
        grouped[round(float(base.at[idx,'time_sec']),6)].append(idx)
    times=sorted(grouped)

    # ---------- Phase A: measure the actual requested-crop distribution ----------
    quality_rows=[]
    cap_q=open_video(video_path)
    try:
        for ti,t in enumerate(times):
            frame=read_frame_at(cap_q,float(t)); by_cam=defaultdict(list)
            for idx in grouped[t]: by_cam[str(base.at[idx,'cctv'])].append(idx)
            for cctv,cam_idxs in by_cam.items():
                if cctv not in rois: continue
                warped=crop_roi(frame,rois[cctv])
                for idx in cam_idxs:
                    r=base.loc[idx]; lid=str(r['local_id']); s=slot_lookup.get(lid)
                    if s is None: continue
                    try: x1=int(r.get('crop_x1'));y1=int(r.get('crop_y1'));x2=int(r.get('crop_x2'));y2=int(r.get('crop_y2'))
                    except Exception:
                        px,py=map(float,s.get('point',[warped.shape[1]/2,warped.shape[0]/2]));x1=int(px-110);x2=int(px+110);y1=int(py-125);y2=int(py+125)
                    x1=max(0,min(warped.shape[1]-1,x1));x2=max(x1+1,min(warped.shape[1],x2));y1=max(0,min(warped.shape[0]-1,y1));y2=max(y1+1,min(warped.shape[0],y2))
                    crop=warped[y1:y2,x1:x2]
                    if crop.size==0: continue
                    q=image_quality_metrics(crop)
                    base.at[idx,'seg_input_highlight_ratio']=q['highlight_ratio'];base.at[idx,'seg_input_dark_ratio']=q['dark_ratio'];base.at[idx,'seg_input_sharpness']=q['sharpness_laplacian_var'];base.at[idx,'seg_input_stripe_energy_ratio']=q['stripe_energy_ratio']
                    quality_rows.append({'idx':int(idx),'cctv':cctv,**q})
            if progress and ti%max(1,len(times)//20)==0:
                progress(0.08*(ti+1)/max(1,len(times)),f'Quality calibration {ti+1}/{len(times)}')
    finally:
        cap_q.release()

    from split_protocol import calibration_quality_rows
    calibration_rows=calibration_quality_rows(quality_rows,base,settings)
    thresholds=settings.get('frozen_segmentation_quality_thresholds')
    if thresholds is None: thresholds=build_relative_quality_thresholds(calibration_rows,cfg)
    th_rows=[]
    for cam,th in thresholds.items():
        if cam=='__GLOBAL__': continue
        th_rows.append({'cctv':cam,**th})
    pd.DataFrame(th_rows).to_csv(out_dir/'segmentation_condition_thresholds.csv',index=False,encoding='utf-8-sig')
    cond_counts=defaultdict(int)
    for q in quality_rows:
        idx=int(q['idx']); cctv=str(q['cctv'])
        cond,score=classify_quality_condition(q,thresholds.get(cctv,thresholds.get('__GLOBAL__',{})))
        base.at[idx,'seg_adaptive_condition']=cond; base.at[idx,'seg_condition_score']=score; cond_counts[(cctv,cond)]+=1
    pd.DataFrame([{'cctv':c,'condition':cond,'N':n} for (c,cond),n in sorted(cond_counts.items())]).to_csv(out_dir/'segmentation_condition_counts.csv',index=False,encoding='utf-8-sig')

    try:
        segmenter=VehicleSegmenter(cfg); segmenter._load()
    except Exception as exc:
        base.loc[base['seg_requested'].astype(int)>0,'seg_status']='MODEL_UNAVAILABLE';base['seg_available']=0
        (out_dir/'SEGMENTATION_UNAVAILABLE.txt').write_text(
            'Parking Slot Engine v16 segmentation assist could not load the configured model.\n\n'
            f"model={cfg.get('model','yolov8m-seg.pt')}\nerror={type(exc).__name__}: {exc}\n\n"
            'SAFE_BASELINE and TRANSITION_GUARD remain valid.\n',encoding='utf-8')
        base.to_csv(out_dir/'segmentation_evidence.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{'enabled':True,'model_available':False,'requested_rows':requested_total,'positive_rows':0,
                       'request_ratio':requested_total/max(1,len(base)),'model':cfg.get('model','')}]).to_csv(out_dir/'segmentation_summary.csv',index=False,encoding='utf-8-sig')
        return base

    recovery_detector=None
    if bool(cfg.get('adaptive_detector_enabled',True)):
        try:
            dcfg={'model':'yolov8m.pt','imgsz':int(cfg.get('imgsz',640)),'conf':float(cfg.get('adaptive_detector_conf',0.05)),
                  'vehicle_class_ids':cfg.get('vehicle_class_ids',[2,3,5,7]),'tiling':False,
                  'physical_vehicle_nms_iou':0.42,'physical_vehicle_overlap_min':0.68}
            recovery_detector=VehicleDetector(dcfg); recovery_detector._load()
        except Exception:
            recovery_detector=None

    batch_size=max(1,int(cfg.get('batch_size',12)))
    min_core=float(cfg.get('min_core_overlap_ratio',0.08));positive_conf=float(cfg.get('positive_conf',0.10));positive_core=float(cfg.get('positive_core_overlap_ratio',0.12))
    temporal_frames=max(3,int(cfg.get('temporal_clean_frames',5))); temporal_step=float(cfg.get('temporal_clean_step_sec',1.0))
    detector_conditions={str(x).upper() for x in cfg.get('adaptive_detector_conditions',['SUN_GLARE','MONITOR_STRIPES'])}
    cap=open_video(video_path); hist_cap=open_video(video_path); processed=0

    def temporal_crops(t,cctv,rect):
        x1,y1,x2,y2=rect; arr=[]
        for k in reversed(range(temporal_frames)):
            tt=max(float(settings.get('inference_start_sec',0.0)),float(t)-k*temporal_step)
            fr=read_frame_at(hist_cap,tt)
            if cctv not in rois: continue
            wr=crop_roi(fr,rois[cctv]); xx1=max(0,min(wr.shape[1]-1,x1));xx2=max(xx1+1,min(wr.shape[1],x2));yy1=max(0,min(wr.shape[0]-1,y1));yy2=max(yy1+1,min(wr.shape[0],y2))
            cr=wr[yy1:yy2,xx1:xx2].copy()
            if cr.size: arr.append(cr)
        return arr

    def candidate_from(idx,lid,x1,y1,dets,s,cctv,source):
        best=choose_best_segment(dets,min_core)
        avail_col='raw_seg_available' if source=='RAW' else 'enh_seg_available';stat_col='raw_seg_status' if source=='RAW' else 'enh_seg_status'
        base.at[idx,avail_col]=1
        if best is None:
            base.at[idx,stat_col]='NEGATIVE';return None
        gbox=[best.x1+x1,best.y1+y1,best.x2+x1,best.y2+y1];gc=box_center(gbox);same=by_cctv.get(cctv,[])
        nearest=min(same,key=lambda q:math.hypot(float(q['point'][0])-gc[0],float(q['point'][1])-gc[1])) if same else s
        if str(nearest.get('local_id'))!=lid:
            base.at[idx,stat_col]='REJECTED_NEIGHBOR_SLOT';return None
        positive=bool(best.conf>=positive_conf and (best.point_covered or best.core_overlap_ratio>=positive_core))
        return {'idx':idx,'lid':lid,'box':gbox,'conf':float(best.conf),'cls':int(best.cls),'mask_ratio':float(best.mask_ratio),'core':float(best.core_overlap_ratio),'point':int(best.point_covered),'score':float(best.score()),'positive':positive,'source':source}

    try:
        for ti,t in enumerate(times):
            frame=read_frame_at(cap,float(t));by_cam=defaultdict(list)
            for idx in grouped[t]:by_cam[str(base.at[idx,'cctv'])].append(idx)
            for cctv,cam_idxs in by_cam.items():
                if cctv not in rois:
                    for idx in cam_idxs:base.at[idx,'seg_status']='NO_ROI'
                    continue
                warped=crop_roi(frame,rois[cctv]);raw_imgs=[];enh_imgs=[];points=[];radii=[];meta=[];det_jobs=[];det_meta=[]
                for idx in cam_idxs:
                    r=base.loc[idx];lid=str(r['local_id']);s=slot_lookup.get(lid)
                    if s is None:base.at[idx,'seg_status']='NO_SLOT';continue
                    try:x1=int(r.get('crop_x1'));y1=int(r.get('crop_y1'));x2=int(r.get('crop_x2'));y2=int(r.get('crop_y2'))
                    except Exception:
                        px,py=map(float,s.get('point',[warped.shape[1]/2,warped.shape[0]/2]));x1=int(px-110);x2=int(px+110);y1=int(py-125);y2=int(py+125)
                    x1=max(0,min(warped.shape[1]-1,x1));x2=max(x1+1,min(warped.shape[1],x2));y1=max(0,min(warped.shape[0]-1,y1));y2=max(y1+1,min(warped.shape[0],y2))
                    crop=warped[y1:y2,x1:x2].copy();px,py=map(float,s.get('point',[0,0]));lp=(px-x1,py-y1);rad=float(r.get('crop_core_radius_px',max(10.0,0.15*min(crop.shape[:2]))))
                    cond=str(base.at[idx,'seg_adaptive_condition'] or 'CLEAN').upper()
                    hist=temporal_crops(float(t),cctv,(x1,y1,x2,y2)) if cond in ('MONITOR_STRIPES','LOW_RES') else None
                    det_adapt=conditioned_parking_crop(crop,cond,temporal_frames=hist)
                    # v16 model-specific input: temporal median is a YOLO recovery path for stripes;
                    # SEG keeps RAW stripes because v15.4 GT showed temporal masks collapsing.
                    seg_adapt=crop if cond=='MONITOR_STRIPES' else det_adapt
                    raw_imgs.append(crop);enh_imgs.append(seg_adapt);points.append(lp);radii.append(rad);meta.append((idx,lid,x1,y1,s,cond,lp,rad))
                    if recovery_detector is not None and cond in detector_conditions:
                        base.at[idx,'adaptive_det_requested']=1;det_jobs.append(det_adapt);det_meta.append((idx,lp,rad))
                raw_res=[];enh_res=[]
                for st in range(0,len(raw_imgs),batch_size):
                    raw_res.extend(segmenter.segment_batch(raw_imgs[st:st+batch_size],points[st:st+batch_size],radii[st:st+batch_size]))
                    enh_res.extend(segmenter.segment_batch(enh_imgs[st:st+batch_size],points[st:st+batch_size],radii[st:st+batch_size]))
                if recovery_detector is not None and det_jobs:
                    dr=[]
                    for st in range(0,len(det_jobs),batch_size): dr.extend(recovery_detector.detect_batch(det_jobs[st:st+batch_size]))
                    for (idx,lp,rad),dets in zip(det_meta,dr):
                        best=0.0
                        for d in dets:
                            cx,cy=box_center(d.box);inside=(d.x1<=lp[0]<=d.x2 and d.y1<=lp[1]<=d.y2);dist=math.hypot(cx-lp[0],cy-lp[1])
                            if inside or dist<=max(18.0,1.35*rad): best=max(best,float(d.conf))
                        base.at[idx,'adaptive_det_detected']=int(best>0);base.at[idx,'adaptive_det_conf']=best;base.at[idx,'adaptive_det_status']='POSITIVE' if best>0 else 'NEGATIVE'
                source_candidates={'RAW':[],'ADAPTIVE':[]}
                for mm,rr,er in zip(meta,raw_res,enh_res):
                    idx,lid,x1,y1,s,cond,lp,rad=mm
                    for source,dets in [('RAW',rr),('ADAPTIVE',er)]:
                        cand=candidate_from(idx,lid,x1,y1,dets,s,cctv,source)
                        if cand is not None:source_candidates[source].append(cand)
                kept_by_source={}
                for source,cands in source_candidates.items():
                    kept=[]
                    for cand in sorted(cands,key=lambda z:z['score'],reverse=True):
                        if any(_seg_boxes_overlap(cand['box'],k['box']) for k in kept):
                            base.at[cand['idx'],'raw_seg_status' if source=='RAW' else 'enh_seg_status']='REJECTED_DUPLICATE_MASK';continue
                        kept.append(cand)
                    kept_by_source[source]=kept
                    for cand in kept:
                        prefix='raw_seg' if source=='RAW' else 'enh_seg';idx=cand['idx']
                        base.at[idx,prefix+'_detected']=int(cand['positive']);base.at[idx,prefix+'_conf']=cand['conf'];base.at[idx,prefix+'_mask_ratio']=cand['mask_ratio'];base.at[idx,prefix+'_core_overlap_ratio']=cand['core'];base.at[idx,prefix+'_point_covered']=cand['point'];base.at[idx,prefix+'_status']='POSITIVE' if cand['positive'] else 'WEAK_MASK'
                combined=[c for arr in kept_by_source.values() for c in arr if c['positive']];clusters=[]
                for cand in sorted(combined,key=lambda z:z['score'],reverse=True):
                    hit=None
                    for cl in clusters:
                        if any(_seg_boxes_overlap(cand['box'],x['box']) for x in cl):hit=cl;break
                    if hit is None:clusters.append([cand])
                    else:hit.append(cand)
                winners={}
                for cl in clusters:
                    by_lid=defaultdict(list)
                    for c in cl:by_lid[c['lid']].append(c)
                    winner=max(by_lid.items(),key=lambda kv:max(x['score'] for x in kv[1]))[0]
                    for lid,arr in by_lid.items():
                        if lid!=winner:
                            for c in arr:base.at[c['idx'],'seg_status']='REJECTED_CROSS_SOURCE_NEIGHBOR'
                            continue
                        winners.setdefault(arr[0]['idx'],[]).extend(arr)
                for idx,arr in winners.items():
                    best=max(arr,key=lambda z:z['score']);sources={x['source'] for x in arr};src='BOTH' if len(sources)>1 else next(iter(sources))
                    base.at[idx,'seg_available']=1;base.at[idx,'seg_detected']=1;base.at[idx,'seg_conf']=best['conf'];base.at[idx,'seg_cls']=best['cls'];base.at[idx,'seg_mask_ratio']=best['mask_ratio'];base.at[idx,'seg_core_overlap_ratio']=best['core'];base.at[idx,'seg_point_covered']=best['point'];base.at[idx,'seg_one_mask_one_slot']=1;base.at[idx,'seg_box_x1']=best['box'][0];base.at[idx,'seg_box_y1']=best['box'][1];base.at[idx,'seg_box_x2']=best['box'][2];base.at[idx,'seg_box_y2']=best['box'][3];base.at[idx,'seg_source']=src;base.at[idx,'seg_status']='POSITIVE_'+src
                for idx in cam_idxs:
                    if int(base.at[idx,'seg_available'])==0: base.at[idx,'seg_available']=int(base.at[idx,'raw_seg_available'] or base.at[idx,'enh_seg_available'])
                    if not str(base.at[idx,'seg_status']):
                        rp=int(base.at[idx,'raw_seg_detected']);ep=int(base.at[idx,'enh_seg_detected']);base.at[idx,'seg_source']='BOTH' if rp and ep else ('RAW' if rp else ('ADAPTIVE' if ep else 'NONE'));base.at[idx,'seg_status']='NEGATIVE_BOTH' if not (rp or ep) else 'POSITIVE_'+str(base.at[idx,'seg_source'])
                processed+=len(cam_idxs)
            if progress:progress(0.08+0.92*(ti+1)/max(1,len(times)),f'v16 transition-time RAW+ADAPTIVE SEG {ti+1}/{len(times)} | requested rows {processed}/{requested_total}')
    finally:
        cap.release();hist_cap.release()

    int_cols=['seg_available','seg_requested','seg_detected','seg_point_covered','seg_one_mask_one_slot','raw_seg_available','raw_seg_detected','raw_seg_point_covered','enh_seg_available','enh_seg_detected','enh_seg_point_covered','adaptive_det_requested','adaptive_det_detected']
    for c in int_cols:base[c]=pd.to_numeric(base[c],errors='coerce').fillna(0).astype(int)
    base.to_csv(out_dir/'segmentation_evidence.csv',index=False,encoding='utf-8-sig')
    reqdf=base[base['seg_requested']>0];pos=reqdf[reqdf['seg_detected']>0]
    summary={'enabled':True,'model_available':True,'model':cfg.get('model',''),'total_slot_frames':len(base),'requested_rows':len(reqdf),'positive_rows':len(pos),'request_ratio':len(reqdf)/max(1,len(base)),
             'positive_ratio_within_requests':len(pos)/max(1,len(reqdf)),'raw_positive_rows':int(reqdf['raw_seg_detected'].sum()),'adaptive_positive_rows':int(reqdf['enh_seg_detected'].sum()),
             'adaptive_only_positive_rows':int(((reqdf['raw_seg_detected']==0)&(reqdf['enh_seg_detected']==1)).sum()),'raw_only_positive_rows':int(((reqdf['raw_seg_detected']==1)&(reqdf['enh_seg_detected']==0)).sum()),
             'adaptive_detector_requested_rows':int(reqdf['adaptive_det_requested'].sum()),'adaptive_detector_positive_rows':int(reqdf['adaptive_det_detected'].sum()),
             'one_mask_one_slot_enforced':True,'condition_classifier':'per_cctv_percentile','lazy_trigger_policy':'baseline_state_transition_plus_empty_ref_crop_disagreement'}
    pd.DataFrame([summary]).to_csv(out_dir/'segmentation_summary.csv',index=False,encoding='utf-8-sig')
    return base

def _rebuild_global_from_local_v15_1(local_df: pd.DataFrame, params: Dict, evidence_mode: str) -> Tuple[pd.DataFrame,pd.DataFrame]:
    if local_df.empty:
        return pd.DataFrame(),pd.DataFrame()
    fusion=str(params.get('global_merge','ANY')).upper();global_thr=float(params.get('global_confidence_threshold',0.0))
    global_rows=[];trans=[];prev={}
    for t,g in local_df.sort_values(['time_sec','local_id']).groupby('time_sec',sort=True):
        for gid,items_df in g.groupby('global_id',sort=False):
            items=items_df.to_dict('records');occ=sum(1 for x in items if str(x.get('state','')).upper()=='OCCUPIED');scores=[float(x.get('occupied_score',0.0)) for x in items]
            if fusion=='CONF_MAX':gscore=max(scores) if scores else 0.0;go=gscore>=global_thr
            elif fusion=='CONF_MEAN':gscore=float(np.mean(scores)) if scores else 0.0;go=gscore>=global_thr
            elif fusion=='MAJORITY':gscore=float(np.mean(scores)) if scores else 0.0;go=occ>=math.ceil(len(items)/2)
            else:gscore=max(scores) if scores else 0.0;go=occ>0
            state='OCCUPIED' if go else 'EMPTY';ph=[str(x.get('phase','')) for x in items]
            phase='MANEUVERING' if 'MANEUVERING' in ph else ('LEAVING' if 'LEAVING' in ph else state)
            row={'time_sec':float(t),'timestamp':items[0].get('timestamp',format_timestamp(float(t))),'global_id':str(gid),'state':state,'phase':phase,
                 'global_score':gscore,'evidence_mode':evidence_mode,'source_local_slots':';'.join(str(x.get('local_id','')) for x in items),
                 'occupied_votes':occ,'total_votes':len(items),'warmup':int(max(int(x.get('warmup',0)) for x in items) if items else 0)}
            global_rows.append(row)
            if gid in prev and prev[gid]!=state:
                trans.append({'scope':'GLOBAL','time_sec':float(t),'timestamp':row['timestamp'],'id':str(gid),'global_id':str(gid),'from_state':prev[gid],'to_state':state,'cctv':'MULTI','phase':phase,'dominant_track_id':-1,'evidence_mode':evidence_mode,'guard_reason':'GLOBAL_FUSION'})
            prev[gid]=state
    return pd.DataFrame(global_rows),pd.DataFrame(trans)


def _run_state_engine_v15_1_seg_assist(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Transition guard plus conservative segmentation confirmation.

    SEG can protect an established OCCUPIED state from a false clear. SEG alone can never
    create a new OCCUPIED state. A negative SEG result is only one confirmation cue; it is
    not allowed to clear a slot by itself.
    """
    local,_,base_trans=_run_state_engine_v15_transition_guard(evidence,params)
    if local.empty:
        return local,pd.DataFrame(),pd.DataFrame()
    ev_idx={(round(float(r.time_sec),6),str(r.local_id)):r for r in evidence.itertuples()}
    protect_conf=float(params.get('seg_protect_occupied_conf',0.10))
    protect_core=float(params.get('seg_protect_occupied_core_overlap_ratio',0.12))
    rows=[];trans=[]
    for lid,g in local.sort_values(['local_id','time_sec']).groupby('local_id',sort=False):
        carried=None;prev=None
        for rr in g.itertuples():
            d=rr._asdict();key=(round(float(d['time_sec']),6),str(lid));er=ev_idx.get(key)
            desired=str(d.get('state','EMPTY')).upper()
            if carried is None:
                carried=desired
            seg_req=int(getattr(er,'seg_requested',0)) if er is not None else 0
            seg_av=int(getattr(er,'seg_available',0)) if er is not None else 0
            seg_hit=int(getattr(er,'seg_detected',0)) if er is not None else 0
            seg_conf=float(getattr(er,'seg_conf',0.0)) if er is not None else 0.0
            seg_core=float(getattr(er,'seg_core_overlap_ratio',0.0)) if er is not None else 0.0
            seg_point=int(getattr(er,'seg_point_covered',0)) if er is not None else 0
            seg_source=str(getattr(er,'seg_source','NONE')) if er is not None else 'NONE'
            raw_seg_hit=int(getattr(er,'raw_seg_detected',0)) if er is not None else 0
            enh_seg_hit=int(getattr(er,'enh_seg_detected',0)) if er is not None else 0
            seg_positive=bool(seg_av and seg_hit and seg_conf>=protect_conf and (seg_point or seg_core>=protect_core))
            adap_det_hit=int(getattr(er,'adaptive_det_detected',0)) if er is not None else 0
            adap_det_conf=float(getattr(er,'adaptive_det_conf',0.0)) if er is not None else 0.0
            adap_det_positive=bool(adap_det_hit and adap_det_conf>=float(params.get('adaptive_det_protect_conf',0.12)))
            decision='NO_SEG_INTERVENTION'
            # Recovery sensors may only protect an established OCCUPIED state; they never create occupancy.
            if carried=='OCCUPIED' and desired=='EMPTY' and (seg_positive or adap_det_positive):
                d['state']='OCCUPIED';d['phase']='OCCUPIED';d['occupied_score']=max(0.51,float(d.get('occupied_score',0.51)))
                decision='BLOCK_EMPTY_SEG_VEHICLE' if seg_positive else 'BLOCK_EMPTY_ADAPTIVE_DET_VEHICLE'
            else:
                d['state']=desired
                if carried=='EMPTY' and desired=='OCCUPIED':
                    decision='ENTRY_BASELINE_GUARD_ONLY_SEG_NOT_CREATOR' if not seg_positive else 'ENTRY_CONFIRMED_SEG_SUPPORT'
                elif carried=='OCCUPIED' and desired=='EMPTY':
                    decision='EXIT_SEG_NEGATIVE_OR_UNAVAILABLE_OTHER_GUARDS_DECIDE'
            carried=str(d['state']).upper()
            d.update({'evidence_mode':'SEG_ASSIST','seg_requested':seg_req,'seg_available':seg_av,'seg_detected':seg_hit,'seg_conf':seg_conf,
                      'seg_core_overlap_ratio':seg_core,'seg_point_covered':seg_point,'seg_source':seg_source,
                      'raw_seg_detected':raw_seg_hit,'enh_seg_detected':enh_seg_hit,'adaptive_det_detected':adap_det_hit,'adaptive_det_conf':adap_det_conf,'seg_decision':decision})
            rows.append(d)
            if prev is not None and prev!=carried:
                trans.append({'scope':'LOCAL','time_sec':float(d['time_sec']),'timestamp':d['timestamp'],'id':str(lid),'global_id':d['global_id'],
                              'from_state':prev,'to_state':carried,'cctv':d['cctv'],'phase':d.get('phase',carried),'dominant_track_id':-1,
                              'evidence_mode':'SEG_ASSIST','guard_reason':decision})
            prev=carried
    local2=pd.DataFrame(rows).sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    global2,gtrans=_rebuild_global_from_local_v15_1(local2,params,'SEG_ASSIST')
    return local2,global2,pd.concat([pd.DataFrame(trans),gtrans],ignore_index=True) if (trans or not gtrans.empty) else pd.DataFrame()


# Keep a handle to the v15 implementation for diagnostic reuse if needed.
_search_state_parameters_v15 = search_state_parameters


def search_state_parameters(evidence: pd.DataFrame, gt: pd.DataFrame, settings: Dict, output_dir: str,
                            progress=None, slot_gt_events_path: Optional[str]=None) -> Dict:
    """v15.4 baseline-first comparison: SAFE_BASELINE / TRANSITION_GUARD / SEG_ASSIST.

    Selection uses DEV (and optional slot-GT) only. TEST is strictly evaluation-only.
    SEG_ASSIST is conservative: it may prevent a false EMPTY transition but never creates
    occupancy from segmentation alone.
    """
    out_dir=ensure_dir(output_dir);temporal=settings.get('temporal',{});search=settings.get('state_search',{});segcfg=settings.get('segmentation_assist',{}) or {}
    base={
        'evidence_mode':'FULL','entry_hit_ratio':float(search.get('entry_hit_ratio',[0.35])[0]),'exit_hit_ratio':float(search.get('exit_hit_ratio',[0.10])[0]),
        'stable_track_ratio':float(search.get('stable_track_ratio',[0.55])[0]),'occupied_visual_diff_threshold':float(search.get('occupied_visual_diff_threshold',[0.08])[0]),
        'global_merge':'ANY','global_confidence_threshold':0.0,'entry_mean_conf_min':float(search.get('entry_mean_conf_min',[0.08])[0]),
        'entry_no_appearance_conf':float(search.get('entry_no_appearance_conf',[0.25])[0]),'full_min_conf':0.0,
        'temporal_window_sec':float(temporal.get('window_sec',10.0)),'recent_window_sec':float(temporal.get('recent_window_sec',3.0)),
        'stationary_speed_px_s':float(temporal.get('stationary_speed_px_s',14.0)),'stationary_motion_span_px':float(temporal.get('stationary_motion_span_px',32.0)),
        'crop_stationary_motion_span_px':float(temporal.get('crop_stationary_motion_span_px',28.0)),'max_track_switches_for_entry':int(temporal.get('max_track_switches_for_entry',1)),
    }
    guard={**base,
        'transition_guard_window_sec':float(temporal.get('transition_guard_window_sec',10.0)),'transition_guard_recent_sec':float(temporal.get('transition_guard_recent_sec',3.0)),
        'entry_guard_motion_span_px':float(temporal.get('entry_guard_motion_span_px',45.0)),'entry_guard_recent_motion_span_px':float(temporal.get('entry_guard_recent_motion_span_px',22.0)),
        'entry_guard_settle_sec':float(temporal.get('entry_guard_settle_sec',7.0)),'entry_guard_min_recent_hit_ratio':float(temporal.get('entry_guard_min_recent_hit_ratio',0.50)),
        'entry_guard_min_mean_conf':float(temporal.get('entry_guard_min_mean_conf',0.08)),'zone_motion_step_px':float(temporal.get('zone_motion_step_px',14.0)),
        'exit_guard_no_full_sec':float(temporal.get('exit_guard_no_full_sec',4.0)),'exit_guard_aux_window_sec':float(temporal.get('exit_guard_aux_window_sec',5.0)),
        'exit_guard_aux_min_checks':int(temporal.get('exit_guard_aux_min_checks',2)),'exit_guard_aux_conf':float(temporal.get('exit_guard_aux_conf',0.25)),
        'exit_guard_aux_min_scales':int(temporal.get('exit_guard_aux_min_scales',2)),
    }
    segp={**guard,'seg_protect_occupied_conf':float(segcfg.get('protect_occupied_conf',0.10)),
          'seg_protect_occupied_core_overlap_ratio':float(segcfg.get('protect_occupied_core_overlap_ratio',0.12)),
          'adaptive_det_protect_conf':float(segcfg.get('adaptive_detector_protect_conf',0.12))}
    dev_end=float(settings.get('dev_end_sec',900));warmup=float(settings.get('evaluation_warmup_sec',10.0));gt_times=gt['time_sec'].tolist()
    slot_events=load_slot_gt_events(slot_gt_events_path) if slot_gt_events_path and Path(slot_gt_events_path).exists() else None
    variants=[]
    runners=[('SAFE_BASELINE',_run_state_engine_v13_safe,base),('TRANSITION_GUARD',_run_state_engine_v15_transition_guard,guard),('SEG_ASSIST',_run_state_engine_v15_1_seg_assist,segp)]
    for idx,(name,runner,params) in enumerate(runners):
        local_df,global_df,transitions=runner(evidence,params);ev=evaluate_state_output(global_df,gt,dev_end,warmup)
        slot_eval=evaluate_slot_level(global_df,slot_events,gt_times,dev_end,warmup) if slot_events is not None else None
        variants.append({'name':name,'params':params,'local':local_df,'global':global_df,'transitions':transitions,'ev':ev,'slot_eval':slot_eval})
        if progress:progress((idx+1)/3.0,f'v16 base temporal comparison {idx+1}/3 | {name}')
    b,g,s=variants;bd=b['ev']['DEV'];dev_confounded=bool(bd['false_empty_bias_rate']>=0.35 and bd['exact_rate']<0.60)

    def safe_vs_base(v):
        d=v['ev']['DEV']
        return bool(d['mae']<=bd['mae']+0.08 and d['exact_rate']>=bd['exact_rate']-0.04 and d['max_abs_error']<=bd['max_abs_error']+1 and d['over_rate']<=bd['over_rate']+0.02)
    def clear_gain(v):
        d=v['ev']['DEV']
        return bool(d['mae']<=bd['mae']-0.02 or d['over_rate']<bd['over_rate']-0.02 or d['exact_rate']>=bd['exact_rate']+0.03)
    def slot_gain(v):
        if b['slot_eval'] is None or v['slot_eval'] is None:return False
        return bool(v['slot_eval']['DEV']['slot_accuracy']>=b['slot_eval']['DEV']['slot_accuracy']+0.01 and v['slot_eval']['DEV']['false_empty_rate']<=b['slot_eval']['DEV']['false_empty_rate'])

    # Baseline-first. Confounded count DEV cannot promote an experimental variant unless slot-GT supports it.
    eligible=[]
    for v in [g,s]:
        if safe_vs_base(v) and ((not dev_confounded and clear_gain(v)) or slot_gain(v)):
            eligible.append(v)
    if eligible:
        chosen=max(eligible,key=lambda v:(v['slot_eval']['DEV']['slot_accuracy'] if v['slot_eval'] is not None else v['ev']['DEV']['exact_rate'], -v['ev']['DEV']['mae']))
        reason=f"{chosen['name']} selected: DEV/slot-GT safety envelope showed trustworthy improvement."
    else:
        chosen=b;reason='SAFE_BASELINE retained: DEV is confounded or no experimental variant showed a trustworthy, non-regressive improvement.'

    rows=[]
    for v in variants:
        d=v['ev']['DEV'];te=v['ev']['TEST'];row={'temporal_variant':v['name'],**v['params'],'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],'dev_under_rate':d['false_empty_bias_rate'],'dev_over_rate':d['over_rate'],'dev_max_abs_error':d['max_abs_error'],'test_N':te['N'],'test_exact_rate':te['exact_rate'],'test_MAE':te['mae'],'test_under_rate':te['false_empty_bias_rate'],'test_over_rate':te['over_rate'],'test_max_abs_error':te['max_abs_error']}
        if v['slot_eval'] is not None:row.update({'dev_slot_accuracy':v['slot_eval']['DEV']['slot_accuracy'],'dev_slot_false_empty_rate':v['slot_eval']['DEV']['false_empty_rate'],'dev_all_slots_exact_time_rate':v['slot_eval']['DEV']['all_slots_exact_time_rate']})
        rows.append(row)
    board=pd.DataFrame(rows)
    board.to_csv(out_dir/'temporal_variant_comparison.csv',index=False,encoding='utf-8-sig')
    board.to_csv(out_dir/'state_search_leaderboard.csv',index=False,encoding='utf-8-sig')
    # temporal_variant and the underlying detector parameter evidence_mode are different
    # concepts. v16 renamed temporal_variant -> evidence_mode while evidence_mode already
    # existed, producing duplicate CSV headers. Preserve both explicitly in v16.1.
    det_board=board.copy()
    if 'evidence_mode' in det_board.columns:
        det_board=det_board.rename(columns={'evidence_mode':'source_evidence_mode'})
    det_board['evidence_mode']=det_board['temporal_variant'].astype(str)
    det_board=det_board.drop(columns=['temporal_variant'])
    det_board.to_csv(out_dir/'detection_mode_comparison.csv',index=False,encoding='utf-8-sig')
    tagmap={'SAFE_BASELINE':'baseline','TRANSITION_GUARD':'transition_guard','SEG_ASSIST':'seg_assist'}
    for v in variants:
        tag=tagmap[v['name']];v['ev']['timeseries'].to_csv(out_dir/f'{tag}_count_timeseries.csv',index=False,encoding='utf-8-sig');v['global'].to_csv(out_dir/f'{tag}_global_slot_timeseries.csv',index=False,encoding='utf-8-sig');v['local'].to_csv(out_dir/f'{tag}_slot_timeseries.csv',index=False,encoding='utf-8-sig');v['transitions'].to_csv(out_dir/f'{tag}_state_transitions.csv',index=False,encoding='utf-8-sig')
    local_df=chosen['local'];global_df=chosen['global'];transitions=chosen['transitions'];ev=chosen['ev'];slot_eval=chosen['slot_eval']
    local_df.to_csv(out_dir/'selected_slot_timeseries.csv',index=False,encoding='utf-8-sig');global_df.to_csv(out_dir/'selected_global_slot_timeseries.csv',index=False,encoding='utf-8-sig');transitions.to_csv(out_dir/'selected_state_transitions.csv',index=False,encoding='utf-8-sig');ev['timeseries'].to_csv(out_dir/'selected_count_timeseries.csv',index=False,encoding='utf-8-sig')
    selected_params={**chosen['params'],'temporal_variant':chosen['name']};save_json(out_dir/'selected_state_params.json',selected_params);pd.DataFrame([{'split':sp,**ev[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'metrics_dev_test.csv',index=False,encoding='utf-8-sig');ev['timeseries'][(ev['timeseries']['split']=='TEST')&(ev['timeseries']['error']!=0)].to_csv(out_dir/'test_error_cases.csv',index=False,encoding='utf-8-sig')
    audit={'selected_temporal_variant':chosen['name'],'reason':reason,'dev_confounded':dev_confounded,'selection_uses_test':False,'baseline_first':True,
           'transition_guard_safe':safe_vs_base(g),'seg_assist_safe':safe_vs_base(s),'segmentation_never_creates_occupancy_alone':True}
    save_json(out_dir/'mode_selection_audit.json',audit)
    if slot_eval is not None:
        slot_eval['rows'].to_csv(out_dir/'slot_level_comparison.csv',index=False,encoding='utf-8-sig');slot_eval['by_slot'].to_csv(out_dir/'slot_level_by_slot.csv',index=False,encoding='utf-8-sig');slot_eval['rows'][(slot_eval['rows']['split']=='TEST')&(slot_eval['rows']['correct']==0)].to_csv(out_dir/'slot_level_test_errors.csv',index=False,encoding='utf-8-sig');pd.DataFrame([{'split':sp,**slot_eval[sp]} for sp in ['DEV','TEST']]).to_csv(out_dir/'slot_level_metrics.csv',index=False,encoding='utf-8-sig')
    with (out_dir/'REPORT.txt').open('w',encoding='utf-8') as f:
        f.write('Parking Slot State Engine v16 - baseline-safe transition fusion + empty-reference experiment\n\n')
        f.write(f'Warm-up excluded: 0 <= t < {warmup:.1f}s\nEvery decision uses only current/past frames.\n')
        f.write('Operational SEG is lazy and runs only on transition-relevant ambiguous crops. RAW and condition-specific ADAPTIVE segmentation are fused; one physical mask may support at most one local slot.\n')
        f.write('SEG alone never creates occupancy. Positive SEG can protect an established OCCUPIED state from an unsafe clear.\n')
        f.write('Ground truth is never auto-edited from detector/segmenter output.\n\nSELECTION\n'+reason+f'\ndev_confounded={dev_confounded}\nselection_uses_TEST=False\n\n')
        for v in variants:
            d=v['ev']['DEV'];te=v['ev']['TEST'];f.write(f"[{v['name']}] DEV exact={d['exact_rate']*100:.2f}% MAE={d['mae']:.4f} under={d['false_empty_bias_rate']*100:.2f}% | TEST exact={te['exact_rate']*100:.2f}% MAE={te['mae']:.4f} max={te['max_abs_error']}\n")
        f.write(f'\nSELECTED={chosen["name"]}\n')
    return {'best_params':selected_params,'evaluation':ev,'slot_evaluation':slot_eval,'leaderboard':board,'mode_comparison':board,'selection_audit':audit,'output_dir':str(out_dir)}

# =============================================================================
# v16 experimental EMPTY-reference assist
# Fixed CCTV gives a free appearance reference for slots that were manually marked
# EMPTY at startup. This variant is evaluation-only unless slot-level GT later proves
# it is safe; SAFE_BASELINE selection remains protected.
# =============================================================================

_search_state_parameters_v15_4 = search_state_parameters


def _run_state_engine_v16_empty_ref_assist(evidence: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    local,_,_=_run_state_engine_v15_1_seg_assist(evidence,params)
    if local.empty:
        return local,pd.DataFrame(),pd.DataFrame()
    ev=_v16_augment_empty_reference(evidence,{'empty_reference':params.get('empty_reference_cfg',{})})
    ev_idx={(round(float(r.time_sec),6),str(r.local_id)):r for r in ev.itertuples()}
    cfg=dict(params.get('empty_reference_cfg',{}) or {})
    entry_streak=float(cfg.get('entry_streak_sec',3.0));min_conf=float(cfg.get('vehicle_support_min_conf',0.05));protect=bool(cfg.get('protect_occupied',True))
    require_vehicle=bool(cfg.get('require_vehicle_sensor',True));sample_sec=max(0.1,float(params.get('sample_sec',1.0)))
    rows=[];trans=[]
    for lid,g in local.sort_values(['local_id','time_sec']).groupby('local_id',sort=False):
        carried=None;prev=None;entry_acc=0.0;clear_acc=0.0
        for rr in g.itertuples():
            d=rr._asdict();key=(round(float(d['time_sec']),6),str(lid));er=ev_idx.get(key)
            desired=str(d.get('state','EMPTY')).upper();init=str(getattr(er,'initial_state','')) if er is not None else ''
            eref=bool(int(getattr(er,'empty_ref_positive',0))) if er is not None else False
            margin=float(getattr(er,'empty_ref_margin',0.0)) if er is not None else 0.0
            sensor=False
            if er is not None:
                sensor=bool(
                    (int(getattr(er,'full_detected',0)) and float(getattr(er,'full_det_conf',0.0))>=min_conf) or
                    (int(getattr(er,'crop_detected',0)) and float(getattr(er,'crop_det_conf',0.0))>=min_conf) or
                    int(getattr(er,'seg_detected',0)) or int(getattr(er,'adaptive_det_detected',0))
                )
            if carried is None: carried=desired
            decision='NO_EMPTY_REF_INTERVENTION'
            if carried=='EMPTY':
                if desired=='OCCUPIED':
                    carried='OCCUPIED';entry_acc=0.0;decision='BASE_VARIANT_ENTRY'
                elif init.upper()=='EMPTY' and eref and (sensor or not require_vehicle):
                    entry_acc+=sample_sec
                    if entry_acc>=entry_streak:
                        carried='OCCUPIED';decision='ENTRY_EMPTY_REF_PLUS_SENSOR'
                    else:
                        decision='EMPTY_REF_ENTRY_ACCUMULATING'
                else:
                    entry_acc=0.0;carried='EMPTY'
            else:  # carried OCCUPIED
                if desired=='OCCUPIED':
                    carried='OCCUPIED';clear_acc=0.0
                elif init.upper()=='EMPTY' and protect and eref:
                    carried='OCCUPIED';clear_acc=0.0;decision='BLOCK_EMPTY_EMPTY_REF'
                else:
                    clear_acc+=sample_sec
                    if clear_acc>=max(2.0,sample_sec):
                        carried='EMPTY';clear_acc=0.0;decision='ALLOW_EMPTY_AFTER_REF_CLEAR'
                    else:
                        decision='EMPTY_REF_CLEAR_CONFIRMING'
            d['state']=carried;d['phase']=carried;d['evidence_mode']='EMPTY_REF_ASSIST';d['empty_ref_positive']=int(eref);d['empty_ref_margin']=margin;d['empty_ref_vehicle_support']=int(sensor);d['empty_ref_decision']=decision
            if carried=='OCCUPIED': d['occupied_score']=max(0.51,float(d.get('occupied_score',0.51)))
            rows.append(d)
            if prev is not None and prev!=carried:
                trans.append({'scope':'LOCAL','time_sec':float(d['time_sec']),'timestamp':d['timestamp'],'id':str(lid),'global_id':d['global_id'],
                              'from_state':prev,'to_state':carried,'cctv':d['cctv'],'phase':carried,'dominant_track_id':-1,
                              'evidence_mode':'EMPTY_REF_ASSIST','guard_reason':decision})
            prev=carried
    local2=pd.DataFrame(rows).sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    global2,gtrans=_rebuild_global_from_local_v15_1(local2,params,'EMPTY_REF_ASSIST')
    return local2,global2,pd.concat([pd.DataFrame(trans),gtrans],ignore_index=True) if (trans or not gtrans.empty) else pd.DataFrame()


def _repair_state_compare_columns(df: pd.DataFrame, context: str='', log_path: Optional[Path]=None) -> pd.DataFrame:
    """Guarantee unique columns while preserving legacy duplicate evidence_mode data.

    Old v16 detection_mode_comparison.csv could contain two evidence_mode headers:
    the temporal variant and the underlying source detector mode. pandas may expose
    the second as evidence_mode.1. v16.1 keeps the former as evidence_mode and
    renames the latter to source_evidence_mode. Other exact duplicates are retained
    with deterministic __dupN suffixes rather than silently discarded.
    """
    if df is None:
        return pd.DataFrame()
    out=df.copy()
    repairs=[]
    cols=list(out.columns)
    if len(set(cols)) != len(cols):
        seen={}
        new_cols=[]
        for col in cols:
            name=str(col); n=seen.get(name,0)+1; seen[name]=n
            if n==1:
                new=name
            elif context=='detection_mode_comparison.csv' and name=='evidence_mode' and 'source_evidence_mode' not in new_cols:
                new='source_evidence_mode'
            else:
                new=f'{name}__dup{n}'
            if new!=name: repairs.append(f'{name} -> {new}')
            new_cols.append(new)
        out.columns=new_cols
    # pandas auto-mangles duplicate CSV headers to evidence_mode.1. Restore its
    # intended semantic name when opening an already-corrupted v16 file.
    if context=='detection_mode_comparison.csv' and 'evidence_mode.1' in out.columns:
        if 'source_evidence_mode' not in out.columns:
            out=out.rename(columns={'evidence_mode.1':'source_evidence_mode'})
            repairs.append('evidence_mode.1 -> source_evidence_mode')
        else:
            alt='source_evidence_mode_legacy'
            out=out.rename(columns={'evidence_mode.1':alt})
            repairs.append(f'evidence_mode.1 -> {alt}')
    if out.columns.duplicated().any():
        # Absolute final guard: preserve all columns with unique positional names.
        seen={}; names=[]
        for col in out.columns:
            name=str(col); n=seen.get(name,0)+1; seen[name]=n
            new=name if n==1 else f'{name}__dup{n}'
            if new!=name: repairs.append(f'{name} -> {new}')
            names.append(new)
        out.columns=names
    if repairs and log_path is not None:
        try:
            with Path(log_path).open('a',encoding='utf-8') as f:
                f.write(f'[{context}] repaired duplicate/legacy columns: ' + '; '.join(repairs) + '\n')
        except Exception:
            pass
    return out


def _append_state_variant_row(fp: Path, row: Dict, repair_log: Optional[Path]=None) -> pd.DataFrame:
    """Append one evaluation variant to a comparison CSV without duplicate columns."""
    context=fp.name
    try:
        board=pd.read_csv(fp,encoding='utf-8-sig')
    except Exception:
        board=pd.DataFrame()
    board=_repair_state_compare_columns(board,context,repair_log)

    add_row=dict(row)
    if context=='detection_mode_comparison.csv':
        # Keep the temporal/state variant in evidence_mode for this legacy table,
        # but preserve the underlying source mode under a distinct name.
        source=add_row.pop('evidence_mode',None)
        variant=add_row.pop('temporal_variant','EMPTY_REF_ASSIST')
        if source is not None:
            add_row.setdefault('source_evidence_mode',source)
        add_row['evidence_mode']=variant
        variant_col='evidence_mode'
    else:
        variant_col='temporal_variant'
    add=_repair_state_compare_columns(pd.DataFrame([add_row]),context,repair_log)

    if not board.empty and variant_col in board.columns:
        board=board[board[variant_col].astype(str)!='EMPTY_REF_ASSIST'].copy()
    combined=pd.concat([board,add],ignore_index=True,sort=False)
    combined=_repair_state_compare_columns(combined,context,repair_log)
    combined.to_csv(fp,index=False,encoding='utf-8-sig')
    return combined


def search_state_parameters(evidence: pd.DataFrame, gt: pd.DataFrame, settings: Dict, output_dir: str,
                            progress=None, slot_gt_events_path: Optional[str]=None) -> Dict:
    """v16 baseline-first selection plus evaluation-only EMPTY-reference variant."""
    def p0(r,text):
        if progress: progress(0.80*r,text)
    result=_search_state_parameters_v15_4(evidence,gt,settings,output_dir,progress=p0 if progress else None,slot_gt_events_path=slot_gt_events_path)
    out_dir=Path(output_dir);selected=result.get('best_params',{}) or _v16_baseline_params(settings)
    empty_cfg=dict(settings.get('empty_reference',{}) or {})
    ep={**selected,'empty_reference_cfg':empty_cfg,'sample_sec':float(settings.get('evidence_sample_sec',1.0))}
    local_df,global_df,trans=_run_state_engine_v16_empty_ref_assist(evidence,ep)
    dev_end=float(settings.get('dev_end_sec',900));warmup=float(settings.get('evaluation_warmup_sec',10.0));ev=evaluate_state_output(global_df,gt,dev_end,warmup)
    slot_events=load_slot_gt_events(slot_gt_events_path) if slot_gt_events_path and Path(slot_gt_events_path).exists() else None
    slot_eval=evaluate_slot_level(global_df,slot_events,gt['time_sec'].tolist(),dev_end,warmup) if slot_events is not None else None
    d=ev['DEV'];te=ev['TEST'];row={'temporal_variant':'EMPTY_REF_ASSIST',**ep,'dev_N':d['N'],'dev_exact_rate':d['exact_rate'],'dev_MAE':d['mae'],'dev_under_rate':d['false_empty_bias_rate'],'dev_over_rate':d['over_rate'],'dev_max_abs_error':d['max_abs_error'],'test_N':te['N'],'test_exact_rate':te['exact_rate'],'test_MAE':te['mae'],'test_under_rate':te['false_empty_bias_rate'],'test_over_rate':te['over_rate'],'test_max_abs_error':te['max_abs_error']}
    if slot_eval is not None: row.update({'dev_slot_accuracy':slot_eval['DEV']['slot_accuracy'],'dev_slot_false_empty_rate':slot_eval['DEV']['false_empty_rate'],'dev_all_slots_exact_time_rate':slot_eval['DEV']['all_slots_exact_time_rate']})
    repair_log=out_dir/'STATE_SEARCH_REPAIR.log'
    for fn in ['temporal_variant_comparison.csv','state_search_leaderboard.csv','detection_mode_comparison.csv']:
        _append_state_variant_row(out_dir/fn,row,repair_log=repair_log)
    local_df.to_csv(out_dir/'empty_ref_assist_slot_timeseries.csv',index=False,encoding='utf-8-sig');global_df.to_csv(out_dir/'empty_ref_assist_global_slot_timeseries.csv',index=False,encoding='utf-8-sig');trans.to_csv(out_dir/'empty_ref_assist_state_transitions.csv',index=False,encoding='utf-8-sig')
    audit_path=out_dir/'mode_selection_audit.json';audit=load_json(audit_path,{}) or {};audit['empty_ref_assist_evaluation_only']=True;audit['empty_ref_assist_not_auto_selected_without_slot_gt']=True;save_json(audit_path,audit)
    with (out_dir/'EMPTY_REF_EXPERIMENT.txt').open('w',encoding='utf-8') as f:
        f.write('v16 EMPTY-reference assist is experimental/evaluation-only.\n')
        f.write('It uses startup EMPTY appearance + an independent vehicle sensor, never appearance alone, to propose occupancy.\n')
        f.write(f"DEV exact={d['exact_rate']*100:.2f}% MAE={d['mae']:.4f} | TEST exact={te['exact_rate']*100:.2f}% MAE={te['mae']:.4f}\n")
    result['empty_ref_evaluation']={'evaluation':ev,'slot_evaluation':slot_eval,'params':ep}
    if progress: progress(1.0,'v16 comparison complete | EMPTY_REF_ASSIST evaluated without changing baseline-safe selection')
    return result
