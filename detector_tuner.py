# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from detector import VehicleDetector, physical_vehicle_nms
from utils import crop_roi, ensure_dir, format_timestamp, load_ground_truth, open_video, read_frame_at, save_json


def _metrics(errors: Sequence[int]) -> Dict[str, float | int]:
    a = np.asarray(list(errors), dtype=np.float32)
    if a.size == 0:
        return {
            'N': 0, 'exact_rate': 0.0, 'MAE': 0.0, 'signed_error': 0.0,
            'under_rate': 0.0, 'over_rate': 0.0, 'max_abs_error': 0,
        }
    return {
        'N': int(a.size),
        'exact_rate': float(np.mean(a == 0)),
        'MAE': float(np.mean(np.abs(a))),
        'signed_error': float(np.mean(a)),
        'under_rate': float(np.mean(a < 0)),
        'over_rate': float(np.mean(a > 0)),
        'max_abs_error': int(np.max(np.abs(a))),
    }


def _gt_column_for_cctv(cctv: str) -> str:
    return f'{str(cctv).lower()}_count'


def _base_detector_cfg(settings: Dict, cctv: str) -> Dict:
    base = dict(settings.get('detector', {}))
    base.update(settings.get('detector_by_cctv', {}).get(cctv, {}))
    return base


def run_warped_detector_sweep(
    video_path: str,
    rois: Dict,
    gt_csv_path: str,
    settings: Dict,
    output_dir: str,
    progress=None,
) -> Dict:
    """Tune detector count settings on perspective-warped CCTV images using DEV only.

    The expensive dimensions are model / imgsz / tiling. For each such inference pass,
    the model runs once at the minimum confidence threshold. Higher confidence settings
    are evaluated by filtering those detections, then applying physical-vehicle NMS.
    This keeps the sweep exhaustive over the configured grid without needlessly repeating
    identical model inference for every confidence value.
    """
    out_dir = ensure_dir(output_dir)
    gt = load_ground_truth(gt_csv_path)
    dev_end = float(settings.get('dev_end_sec', 900))
    dev_gt = gt[gt['time_sec'] < dev_end].copy().reset_index(drop=True)
    if dev_gt.empty:
        raise ValueError('No DEV ground-truth rows were found before dev_end_sec.')

    sweep = settings.get('detector_sweep', {})
    models = list(sweep.get('models', ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']))
    imgszs = [int(x) for x in sweep.get('imgsz', [640, 960, 1280])]
    confs = sorted(set(float(x) for x in sweep.get('conf', [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])))
    tilings = [bool(x) for x in sweep.get('tiling', [False, True])]
    if not models or not imgszs or not confs or not tilings:
        raise ValueError('detector_sweep grid is empty.')
    min_conf = min(confs)

    # Cache only the DEV warped images once. This makes model sweeps much faster than
    # repeatedly seeking and perspective-warping the source video for each candidate.
    frames_by_cctv: Dict[str, List[np.ndarray]] = {c: [] for c in rois}
    cap = open_video(video_path)
    try:
        for i, r in dev_gt.iterrows():
            t = float(r['time_sec'])
            frame = read_frame_at(cap, t)
            for cctv, roi in rois.items():
                frames_by_cctv[cctv].append(crop_roi(frame, roi))
            if progress and (i % max(1, len(dev_gt)//10) == 0 or i == len(dev_gt)-1):
                progress((i+1)/max(1, len(dev_gt)) * 0.08, f'Caching warped DEV frames {format_timestamp(t)}')
    finally:
        cap.release()

    rows: List[Dict] = []
    best_rows: List[Dict] = []
    cctvs = [c for c in rois if _gt_column_for_cctv(c) in dev_gt.columns]
    if not cctvs:
        raise ValueError('Ground Truth CSV needs cctv1_count / cctv2_count / cctv3_count columns for detector tuning.')

    total_base = len(cctvs) * len(models) * len(imgszs) * len(tilings)
    done_base = 0
    for cctv in cctvs:
        gt_col = _gt_column_for_cctv(cctv)
        gt_counts = dev_gt[gt_col].astype(int).tolist()
        images = frames_by_cctv[cctv]
        base_cfg = _base_detector_cfg(settings, cctv)
        base_cfg['conf'] = min_conf

        for model_name in models:
            for imgsz in imgszs:
                for tiling in tilings:
                    cfg = dict(base_cfg)
                    cfg.update({'model': model_name, 'imgsz': int(imgsz), 'conf': float(min_conf), 'tiling': bool(tiling)})
                    detector = VehicleDetector(cfg)
                    raw_per_frame = []
                    for img in images:
                        # detect_raw includes tile merge NMS but intentionally delays
                        # cross-class physical dedup until after confidence filtering.
                        raw_per_frame.append(detector.detect_raw(img))

                    for conf in confs:
                        preds: List[int] = []
                        for raw in raw_per_frame:
                            filt = [d for d in raw if float(d.conf) >= conf]
                            clean = physical_vehicle_nms(
                                filt,
                                float(cfg.get('physical_vehicle_nms_iou', 0.42)),
                                float(cfg.get('physical_vehicle_overlap_min', 0.68)),
                            )
                            preds.append(len(clean))
                        errors = [p-g for p, g in zip(preds, gt_counts)]
                        m = _metrics(errors)
                        rows.append({
                            'cctv': cctv,
                            'model': model_name,
                            'imgsz': int(imgsz),
                            'conf': float(conf),
                            'tiling': bool(tiling),
                            **m,
                        })

                    done_base += 1
                    if progress:
                        progress(
                            0.08 + 0.86 * done_base/max(1, total_base),
                            f'Warped detector sweep {done_base}/{total_base}: {cctv} {model_name} {imgsz} tile={tiling}'
                        )
                    # Release model memory between candidates.
                    try:
                        detector._model = None
                    except Exception:
                        pass
                    del detector, raw_per_frame
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

    board = pd.DataFrame(rows)
    # v10 commercial-oriented selector: do not let a high exact-match rate hide
    # a detector with materially worse average or worst-case count error.
    # Primary objective is MAE, then under-count rate (false-empty risk), then
    # maximum error, then exact rate, then over-count rate.
    board = board.sort_values(
        ['cctv', 'MAE', 'under_rate', 'max_abs_error', 'exact_rate', 'over_rate'],
        ascending=[True, True, True, True, False, True]
    ).reset_index(drop=True)
    board.to_csv(out_dir/'warped_detector_sweep.csv', index=False, encoding='utf-8-sig')

    selected: Dict[str, Dict] = {}
    for cctv in cctvs:
        sub = board[board['cctv'] == cctv]
        if sub.empty:
            continue
        b = sub.iloc[0]
        base_cfg = _base_detector_cfg(settings, cctv)
        chosen = dict(base_cfg)
        chosen.update({
            'model': str(b['model']),
            'imgsz': int(b['imgsz']),
            'conf': float(b['conf']),
            'tiling': bool(b['tiling']),
        })
        selected[cctv] = chosen
        best_rows.append({
            'cctv': cctv,
            'model': chosen['model'],
            'imgsz': chosen['imgsz'],
            'conf': chosen['conf'],
            'tiling': chosen['tiling'],
            'DEV_exact_rate': float(b['exact_rate']),
            'DEV_MAE': float(b['MAE']),
            'DEV_under_rate': float(b['under_rate']),
            'DEV_over_rate': float(b['over_rate']),
            'DEV_max_abs_error': int(b['max_abs_error']),
        })
    pd.DataFrame(best_rows).to_csv(out_dir/'warped_detector_best.csv', index=False, encoding='utf-8-sig')
    save_json(out_dir/'warped_detector_selected.json', selected)

    # Apply selected configurations to the live settings object. The caller may save it.
    if bool(sweep.get('apply_best_to_settings', True)):
        settings.setdefault('detector_by_cctv', {}).update(selected)

    with (out_dir/'WARPED_DETECTOR_REPORT.txt').open('w', encoding='utf-8') as f:
        f.write('Parking Slot Engine v12 - Perspective-warped FULL-CCTV detector DEV sweep\n\n')
        f.write(f'DEV rows: {len(dev_gt)} (time < {dev_end:.1f}s)\n')
        f.write(f'Grid: models={models}, imgsz={imgszs}, conf={confs}, tiling={tilings}\n')
        f.write('TEST is not used by this tuner.\n')
        f.write('Selection policy: MAE -> under-count rate -> max absolute error -> exact rate -> over-count rate.\n\n')
        for r in best_rows:
            f.write(
                f"[{r['cctv']}] {r['model']} imgsz={r['imgsz']} conf={r['conf']:.2f} tile={r['tiling']} | "
                f"exact={r['DEV_exact_rate']*100:.2f}% MAE={r['DEV_MAE']:.4f} "
                f"under={r['DEV_under_rate']*100:.2f}% over={r['DEV_over_rate']*100:.2f}% max={r['DEV_max_abs_error']}\n"
            )

    if progress:
        progress(1.0, 'Warped detector DEV sweep complete.')
    return {
        'leaderboard': board,
        'selected': selected,
        'best': pd.DataFrame(best_rows),
        'output_dir': str(out_dir),
    }
