"""Run fresh GPU restart extraction with explicitly unverified frame-only pseudo labels."""
import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO
from manual_initialization import write_snapshot
from restart_evaluation import run_restart_experiments
from utils import load_ground_truth


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pseudo_snapshot(evidence, ids, start, threshold=.25):
    """Use only detection at start. A non-detection is not an EMPTY label."""
    first = evidence[evidence.time_sec == start]
    if first.empty:
        raise ValueError('No archived observations at exact snapshot time')
    positive = {str(x) for x in first.loc[(first.full_detected == 1) & (first.full_det_conf >= threshold), 'global_id']}
    if not positive.issubset(ids):
        raise ValueError('Evidence contains unknown global slot IDs')
    return {gid: ('OCCUPIED' if gid in positive else 'UNKNOWN') for gid in ids}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--gt', required=True, type=Path)
    p.add_argument('--fit', required=True, type=Path)
    p.add_argument('--app', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--start', type=int, choices=[0, 330, 660, 900], default=900)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this experiment')
    args.output.mkdir(parents=True, exist_ok=False)
    fit = args.output / 'fit_copy'
    shutil.copytree(args.fit, fit, ignore=shutil.ignore_patterns('restart_experiments'))
    settings = json.loads((fit / 'settings_snapshot.json').read_text(encoding='utf-8'))
    request = json.loads((fit / 'worker_input.json').read_text(encoding='utf-8'))
    starts = [args.start]
    archive = sorted((args.fit / 'restart_experiments').glob('run_*'))[-1]
    if json.loads((archive / 'restart_summary.json').read_text(encoding='utf-8'))['status'] != 'completed':
        raise ValueError('Saved evidence run incomplete')
    evidence_path = archive / f'restart_{args.start}' / 'evidence' / 'slot_evidence.csv'
    # This bootstrap uses only archived detections made at the exact start.
    # A miss is UNKNOWN, never an assumed empty space or a human label.
    evidence = pd.read_csv(evidence_path)
    ids = {str(s['global_id']) for s in json.loads((fit / 'slots.json').read_text(encoding='utf-8'))['slots']}
    states = pseudo_snapshot(evidence, ids, args.start)
    positive = {gid for gid, value in states.items() if value == 'OCCUPIED'}
    snapshot = args.output / f'pseudo_detection_{args.start}.csv'
    write_snapshot(snapshot, args.start, states)
    settings['validation']['restart_experiments'] = {'enabled': True, 'starts_sec': starts, 'duration_sec': 330,
        'stable_samples': 3, 'manual_init': {'enabled': True, 'path': str(snapshot), 'source': 'pseudo_archived_frame_detection_unverified'}}
    manifest = {'status': 'running', 'start_sec': args.start, 'split': settings.get('active_split_protocol', {}).get('name'), 'snapshot_source': 'unverified pseudo-label from archived detection at exact start time; not human GT',
        'pseudo_rule': 'full_detected and confidence >= 0.25 => OCCUPIED; all other slots UNKNOWN',
        'video_sha256': digest(args.video), 'gt_sha256': digest(args.gt), 'snapshot_sha256': digest(snapshot),
        'archived_evidence_sha256': digest(evidence_path), 'selected_fit_sha256': digest(fit / 'selected_state_params.json'),
        'weights_sha256': {n: digest(args.app / n) for n in ('yolov8m.pt', 'yolov8l.pt', 'yolov8m-seg.pt')},
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'scope': 'historically reviewed same-video restart; new GPU detections; pseudo-init is not human verified'}
    source_root = Path(__file__).resolve().parent
    manifest['code_sha256'] = {name: digest(source_root / name) for name in
        ('fresh_manual_experiment.py', 'restart_evaluation.py', 'slot_engine.py', 'manual_initialization.py', 'detector.py', 'segmentation.py')}
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    original_predict = YOLO.predict
    counter = {'gpu_calls': 0}
    def checked_predict(self, *a, **kw):
        results = original_predict(self, *a, **kw)
        if not getattr(self, 'predictor', None) or self.predictor.device.type != 'cuda':
            raise RuntimeError('Inference did not use CUDA')
        counter['gpu_calls'] += 1
        return results
    YOLO.predict = checked_predict
    try:
        os.chdir(args.app)
        print('GPU:', manifest['gpu'], 'pseudo occupied:', len(positive), 'unknown:', len(ids-positive), flush=True)
        result = run_restart_experiments(str(args.video), request['rois'], load_ground_truth(str(args.gt)), settings, fit,
                                         lambda fraction, msg: print(f'{fraction:.1%} {msg}', flush=True))
        manifest.update(status='completed', result=str(result), gpu_prediction_calls=counter['gpu_calls'])
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc), gpu_prediction_calls=counter['gpu_calls'])
        raise
    finally:
        YOLO.predict = original_predict
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
