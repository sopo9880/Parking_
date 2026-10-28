"""Bounded, offline replay of frozen restart observations; never changes production SAFE."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from restart_evaluation import DEFAULT_RECOVERY, recovery_metrics
from slot_engine import _run_state_engine_v13_safe, _v16_baseline_params, evaluate_state_output
from utils import load_ground_truth


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select_dev(rows, split, variants):
    dev_rows = pd.DataFrame([r for r in rows if r['split'] == split and r['partition'] == 'DEV'])
    if dev_rows.empty or set(dev_rows.variant) != set(variants):
        raise ValueError('Incomplete DEV candidate results')
    ranked = dev_rows.groupby('variant', sort=False).agg(exact=('exact_rate', 'mean'), mae=('mae', 'mean')).reset_index()
    ranked['tie_order'] = ranked.variant.map({name: i for i, name in enumerate(variants)})
    ranked = ranked.sort_values(['exact', 'mae', 'tie_order'], ascending=[False, True, True])
    return {'split': split, 'selected_on_dev': ranked.iloc[0].variant, 'ranking': ranked.to_dict('records')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path, help='completed split_experiments directory')
    parser.add_argument('--gt', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new empty results directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # These alternatives are fixed before scoring; TEST never selects a setting.
    variants = {
        'SAFE': None,
        'RECOVERY_DEFAULT': dict(DEFAULT_RECOVERY),
        'RECOVERY_5S': {**DEFAULT_RECOVERY, 'window_sec': 5, 'min_observation_sec': 5, 'min_samples': 5},
        'RECOVERY_20S': {**DEFAULT_RECOVERY, 'duration_sec': 60, 'window_sec': 20, 'min_observation_sec': 20, 'min_samples': 16},
    }
    manifest = {'created': datetime.now().isoformat(), 'hypothesis': 'Startup evidence window changes stationary occupancy recovery',
                'variants': variants, 'gt_sha256': sha(args.gt), 'inputs': {},
                'scope': 'historically reviewed same-video count evaluation; frozen saved observations; not independent validation',
                'selection': 'DEV mean exact descending, MAE ascending; fixed tie order', 'status': 'running'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    gt = load_ground_truth(str(args.gt))
    rows = []
    try:
        for split, dev, test in [('original', [0, 330], [900]), ('alternate', [330, 660, 900], [0])]:
            fit = args.source / split
            settings_path = fit / 'settings_snapshot.json'
            params_path = fit / 'selected_state_params.json'
            settings = json.loads(settings_path.read_text(encoding='utf-8'))
            freeze = json.loads((fit / 'selection_freeze.json').read_text(encoding='utf-8'))
            if sha(params_path) != freeze['sha256']:
                raise ValueError(f'{split}: fitted parameter hash mismatch')
            runs = sorted((fit / 'restart_experiments').glob('run_*'))
            if not runs:
                raise FileNotFoundError(f'{split}: no saved restart run')
            source = runs[-1]
            if json.loads((source / 'restart_summary.json').read_text(encoding='utf-8'))['status'] != 'completed':
                raise ValueError(f'{split}: latest restart run incomplete')
            for start in dev + test:
                if start in test:
                    choice = select_dev(rows, split, variants)
                    manifest.setdefault('choices', []).append(choice)
                    (args.output / f'{split}_selection_freeze.json').write_text(json.dumps(choice, indent=2), encoding='utf-8')
                    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                evidence_path = source / f'restart_{start}' / 'evidence' / 'slot_evidence.csv'
                evidence = pd.read_csv(evidence_path)
                if evidence.empty or float(evidence.time_sec.min()) != start:
                    raise ValueError(f'{split} {start}: invalid restart evidence')
                manifest['inputs'][f'{split}/{start}'] = {'evidence_sha256': sha(evidence_path), 'settings_sha256': sha(settings_path), 'fit_sha256': sha(params_path)}
                end = min(float(settings.get('eval_end_sec', 1230)), start + 330)
                # Strictly keep original DEV evaluation inside DEV, avoiding TEST label leakage.
                if split == 'original' and start in dev:
                    end = min(end, 900)
                part = gt[(gt.time_sec >= start) & ((gt.time_sec <= end) if end == 1230 else (gt.time_sec < end))]
                if part.empty or not set(part.time_sec).issubset(set(evidence.time_sec)):
                    raise ValueError(f'{split} {start}: missing GT checkpoints')
                for name, candidate in variants.items():
                    params = {**_v16_baseline_params(settings), 'unlabeled_restart': True}
                    if candidate is not None:
                        params['startup_recovery'] = candidate
                    _, glob, _ = _run_state_engine_v13_safe(evidence, params)
                    trace = evaluate_state_output(glob, part, 0, 0)['timeseries']
                    metrics = recovery_metrics(trace, start)
                    rows.append({'split': split, 'partition': 'DEV' if start in dev else 'TEST', 'start_sec': start, 'variant': name, **metrics})
                    pd.DataFrame(rows).to_csv(args.output / 'metrics_partial.csv', index=False)
                    print(f'{split} {start} {name}: exact={metrics["exact_rate"]:.3f} mae={metrics["mae"]:.3f}', flush=True)
        table = pd.DataFrame(rows)
        table.to_csv(args.output / 'metrics.csv', index=False)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        raise
    finally:
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
