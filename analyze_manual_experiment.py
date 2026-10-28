"""Summarize restart count traces while keeping UNKNOWN separate from EMPTY."""
import argparse
import json
from pathlib import Path
import pandas as pd


def score(trace, manual):
    result = {'N': len(trace)}
    if manual:
        result['unknown_mean'] = float(trace.unknown_slots.mean()) if len(trace) else None
        result['fully_resolved_N'] = int(trace.unknown_slots.eq(0).sum())
        result['gt_within_bounds_rate'] = float(trace.gt_within_bounds.mean()) if len(trace) else None
        # Ordinary exact/MAE require every checkpoint to be fully resolved.
        if trace.unknown_slots.gt(0).any():
            result.update(exact=None, mae=None, under=None, over=None)
            return result
    error = trace['error']
    result.update(exact=float(error.eq(0).mean()) if len(trace) else None,
                  mae=float(error.abs().mean()) if len(trace) else None,
                  under=float(error.lt(0).mean()) if len(trace) else None,
                  over=float(error.gt(0).mean()) if len(trace) else None)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--restart', required=True, type=Path, help='completed restart run directory')
    p.add_argument('--start', type=int, default=900)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if json.loads((args.restart/'restart_summary.json').read_text(encoding='utf-8'))['status'] != 'completed':
        raise ValueError('Restart run incomplete')
    folder = args.restart / f'restart_{args.start}'
    variants = ['RESET_SAFE', 'RESET_RECOVERY', 'RESET_MANUAL_INIT', 'CONTINUOUS_SAFE', 'CONTINUOUS_CANDIDATE']
    rows = []
    for name in variants:
        manual = name == 'RESET_MANUAL_INIT'
        path = folder / (f'{name}_occupancy_bounds.csv' if manual else f'{name}_count_timeseries.csv')
        if not path.exists():
            rows.append({'variant': name, 'scope': 'MISSING', 'N': 0}); continue
        d = pd.read_csv(path)
        for scope, part in [('initial', d[d.time_sec == args.start]), ('after_initial', d[d.time_sec > args.start]),
                            ('first_30s', d[(d.time_sec > args.start) & (d.time_sec < args.start+30)]),
                            ('first_60s', d[(d.time_sec > args.start) & (d.time_sec < args.start+60)]),
                            ('first_120s', d[(d.time_sec > args.start) & (d.time_sec < args.start+120)])]:
            rows.append({'variant': name, 'scope': scope, **score(part, manual)})
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == '__main__': main()
