"""Inspect whether the default startup gate overlaps already occupied SAFE slots."""
import argparse
import json
from pathlib import Path
import pandas as pd
from restart_evaluation import DEFAULT_RECOVERY, recovery_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    rows = []
    for split, starts in [('original', [0, 330, 900]), ('alternate', [0, 330, 660, 900])]:
        run = sorted((args.source / split / 'restart_experiments').glob('run_*'))[-1]
        if json.loads((run / 'restart_summary.json').read_text(encoding='utf-8'))['status'] != 'completed':
            raise ValueError('Restart run is incomplete')
        for start in starts:
            root = run / f'restart_{start}'
            ev = pd.read_csv(root / 'evidence' / 'slot_evidence.csv')
            t = start + float(DEFAULT_RECOVERY['min_observation_sec'])
            local = pd.read_csv(root / 'RESET_SAFE_slot_timeseries.csv')
            states = local[local.time_sec == t].set_index('local_id')['state']
            ready = [lid for lid, group in ev[(ev.time_sec >= start) & (ev.time_sec <= t)].groupby('local_id')
                     if recovery_gate(group.to_dict('records'), t - start, DEFAULT_RECOVERY)]
            if not set(ready).issubset(states.index):
                raise ValueError('SAFE local states missing at checkpoint')
            occupied = sum(states.loc[lid] == 'OCCUPIED' for lid in ready)
            rows.append({'split': split, 'restart_sec': start, 'checkpoint_sec': t,
                         'ready_local_slots': len(ready), 'already_safe_occupied': occupied,
                         'ready_but_not_occupied': len(ready) - occupied})
    if args.output.exists():
        raise FileExistsError(args.output)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
