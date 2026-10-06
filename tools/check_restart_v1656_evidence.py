"""Independently recompute archived v16.5.6 numerical results and recovery equality."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from restart_evaluation import recovery_metrics
root=ROOT/'paper/data/restart_v1656'
metrics=pd.read_csv(root/'recomputed_metrics.csv');identities=pd.read_csv(root/'recovery_identity.csv')
assert len(metrics)==144 and len(identities)==8
for row in metrics.to_dict('records'):
    folder=root/row['fit']/f"restart_{row['restart_sec']:g}"
    trace=pd.read_csv(folder/f"{row['variant']}_count_timeseries.csv")
    assert np.allclose(trace.error,trace.occupied_pred-trace.ground_truth_occupied_space_count)
    if row['scope']=='AFTER_RESTART_WARMUP': trace=trace[trace.time_sec>=row['restart_sec']+10]
    calculated=recovery_metrics(trace,row['restart_sec'])
    for key,value in calculated.items():
        if value is None: assert pd.isna(row[key]),key
        else: assert np.isclose(value,row[key]),(row['fit'],row['restart_sec'],row['variant'],key)
for row in identities.to_dict('records'):
    folder=root/row['fit']/f"restart_{row['restart_sec']:g}"
    for suffix in ('count_timeseries.csv','state_evidence.csv','transitions.csv'):
        assert (folder/f'RESET_SAFE_{suffix}').read_bytes()==(folder/f'RESET_RECOVERY_{suffix}').read_bytes(),(folder,suffix)
    assert row['same_count_trace'] and row['same_state_phase_score'] and row['same_transitions']
print('Actual v16.5.6 evidence PASS: 144 scoped rows recomputed; all 8 SAFE/recovery pairs identical; manual results remain unmeasured')
