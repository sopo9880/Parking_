"""Check frozen v16.5.5 restart evidence; this does not run new video inference."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from restart_evaluation import recovery_metrics
from experimental_selection import choose_dev_variant

root=ROOT/'paper/data/restart_v1655'
provenance=json.loads((root/'provenance.json').read_text())
for artifact in provenance['artifacts']:
    assert hashlib.sha256((ROOT/artifact['path']).read_bytes()).hexdigest()==artifact['sha256']
recomputed=pd.read_csv(root/'recomputed_restart_metrics.csv')
for protocol in ('original','alternate'):
    folder=root/protocol
    freeze=json.loads((folder/'selection_freeze.json').read_text())
    assert hashlib.sha256((folder/'selected_state_params.json').read_bytes()).hexdigest()==freeze['sha256']
    assert choose_dev_variant(pd.read_csv(folder/'DEV_selection_variant_comparison.csv'))=='TRANSITION_GUARD'
    records=pd.read_csv(folder/'restart_metrics.csv')
    for record in records.to_dict('records'):
        start=record['restart_sec']
        trace=pd.read_csv(folder/f"restart_{start:g}/{record['variant']}_count_timeseries.csv")
        assert np.allclose(trace.error,trace.occupied_pred-trace.ground_truth_occupied_space_count)
        if record['scope']=='AFTER_RESTART_WARMUP':trace=trace[trace.time_sec>=start+10]
        measured=recovery_metrics(trace,start)
        for key,value in measured.items():
            if key not in record:continue
            if value is None:assert pd.isna(record[key]),(protocol,key)
            else:assert np.isclose(value,record[key]),(protocol,start,record['variant'],key)
        if record['scope']=='INCLUSIVE_STARTUP':
            row=recomputed[(recomputed.protocol==protocol)&(recomputed.restart_sec==start)&(recomputed.variant==record['variant'])].iloc[0]
            for key,value in measured.items():
                if value is None:assert pd.isna(row[key]),key
                else:assert np.isclose(value,row[key]),key
print('Archived v16.5.5 restart evidence PASS: all 80 scoped rows and 40 inclusive rows recomputed; no v16.5.6 full inference claim')
