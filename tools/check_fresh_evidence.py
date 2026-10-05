"""Recalculate archived v16.5.4 fresh-split counts, without new inference claims."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'paper/data/fresh_v1654'
provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
for artifact in provenance['artifacts']:
    assert hashlib.sha256((ROOT/artifact['path']).read_bytes()).hexdigest()==artifact['sha256']
for protocol in ('original','alternate'):
    folder=root/protocol
    freeze=json.loads((folder/'selection_freeze.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((folder/'selected_state_params.json').read_bytes()).hexdigest()==freeze['sha256']
    params=json.loads((folder/'selected_state_params.json').read_text(encoding='utf-8'))
    tags={'SAFE_BASELINE':'baseline','TRANSITION_GUARD':'transition_guard','SEG_ASSIST':'seg_assist',
          'EMPTY_REF':'empty_ref_assist','CANDIDATE':'candidate_v164'}
    tags['SELECTED']=tags[params['temporal_variant']]
    board=pd.read_csv(folder/'split_final_metrics.csv')
    for record in board.to_dict('records'):
        trace=pd.read_csv(folder/f"{tags[record['variant']]}_count_timeseries.csv")
        trace=trace[trace.split==record['split']]
        error=trace.candidate_error if record['variant']=='CANDIDATE' else trace.error
        computed={'N':len(error),'exact_rate':float((error==0).mean()),'mae':float(error.abs().mean()),
                  'max_abs_error':int(error.abs().max()),'under_rate':float((error<0).mean()),'over_rate':float((error>0).mean())}
        for key,value in computed.items(): assert np.isclose(value,record[key]),(protocol,record['variant'],key)
print('Archived fresh split evidence PASS: all DEV/TEST counts and frozen selection hashes; restart recovery scores remain unmeasured')
