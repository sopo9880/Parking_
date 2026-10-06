"""DEV-only research selection, separate from protected operational selection."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

TAGS={'SAFE_BASELINE':'baseline','TRANSITION_GUARD':'transition_guard','SEG_ASSIST':'seg_assist'}


def temporal_candidates(board):
    # Legacy EMPTY_REF rows use temporal_variant=SAFE_BASELINE too. They are a
    # distinct experiment, excluded from this predefined three-variant selection.
    return board[board.empty_reference_cfg.isna()] if 'empty_reference_cfg' in board else board


def choose_dev_variant(board):
    # Fixed ranking before TEST: Exact descending, MAE/over/max ascending.
    # Stable ties prefer SAFE, then the simpler non-segmentation guard.
    board=temporal_candidates(board)
    candidates=[]
    for priority,name in enumerate(TAGS):
        matches=board[board.temporal_variant==name]
        if len(matches)!=1: raise ValueError('Expected one DEV candidate per temporal variant')
        row=matches.iloc[0]
        values=[row[k] for k in ('dev_N','dev_exact_rate','dev_MAE','dev_over_rate','dev_max_abs_error')]
        if not np.isfinite(values).all() or row.dev_N<=0: raise ValueError('Invalid DEV selection metrics')
        candidates.append(((-row.dev_exact_rate,row.dev_MAE,row.dev_over_rate,row.dev_max_abs_error,priority),name))
    return min(candidates)[1]


def load_variant_params(folder,variant):
    board=temporal_candidates(pd.read_csv(Path(folder)/'DEV_selection_variant_comparison.csv'))
    matches=board[board.temporal_variant==variant]
    if len(matches)!=1: raise ValueError('Missing frozen DEV variant configuration')
    return {k:v for k,v in matches.iloc[0].to_dict().items()
            if not k.startswith(('dev_','test_')) and k!='empty_reference_cfg' and pd.notna(v)}


def freeze_dev_selection(folder):
    folder=Path(folder)
    board=pd.read_csv(folder/'DEV_selection_variant_comparison.csv')
    variant=choose_dev_variant(board)
    params=load_variant_params(folder,variant)
    path=folder/'dev_selected_experimental_params.json'
    path.write_text(json.dumps(params,indent=2,sort_keys=True),encoding='utf-8')
    record={'variant':variant,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'selection_uses_test':False,'production_promoted':False,
            'ranking':'DEV Exact descending; MAE, over-rate, max error ascending; SAFE/Guard/SEG tie order',
            'eligible_variants':list(TAGS),'scope':'historically reviewed same-video research selection; continuous DEV state'}
    (folder/'dev_selection_freeze.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    for label,tag in [('production_safe_selected','baseline'),('dev_selected_experimental',TAGS[variant])]:
        for suffix in ('slot_timeseries.csv','global_slot_timeseries.csv'):
            shutil.copyfile(folder/f'{tag}_{suffix}',folder/f'{label}_{suffix}')
    return record
