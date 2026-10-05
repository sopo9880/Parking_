"""Evaluation-only comparisons of frozen SAFE/CANDIDATE count traces."""
from pathlib import Path
import hashlib
import json
import math
import re
import zipfile
import numpy as np
import pandas as pd

DEFAULT_WINDOWS = [(0,330),(330,660),(660,900),(900,1230)]

def seconds(text):
    parts=str(text).strip().split(':')
    if len(parts)>3: raise ValueError('Use seconds, MM:SS or HH:MM:SS')
    values=[float(p) for p in parts]
    if any(not math.isfinite(v) or v<0 for v in values): raise ValueError('Time must be finite and nonnegative')
    if len(values)>1 and any(v>=60 for v in values[1:]): raise ValueError('Minutes/seconds must be below 60')
    return sum(v*60**n for n,v in enumerate(reversed(values)))

def parse_windows(text):
    result=[]
    for n,item in enumerate(re.split(r'[;\n]+',str(text).strip())):
        if not item.strip(): continue
        pair=re.split(r'\s*[-–]\s*',item.strip())
        if len(pair)!=2: raise ValueError('Use 0:00-5:30; 5:30-11:00')
        start,end=map(seconds,pair)
        if end<=start: raise ValueError('Window end must be after start')
        if any(w['start_sec']==start and w['end_sec']==end for w in result): raise ValueError('Duplicate evaluation window')
        result.append({'name':f'W{n+1}','start_sec':start,'end_sec':end})
    if not result: raise ValueError('At least one evaluation window is required')
    return result

def default_windows():
    return [{'name':f'W{i+1}','start_sec':a,'end_sec':b} for i,(a,b) in enumerate(DEFAULT_WINDOWS)]

def format_windows(windows):
    def fmt(t):
        t=float(t);return f'{int(t//60)}:{t%60:05.2f}'.rstrip('0').rstrip('.')
    return '; '.join(f"{fmt(w['start_sec'])}-{fmt(w['end_sec'])}" for w in windows)

def fingerprint(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read(path,columns):
    frame=pd.read_csv(path)
    if not set(columns).issubset(frame): raise ValueError(f'Missing required columns in {path.name}')
    frame['time_sec']=pd.to_numeric(frame.time_sec,errors='raise')
    if not np.isfinite(frame.time_sec).all() or (frame.time_sec<0).any() or frame.time_sec.duplicated().any():
        raise ValueError('Trace timestamps must be unique, finite and nonnegative')
    return frame.sort_values('time_sec')

def _metrics(pred,gt):
    if len(gt)==0: return {k:None for k in ('exact_rate','MAE','max_abs_error','over_rate','under_rate')}
    err=pred.to_numpy()-gt.to_numpy(); ae=np.abs(err)
    return {'exact_rate':float(np.mean(ae==0)),'MAE':float(np.mean(ae)),
            'max_abs_error':float(ae.max()),'over_rate':float(np.mean(err>0)),'under_rate':float(np.mean(err<0))}

def write_window_evaluation(run_dir,settings):
    """Never infer, select parameters, tune, or overwrite source traces."""
    root=Path(run_dir)
    paths=[root/'baseline_count_timeseries.csv',root/'candidate_v164_count_timeseries.csv']
    if not all(p.is_file() for p in paths): raise FileNotFoundError('Both frozen SAFE and Candidate count traces are required')
    cfg=settings.get('validation',{}) or {}
    windows=cfg.get('evaluation_windows',default_windows())
    # Validate saved settings through the same parser as the form; retain user names.
    checked=[]
    for n,original in enumerate(windows):
        start,end=float(original['start_sec']),float(original['end_sec'])
        if not math.isfinite(start) or not math.isfinite(end) or start<0 or end<=start: raise ValueError('Invalid evaluation interval')
        if any(w['start_sec']==start and w['end_sec']==end for w in checked): raise ValueError('Duplicate evaluation window')
        checked.append({'name':str(original.get('name',f'W{n+1}')),'start_sec':start,'end_sec':end})
    if not checked: raise ValueError('At least one evaluation window is required')
    windows=checked
    if len(set(w['name'] for w in windows))!=len(windows) or any(w['name']=='ALL' for w in windows): raise ValueError('Window names must be unique and not ALL')
    sources={p.name:fingerprint(p) for p in paths}
    for name in ('candidate_v164_config.json','selected_state_params.json','settings.json','settings_snapshot.json'):
        if (root/name).is_file(): sources[name]=fingerprint(root/name)
    base=_read(paths[0],['time_sec','occupied_pred','ground_truth_occupied_space_count'])
    candidate=_read(paths[1],['time_sec','occupied_pred_candidate'])
    if 'ground_truth_occupied_space_count' in candidate:
        check=base[['time_sec','ground_truth_occupied_space_count']].merge(candidate[['time_sec','ground_truth_occupied_space_count']],on='time_sec',suffixes=('_safe','_candidate'))
        g1=pd.to_numeric(check.ground_truth_occupied_space_count_safe,errors='coerce');g2=pd.to_numeric(check.ground_truth_occupied_space_count_candidate,errors='coerce')
        if ((g1!=g2)&~(g1.isna()&g2.isna())).any(): raise ValueError('SAFE and Candidate ground truth differ')
    data=base.merge(candidate[['time_sec','occupied_pred_candidate']],on='time_sec',how='left',validate='one_to_one')
    cols=['occupied_pred','occupied_pred_candidate','ground_truth_occupied_space_count']
    for col in cols: data[col]=pd.to_numeric(data[col],errors='coerce')
    common=np.isfinite(data[cols]).all(axis=1)&(data[cols]>=0).all(axis=1)
    warmup=float(settings.get('evaluation_warmup_sec',10))
    cut_warmup=float(cfg.get('cut_warmup_sec',warmup))
    cuts=[float(t) for t in cfg.get('cut_boundaries_sec',[])]
    if any(not math.isfinite(t) or t<0 for t in [warmup,cut_warmup,*cuts]): raise ValueError('Invalid warm-up/cut settings')
    valid=data.time_sec>=warmup
    for cut in cuts: valid &= ~((data.time_sec>=cut)&(data.time_sec<cut+cut_warmup))
    for col in ('eval_valid','episode_eval_valid'):
        if col in data: valid &= pd.to_numeric(data[col],errors='coerce').eq(1)
    rows=[];comparison=[]; terminal=max(w['end_sec'] for w in windows)
    all_windows=[*windows,{'name':'ALL','start_sec':0,'end_sec':None}]
    for w in all_windows:
        mask=(data.time_sec>=w['start_sec'])
        if w['end_sec'] is not None:
            mask &= data.time_sec.le(w['end_sec']) if w['end_sec']==terminal else data.time_sec.lt(w['end_sec'])
        eligible=mask&valid;selected=data[eligible&common]
        metrics={}
        for algorithm,col in [('SAFE_BASELINE','occupied_pred'),('CANDIDATE','occupied_pred_candidate')]:
            m=_metrics(selected[col],selected.ground_truth_occupied_space_count);metrics[algorithm]=m
            rows.append({'window':w['name'],'start_sec':w['start_sec'],'end_sec':w['end_sec'],
                         'algorithm':algorithm,'N':len(selected),'eligible_N':int(eligible.sum()),
                         'missing_pair_N':int((eligible&~common).sum()),'status':'evaluated' if len(selected) else 'no_valid_pairs',**m})
        s,c=metrics['SAFE_BASELINE'],metrics['CANDIDATE']
        comparison.append({'window':w['name'],'N':len(selected),'SAFE_exact':s['exact_rate'],'Candidate_exact':c['exact_rate'],
                           'exact_delta_pp':100*(c['exact_rate']-s['exact_rate']) if len(selected) else None,
                           'SAFE_MAE':s['MAE'],'Candidate_MAE':c['MAE'],'MAE_delta':c['MAE']-s['MAE'] if len(selected) else None})
    metrics_frame=pd.DataFrame(rows);summary={'schema_version':1,'evidence_scope':'cross_window_stability_not_independent_holdout',
        'parameters_tuned':False,'state_mode':'continuous saved trace; no reset or extra warm-up at window boundaries',
        'window_boundary_rule':'start inclusive, end exclusive; largest configured end inclusive; ALL unique full trace',
        'windows':windows,'input_sha256':sources,'global_warmup_sec':warmup,'cut_warmup_sec':cut_warmup,'cuts_sec':cuts,
        'summary_definition':'unweighted macro mean/population std of nonempty configured windows; ALL excluded',
        'overlapping_windows':any(a['start_sec']<b['end_sec'] and b['start_sec']<a['end_sec'] for i,a in enumerate(windows) for b in windows[i+1:]),
        'warning':'Historical DEV or error-reviewed segments are not independent tests. No automatic SAFE promotion.', 'algorithms':{}}
    for algorithm,group in metrics_frame[metrics_frame.window!='ALL'].groupby('algorithm'):
        group=group[group.N>0]
        summary['algorithms'][algorithm]={'evaluated_windows':len(group),'mean_exact':float(group.exact_rate.mean()) if len(group) else None,
            'std_exact':float(group.exact_rate.std(ddof=0)) if len(group) else None,'mean_MAE':float(group.MAE.mean()) if len(group) else None,
            'best_window':str(group.loc[group.exact_rate.idxmax(),'window']) if len(group) else None,
            'worst_window':str(group.loc[group.exact_rate.idxmin(),'window']) if len(group) else None}
    comp=pd.DataFrame(comparison);eligible_comp=comp[(comp.window!='ALL')&(comp.N>0)]
    summary['candidate_exact_improved_windows']=int((eligible_comp.exact_delta_pp>0).sum())
    summary['candidate_MAE_improved_windows']=int((eligible_comp.MAE_delta<0).sum())
    summary['compared_windows']=len(eligible_comp)
    assert all(fingerprint(root/name)==value for name,value in sources.items()),'Source changed during evaluation'
    metrics_frame.to_csv(root/'window_metrics.csv',index=False,encoding='utf-8-sig')
    comp.to_csv(root/'window_comparison.csv',index=False,encoding='utf-8-sig')
    (root/'window_evaluation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (root/'WINDOW_EVALUATION_REPORT.txt').write_text('Frozen SAFE/Candidate cross-window evaluation\n'+summary['warning']+'\n'+summary['state_mode']+'\n'+summary['window_boundary_rule']+'\n'+comp.to_string(index=False)+'\n',encoding='utf-8')
    with zipfile.ZipFile(root/'WINDOW_EVALUATION_TO_CHATGPT.zip','w',zipfile.ZIP_DEFLATED) as archive:
        # Full settings may contain private video paths. Share fingerprints, not that file.
        for name in (*[p.name for p in paths],'candidate_v164_config.json','selected_state_params.json',
                     'window_metrics.csv','window_comparison.csv','window_evaluation_summary.json','WINDOW_EVALUATION_REPORT.txt'):
            if (root/name).is_file(): archive.write(root/name,name)
    return root/'window_comparison.csv'
