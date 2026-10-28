"""Replay exact-time unverified pseudo snapshots on archived restart evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from fresh_manual_experiment import pseudo_snapshot, digest
from manual_initialization import apply_snapshot, occupancy_bounds, write_snapshot
from restart_evaluation import DEFAULT_RECOVERY, recovery_metrics
from slot_engine import _run_state_engine_v13_safe, _v16_baseline_params, evaluate_state_output
from utils import load_ground_truth


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    gt=load_ground_truth(args.gt);rows=[]
    manifest={'status':'running','scope':'unverified pseudo-init; saved evidence replay; same historically reviewed video',
              'rule':'exact-time full detection >= 0.25 is OCCUPIED; all others UNKNOWN', 'gt_sha256':digest(args.gt),'inputs':{}}
    try:
        for split in ('original','alternate'):
            fit=args.source/split;settings=json.loads((fit/'settings_snapshot.json').read_text(encoding='utf-8'))
            freeze=json.loads((fit/'selection_freeze.json').read_text(encoding='utf-8'))
            if digest(fit/'selected_state_params.json') != freeze['sha256']: raise ValueError('Fit hash mismatch')
            run=sorted((fit/'restart_experiments').glob('run_*'))[-1]
            if json.loads((run/'restart_summary.json').read_text(encoding='utf-8'))['status']!='completed': raise ValueError('Incomplete run')
            ids={str(s['global_id']) for s in json.loads((fit/'slots.json').read_text(encoding='utf-8'))['slots']}
            for start in (0,330,660,900):
                source=run/f'restart_{start}'/'evidence'/'slot_evidence.csv'; evidence=pd.read_csv(source)
                states=pseudo_snapshot(evidence,ids,start); label=args.output/f'{split}_{start}_pseudo.csv';write_snapshot(label,start,states)
                manifest['inputs'][f'{split}/{start}']={'evidence_sha256':digest(source),'snapshot_sha256':digest(label),'selected_fit_sha256':freeze['sha256']}
                end=min(1230,start+330);part=gt[(gt.time_sec>=start)&((gt.time_sec<=end) if end==1230 else (gt.time_sec<end))]
                params={**_v16_baseline_params(settings),'unlabeled_restart':True}
                traces={}
                for name,kwargs in [('RESET_SAFE',{}),('RESET_RECOVERY',{'startup_recovery':DEFAULT_RECOVERY}),
                                    ('RESET_UNKNOWN',{'preserve_unknown':True}),('RESET_MANUAL_INIT',{'preserve_unknown':True,'manual_initialization':True})]:
                    data=apply_snapshot(evidence,states) if name=='RESET_MANUAL_INIT' else evidence
                    _,glob,_=_run_state_engine_v13_safe(data,{**params,**kwargs});traces[name]=glob
                for name,file in [('CONTINUOUS_SAFE','baseline_global_slot_timeseries.csv'),('CONTINUOUS_CANDIDATE','candidate_v164_global_slot_timeseries.csv')]:
                    traces[name]=pd.read_csv(fit/file)
                for name,trace in traces.items():
                    result=evaluate_state_output(trace,part,0,0)['timeseries'];manual=name in ('RESET_MANUAL_INIT','RESET_UNKNOWN')
                    if manual:
                        result=result.merge(occupancy_bounds(trace),on='time_sec',validate='one_to_one')
                        result['gt_within_bounds']=(result.ground_truth_occupied_space_count>=result.confirmed_occupied)&(result.ground_truth_occupied_space_count<=result.possible_occupied)
                    result.to_csv(args.output/f'{split}_{start}_{name}.csv',index=False)
                    for scope,segment in [('INITIAL',result[result.time_sec==start]),('AFTER_INITIAL',result[result.time_sec>start])]:
                        metric=recovery_metrics(segment,start)
                        if manual:
                            metric['unknown_mean']=float(segment.unknown_slots.mean()) if len(segment) else None
                            metric['fully_resolved_N']=int(segment.unknown_slots.eq(0).sum())
                            metric['gt_within_bounds_rate']=float(segment.gt_within_bounds.mean()) if len(segment) else None
                            if segment.unknown_slots.gt(0).any():
                                for key in ('exact_rate','mae','max_abs_error','over_rate','under_rate','time_to_first_exact_sec','time_to_stable_exact_sec','stable_confirmed_at_sec','never_stabilized','post_stable_exact_rate','post_stable_mae','time_to_first_relapse_sec','relapse_count','longest_exact_run_samples','tail_exact_samples'):
                                    metric[key]=None
                                for key in list(metric):
                                    if key.startswith('first_'): metric[key]=None
                        rows.append({'split':split,'start_sec':start,'variant':name,'scope':scope,**metric})
                pd.DataFrame(rows).to_csv(args.output/'metrics_partial.csv',index=False)
                print(split,start,'pseudo occupied',sum(x=='OCCUPIED' for x in states.values()),'unknown',sum(x=='UNKNOWN' for x in states.values()),flush=True)
        pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False);manifest['status']='completed'
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__': main()
