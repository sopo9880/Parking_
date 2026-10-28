"""Compare verified human and pseudo initialization on saved fresh-GPU observations."""
import argparse
import json
from pathlib import Path
import pandas as pd
from human_slot_review import sha
from manual_initialization import load_snapshot, apply_snapshot, occupancy_bounds
from restart_evaluation import recovery_metrics
from slot_engine import _run_state_engine_v13_safe, _v16_baseline_params, evaluate_state_output
from utils import load_ground_truth


def verified_snapshot(snapshot, start, ids, manifest, fit):
    audit_path = snapshot.with_suffix('.audit.json')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    prov = audit['provenance']
    if audit.get('status') != 'human_review_complete' or audit.get('source') != 'operator_review': raise ValueError('Human review is not complete')
    if audit['snapshot_sha256'] != sha(snapshot): raise ValueError('Snapshot/audit hash mismatch')
    if prov['time_sec'] != start or set(audit['reviewed_slot_ids']) != set(ids) or audit['reviewed_count'] != len(ids): raise ValueError('Review time or slots mismatch')
    for name, expected in [('video_sha256',manifest['video_sha256']),('slots_sha256',sha(fit/'slots.json')),
                           ('roi_source_sha256',sha(fit/'worker_input.json')),('pseudo_sha256',manifest['snapshot_sha256'])]:
        if prov[name] != expected: raise ValueError(f'Review provenance mismatch: {name}')
    result = load_snapshot(snapshot,start,ids)
    if result is None: raise ValueError('Exact-time snapshot missing')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh',required=True,type=Path);p.add_argument('--snapshot',required=True,type=Path)
    p.add_argument('--gt',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    m=json.loads((args.fresh/'manifest.json').read_text(encoding='utf-8'))
    if m['status']!='completed':raise ValueError('Fresh GPU experiment is incomplete')
    start=int(m['start_sec']);fit=args.fresh/'fit_copy';root=Path(m['result']);run=root/f'restart_{start}'
    ids={str(x['global_id']) for x in json.loads((fit/'slots.json').read_text(encoding='utf-8'))['slots']}
    human=verified_snapshot(args.snapshot,start,ids,m,fit)
    pseudo=load_snapshot(args.fresh/f'pseudo_detection_{start}.csv',start,ids)
    if sha(args.fresh/f'pseudo_detection_{start}.csv') != m['snapshot_sha256']:raise ValueError('Pseudo hash mismatch')
    evidence_path=run/'evidence'/'segmentation_evidence.csv';evidence=pd.read_csv(evidence_path)
    if evidence.empty or evidence.time_sec.min()!=start:raise ValueError('Evidence start mismatch')
    cfg=json.loads((fit/'settings_snapshot.json').read_text(encoding='utf-8'))
    params={**_v16_baseline_params(cfg),'unlabeled_restart':True,'preserve_unknown':True,'manual_initialization':True}
    traces={};local={};transitions={}
    # GT is intentionally read only after both inference traces are finalized.
    for name,states in [('AI_PSEUDO_INIT',pseudo),('HUMAN_MANUAL_INIT',human)]:
        local[name],traces[name],transitions[name]=_run_state_engine_v13_safe(apply_snapshot(evidence,states),params)
        first=traces[name][traces[name].time_sec==start].set_index('global_id')['state'].to_dict()
        if first!=states:raise ValueError(f'{name} first frame did not preserve the supplied snapshot')
    original=pd.read_csv(run/'RESET_MANUAL_INIT_global_slot_timeseries.csv')
    columns=['time_sec','global_id','state']
    paired=traces['AI_PSEUDO_INIT'][columns].merge(original[columns],on=['time_sec','global_id'],validate='one_to_one',suffixes=('_replay','_original'))
    if len(paired)!=len(original) or len(paired)!=len(traces['AI_PSEUDO_INIT']) or not paired.state_replay.eq(paired.state_original).all():
        raise ValueError('Pseudo replay differs from saved fresh-GPU inference; refusing comparison')
    args.output.mkdir(parents=True,exist_ok=False)
    meta={'status':'running','start_sec':start,'human_snapshot_sha256':sha(args.snapshot),'evidence_sha256':sha(evidence_path),
          'fresh_manifest_sha256':sha(args.fresh/'manifest.json'),'gt_sha256':sha(args.gt),
          'scope':'same saved fresh GPU observations; continuous modes are historical traces; same-video research, not independent validation'}
    try:
        for name in traces:
            traces[name].to_csv(args.output/f'{name}_global.csv',index=False)
            local[name].to_csv(args.output/f'{name}_local.csv',index=False)
            transitions[name].to_csv(args.output/f'{name}_transitions.csv',index=False)
        gt=load_ground_truth(args.gt);end=min(1230,start+330)
        window=gt[(gt.time_sec>=start)&((gt.time_sec<=end) if end==1230 else (gt.time_sec<end))]
        all_traces={name:evaluate_state_output(trace,window,0,0)['timeseries'].merge(occupancy_bounds(trace),on='time_sec',validate='one_to_one') for name,trace in traces.items()}
        for name in ('RESET_SAFE','RESET_RECOVERY','CONTINUOUS_SAFE','CONTINUOUS_CANDIDATE'):
            all_traces[name]=pd.read_csv(run/f'{name}_count_timeseries.csv')
        rows=[]
        for name,trace in all_traces.items():
            manual=name in traces
            if manual:trace['gt_within_bounds']=(trace.ground_truth_occupied_space_count>=trace.confirmed_occupied)&(trace.ground_truth_occupied_space_count<=trace.possible_occupied)
            trace.to_csv(args.output/f'{name}_counts.csv',index=False)
            for scope,part in [('INITIAL',trace[trace.time_sec==start]),('AFTER_INITIAL',trace[trace.time_sec>start]),
                               *[(f'FIRST_{n}_SEC',trace[(trace.time_sec>start)&(trace.time_sec<start+n)]) for n in (30,60,120)]]:
                metric=recovery_metrics(part,start)
                if manual:
                    metric.update(unknown_mean=float(part.unknown_slots.mean()) if len(part) else None,fully_resolved_N=int(part.unknown_slots.eq(0).sum()),
                                  gt_within_bounds_rate=float(part.gt_within_bounds.mean()) if len(part) else None)
                    if part.unknown_slots.gt(0).any():
                        for key in ('exact_rate','mae','max_abs_error','over_rate','under_rate','time_to_first_exact_sec','time_to_stable_exact_sec','stable_confirmed_at_sec','never_stabilized','post_stable_exact_rate','post_stable_mae','time_to_first_relapse_sec','relapse_count','longest_exact_run_samples','tail_exact_samples'):
                            metric[key]=None
                        for key in list(metric):
                            if key.startswith('first_'):metric[key]=None
                rows.append({'variant':name,'scope':scope,**metric})
        pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False);meta['status']='completed'
    except BaseException as exc:
        meta.update(status='failed',error=repr(exc));raise
    finally:(args.output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
