"""Explore research policy on DEV windows with explicit pseudo-initialization only."""
import argparse
import json
from pathlib import Path
import pandas as pd
from evidence_release_candidate import run_candidate
from human_slot_review import sha
from manual_initialization import load_snapshot,apply_snapshot,occupancy_bounds
from slot_engine import _run_state_engine_v13_safe,_v16_baseline_params
from utils import load_ground_truth


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',required=True,type=Path);p.add_argument('--pseudo-dir',required=True,type=Path)
    p.add_argument('--gt',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    settings=json.loads((args.fit/'settings_snapshot.json').read_text(encoding='utf-8'))
    freeze=json.loads((args.fit/'selection_freeze.json').read_text(encoding='utf-8'))
    if sha(args.fit/'selected_state_params.json')!=freeze['sha256']:raise ValueError('Fit freeze mismatch')
    run=sorted((args.fit/'restart_experiments').glob('run_*'))[-1]
    if json.loads((run/'restart_summary.json').read_text(encoding='utf-8'))['status']!='completed':raise ValueError('Saved restart not complete')
    args.output.mkdir(parents=True,exist_ok=False);meta={'status':'running','scope':'same historically reviewed video DEV segments; pseudo init, not human or independent',
        'inputs':{},'candidate_code_sha256':sha(Path(__file__).with_name('evidence_release_candidate.py')),'gt_sha256':sha(args.gt),'test_labels_used_for_selection':False}
    rows=[]
    try:
        for start in (330,660):
            src=run/f'restart_{start}'/'evidence'/'segmentation_evidence.csv';ev=pd.read_csv(src,low_memory=False)
            ev=ev[(ev.time_sec>=start)&(ev.time_sec<900)].copy()
            ids=set(ev.global_id.astype(str));snap=args.pseudo_dir/f'original_{start}_pseudo.csv';states=load_snapshot(snap,start,ids)
            if states is None:raise ValueError('Missing exact pseudo snapshot')
            params={**_v16_baseline_params(settings),'unlabeled_restart':True,'preserve_unknown':True,'manual_initialization':True}
            _,base,_=_run_state_engine_v13_safe(apply_snapshot(ev,states),params)
            cand=run_candidate(base,ev)
            # Scoring starts only after inference; GT does not select candidate.
            gt=load_ground_truth(args.gt);part=gt[(gt.time_sec>start)&(gt.time_sec<min(900,start+330))]
            for name,trace in [('PSEUDO_SAFE',base),('RELEASE_CANDIDATE',cand)]:
                bounds=occupancy_bounds(trace).merge(part[['time_sec','ground_truth_occupied_space_count']],on='time_sec',validate='one_to_one')
                bounds['gt_in_bounds']=(bounds.ground_truth_occupied_space_count>=bounds.confirmed_occupied)&(bounds.ground_truth_occupied_space_count<=bounds.possible_occupied)
                resolved=bounds[bounds.unknown_slots==0];error=resolved.confirmed_occupied-resolved.ground_truth_occupied_space_count
                rows.append({'start_sec':start,'end_exclusive':min(900,start+330),'mode':name,'N':len(bounds),'fully_resolved_N':len(resolved),
                    'unknown_mean':bounds.unknown_slots.mean(),'gt_in_bounds_rate':bounds.gt_in_bounds.mean(),
                    'exact':float(error.eq(0).mean()) if len(resolved)==len(bounds) and len(bounds) else None,
                    'mae':float(error.abs().mean()) if len(resolved)==len(bounds) and len(bounds) else None})
                bounds.to_csv(args.output/f'{start}_{name}_bounds.csv',index=False)
            cand.to_csv(args.output/f'{start}_candidate.csv',index=False)
            meta['inputs'][str(start)]={'evidence_sha256':sha(src),'pseudo_snapshot_sha256':sha(snap),'fit_sha256':freeze['sha256']}
        pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False);meta['status']='completed'
    except BaseException as exc:meta.update(status='failed',error=repr(exc));raise
    finally:(args.output/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


if __name__=='__main__':main()
