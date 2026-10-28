"""Same-video DEV exploration; pseudo initialization, frozen candidate, no human GT."""
import argparse
import json
from pathlib import Path
import pandas as pd
from human_slot_review import sha
from initial_occupied_candidate import InitialOccupiedCue,run_initial_candidate
from manual_initialization import load_snapshot,apply_snapshot,occupancy_bounds
from slot_engine import _run_state_engine_v13_safe,_v16_baseline_params
from utils import crop_roi,open_video,read_frame_at,load_ground_truth


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--pseudo-dir',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--gt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    fit=args.fit;settings=json.loads((fit/'settings_snapshot.json').read_text(encoding='utf8'));freeze=json.loads((fit/'selection_freeze.json').read_text(encoding='utf8'))
    if sha(fit/'selected_state_params.json')!=freeze['sha256']:raise ValueError('Fit freeze mismatch')
    slots=json.loads((fit/'slots.json').read_text(encoding='utf8'))['slots'];rois=json.loads((fit/'worker_input.json').read_text(encoding='utf8'))['rois']
    request=json.loads((fit/'worker_input.json').read_text(encoding='utf8'))
    if Path(request['video']).resolve()!=args.video.resolve():raise ValueError('Video does not match recorded fit path')
    run=sorted((fit/'restart_experiments').glob('run_*'))[-1]
    if json.loads((run/'restart_summary.json').read_text(encoding='utf8'))['status']!='completed':raise ValueError('Incomplete restart source')
    args.output.mkdir(parents=True,exist_ok=False)
    meta={'status':'running','scope':'exploratory, previously reviewed same-video DEV; pseudo initialization, no human GT; thresholds informed by reviewed G018/G021',
          'candidate_sha256':sha(Path(__file__).with_name('initial_occupied_candidate.py')),'video_sha256':sha(args.video),'gt_sha256':sha(args.gt),'inputs':{}}
    (args.output/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    scores=[];cap=open_video(str(args.video))
    try:
        for start,end in [(330,660),(660,900)]:
            src=run/f'restart_{start}'/'evidence'/'segmentation_evidence.csv';ev=pd.read_csv(src,low_memory=False)
            ev=ev[(ev.time_sec>=start)&(ev.time_sec<end)].copy();ids={str(s['global_id']) for s in slots};snapshot=args.pseudo_dir/f'original_{start}_pseudo.csv'
            states=load_snapshot(snapshot,start,ids);params={**_v16_baseline_params(settings),'unlabeled_restart':True,'preserve_unknown':True,'manual_initialization':True}
            if states is None:raise ValueError('Missing pseudo snapshot')
            _,base,_=_run_state_engine_v13_safe(apply_snapshot(ev,states),params)
            index=ev.set_index(['time_sec','local_id']);detectors={s['local_id']:InitialOccupiedCue(states[str(s['global_id'])],s.get('manual_point',s['point']),start) for s in slots}
            rows=[]
            for t in sorted(base.time_sec.unique()):
                frame=read_frame_at(cap,float(t));warped={c:crop_roi(frame,r) for c,r in rois.items()}
                for s in slots:
                    lid=s['local_id'];img=warped[s['cctv']];x,y=map(int,s.get('manual_point',s['point']));h,w=img.shape[:2];crop=img[max(0,y-80):min(h,y+80),max(0,x-80):min(w,x+80)].copy();o=index.loc[(t,lid)]
                    feature=detectors[lid].observe(t,crop,int(o.full_detected),float(o.full_det_conf),int(o.full_track_id),(o.full_center_x,o.full_center_y),int(o.aux_crop_detected),float(o.aux_crop_det_conf),int(o.aux_support_scales))
                    rows.append({'time_sec':t,'global_id':str(s['global_id']),'local_id':lid,'cctv':s['cctv'],**feature})
            candidate=run_initial_candidate(base,pd.DataFrame(rows));candidate.to_csv(args.output/f'{start}_candidate.csv',index=False)
            # Only read GT after candidate inference has been generated.
            gt=load_ground_truth(args.gt);window=gt[(gt.time_sec>start)&(gt.time_sec<end)]
            for name,trace in [('PSEUDO_SAFE',base),('INITIAL_VISUAL_CANDIDATE',candidate)]:
                bounds=occupancy_bounds(trace).merge(window[['time_sec','ground_truth_occupied_space_count']],on='time_sec',validate='one_to_one')
                bounds['gt_within_bounds']=(bounds.ground_truth_occupied_space_count>=bounds.confirmed_occupied)&(bounds.ground_truth_occupied_space_count<=bounds.possible_occupied)
                resolved=bounds[bounds.unknown_slots==0];err=resolved.confirmed_occupied-resolved.ground_truth_occupied_space_count
                scores.append({'start_sec':start,'end_exclusive':end,'variant':name,'N':len(bounds),'resolved_N':len(resolved),'mean_unknown':bounds.unknown_slots.mean(),
                               'exact':float(err.eq(0).mean()) if len(resolved)==len(bounds) and len(bounds) else None,'mae':float(err.abs().mean()) if len(resolved)==len(bounds) and len(bounds) else None,
                               'gt_in_bounds_rate':bounds.gt_within_bounds.mean()})
                bounds.to_csv(args.output/f'{start}_{name}_bounds.csv',index=False)
            meta['inputs'][str(start)]={'evidence_sha256':sha(src),'snapshot_sha256':sha(snapshot),'fit_sha256':freeze['sha256']}
            print('completed',start,flush=True)
        pd.DataFrame(scores).to_csv(args.output/'metrics.csv',index=False);meta['status']='completed'
    except BaseException as exc:meta.update(status='failed',error=repr(exc));raise
    finally:cap.release();(args.output/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')


if __name__=='__main__':main()
