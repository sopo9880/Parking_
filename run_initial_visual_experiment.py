"""Replay an unvalidated initial-occupied visual/motion candidate on saved GPU evidence."""
import argparse
import json
from pathlib import Path
import pandas as pd
from human_slot_review import sha
from initial_occupied_candidate import InitialOccupiedCue,run_initial_candidate
from manual_initialization import load_snapshot,occupancy_bounds
from utils import crop_roi,open_video,read_frame_at,load_ground_truth


def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    m=json.loads((args.fresh/'manifest.json').read_text(encoding='utf8'));comp=json.loads((args.comparison/'manifest.json').read_text(encoding='utf8'))
    if m['status']!='completed' or comp['status']!='completed' or comp['fresh_manifest_sha256']!=sha(args.fresh/'manifest.json') or sha(args.video)!=m['video_sha256'] or sha(args.snapshot)!=comp['human_snapshot_sha256']:
        raise ValueError('Inputs do not match completed human-initialized GPU experiment')
    start=int(m['start_sec']);fit=args.fresh/'fit_copy';run=Path(m['result'])/f'restart_{start}'
    slots=json.loads((fit/'slots.json').read_text(encoding='utf8'))['slots'];rois=json.loads((fit/'worker_input.json').read_text(encoding='utf8'))['rois']
    ids={str(s['global_id']) for s in slots};initial=load_snapshot(args.snapshot,start,ids)
    source=run/'evidence'/'segmentation_evidence.csv'
    if sha(source)!=comp['evidence_sha256']:raise ValueError('Saved evidence changed')
    e=pd.read_csv(source,low_memory=False).set_index(['time_sec','local_id']);base=pd.read_csv(args.comparison/'HUMAN_MANUAL_INIT_global.csv')
    detectors={s['local_id']:InitialOccupiedCue(initial[str(s['global_id'])],s.get('manual_point',s['point']),start) for s in slots}
    times=sorted(base.time_sec.unique());rows=[];cap=open_video(str(args.video))
    try:
        for t in times:
            frame=read_frame_at(cap,float(t));warped={c:crop_roi(frame,r) for c,r in rois.items()}
            for s in slots:
                lid=s['local_id'];img=warped[s['cctv']];x,y=map(int,s.get('manual_point',s['point']));h,w=img.shape[:2]
                crop=img[max(0,y-80):min(h,y+80),max(0,x-80):min(w,x+80)].copy();o=e.loc[(t,lid)]
                feat=detectors[lid].observe(t,crop,int(o.full_detected),float(o.full_det_conf),int(o.full_track_id),
                    (o.full_center_x,o.full_center_y),int(o.aux_crop_detected),float(o.aux_crop_det_conf),int(o.aux_support_scales))
                rows.append({'time_sec':t,'global_id':str(s['global_id']),'local_id':lid,'cctv':s['cctv'],**feat})
    finally:cap.release()
    features=pd.DataFrame(rows);candidate=run_initial_candidate(base,features)
    args.output.mkdir(parents=True,exist_ok=False)
    features.to_csv(args.output/'features.csv',index=False);candidate.to_csv(args.output/'candidate_global.csv',index=False)
    # GT enters only after all candidate states have been generated.
    if sha(args.gt)!=comp['gt_sha256']:raise ValueError('Count GT changed')
    gt=load_ground_truth(args.gt);end=min(1230,start+330);window=gt[(gt.time_sec>start)&((gt.time_sec<=end) if end==1230 else (gt.time_sec<end))]
    bounds=occupancy_bounds(candidate).merge(window[['time_sec','ground_truth_occupied_space_count']],on='time_sec',validate='one_to_one')
    bounds['gt_within_bounds']=(bounds.ground_truth_occupied_space_count>=bounds.confirmed_occupied)&(bounds.ground_truth_occupied_space_count<=bounds.possible_occupied)
    bounds.to_csv(args.output/'bounds.csv',index=False)
    resolved=bounds[bounds.unknown_slots==0];err=resolved.confirmed_occupied-resolved.ground_truth_occupied_space_count
    summary={'status':'completed','start_sec':start,'N':len(bounds),'fully_resolved_N':len(resolved),'mean_unknown':float(bounds.unknown_slots.mean()),
             'exact_all':float(err.eq(0).mean()) if len(resolved)==len(bounds) else None,'mae_all':float(err.abs().mean()) if len(resolved)==len(bounds) else None,
             'fraction_within_bounds':float(bounds.gt_within_bounds.mean()),'candidate_sha256':sha(Path(__file__).with_name('initial_occupied_candidate.py')),
             'runner_sha256':sha(Path(__file__)),'video_sha256':m['video_sha256'],'snapshot_sha256':comp['human_snapshot_sha256'],'evidence_sha256':comp['evidence_sha256'],'gt_sha256':comp['gt_sha256'],
             'scope':'exploratory on previously inspected same-video cases; thresholds hand-specified after observation; not independent validation',
             'warning':'Image similarity and track motion are unvalidated vacancy proxies; cannot establish an empty floor or rule out occlusion.'}
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
