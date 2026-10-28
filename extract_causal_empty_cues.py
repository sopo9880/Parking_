"""Extract uncalibrated empty-reference proposals without reading future GT."""
import argparse
import json
from pathlib import Path
import pandas as pd
from causal_empty_cues import CausalEmptyCue
from human_slot_review import sha
from manual_initialization import load_snapshot
from utils import crop_roi, open_video, read_frame_at


def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    m=json.loads((args.fresh/'manifest.json').read_text(encoding='utf-8'));c=json.loads((args.comparison/'manifest.json').read_text(encoding='utf-8'))
    if m['status']!='completed' or c['status']!='completed' or c['fresh_manifest_sha256']!=sha(args.fresh/'manifest.json') or sha(args.video)!=m['video_sha256'] or sha(args.snapshot)!=c['human_snapshot_sha256']:
        raise ValueError('Fresh, comparison, video or snapshot provenance mismatch')
    start=int(m['start_sec']);fit=args.fresh/'fit_copy';run=Path(m['result'])/f'restart_{start}'
    slots=json.loads((fit/'slots.json').read_text(encoding='utf-8'))['slots'];rois=json.loads((fit/'worker_input.json').read_text(encoding='utf-8'))['rois']
    ids={str(s['global_id']) for s in slots};initial=load_snapshot(args.snapshot,start,ids)
    targets=[s for s in slots if initial[str(s['global_id'])]=='EMPTY']
    source=run/'evidence'/'segmentation_evidence.csv'
    if sha(source)!=c['evidence_sha256']:raise ValueError('Evidence changed')
    evidence=pd.read_csv(source,low_memory=False).set_index(['time_sec','local_id'])
    local=pd.read_csv(args.comparison/'HUMAN_MANUAL_INIT_local.csv').set_index(['time_sec','local_id'])
    detectors={s['local_id']:CausalEmptyCue('EMPTY',s.get('manual_point',s['point']),start) for s in targets}
    times=sorted(local.index.get_level_values('time_sec').unique());rows=[];cap=open_video(str(args.video))
    try:
        for t in times:
            frame=read_frame_at(cap,float(t));warped={s['cctv']:crop_roi(frame,rois[s['cctv']]) for s in targets}
            for s in targets:
                lid=s['local_id'];img=warped[s['cctv']];x,y=map(int,s.get('manual_point',s['point']));h,w=img.shape[:2]
                crop=img[max(0,y-60):min(h,y+60),max(0,x-60):min(w,x+60)].copy()
                row=evidence.loc[(t,lid)];state=local.loc[(t,lid)];center=(row['full_center_x'],row['full_center_y'])
                feature=detectors[lid].observe(t,crop,int(row['full_detected']),float(row['full_det_conf']),int(row['full_track_id']),center,state['state'])
                rows.append({'time_sec':float(t),'global_id':str(s['global_id']),'local_id':lid,'cctv':s['cctv'],**feature})
    finally:cap.release()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);df=pd.DataFrame(rows);df.to_csv(args.output/'per_view_features.csv',index=False)
    if len(df):
        df['joint']=df.outward_motion & df.empty_visible
        group=df.groupby(['time_sec','global_id'],as_index=False).joint.any()
        group['outward_motion']=group.joint;group['empty_visible']=group.joint
        group[['time_sec','global_id','outward_motion','empty_visible']].to_csv(args.output/'exit_cues.csv',index=False)
    else:pd.DataFrame(columns=['time_sec','global_id','outward_motion','empty_visible']).to_csv(args.output/'exit_cues.csv',index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status':'completed','start_sec':start,'video_sha256':sha(args.video),'snapshot_sha256':sha(args.snapshot),
        'evidence_sha256':sha(source),'comparison_manifest_sha256':sha(args.comparison/'manifest.json'),'target_local_slots':len(targets),
        'parameters':{'empty_difference_max':.035,'occupied_difference_min':.06,'empty_streak':3,'radial_increase_px':6,'motion_memory_sec':15},
        'warning':'Uncalibrated visual proxy; crop similarity and apparent radial motion do not prove visible emptiness or actual departure. Same-video exploratory.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
