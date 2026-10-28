"""Measure causal crop changes and detector state for specified local review windows."""
import argparse
import json
from pathlib import Path
import cv2
import pandas as pd
from human_slot_review import sha
from utils import crop_roi, normalized_visual_diff, open_video, read_frame_at


def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    manifest=json.loads((args.fresh/'manifest.json').read_text(encoding='utf-8'));comp=json.loads((args.comparison/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed' or comp['fresh_manifest_sha256']!=sha(args.fresh/'manifest.json') or sha(args.video)!=manifest['video_sha256']:
        raise ValueError('Evidence provenance mismatch')
    start=int(manifest['start_sec']);fit=args.fresh/'fit_copy';run=Path(manifest['result'])/f'restart_{start}'
    slots=json.loads((fit/'slots.json').read_text(encoding='utf-8'))['slots'];rois=json.loads((fit/'worker_input.json').read_text(encoding='utf-8'))['rois']
    ids={'G018','G028','G021'} if start==900 else {'G002','G029'}
    selected=[s for s in slots if str(s['global_id']) in ids]
    times=sorted(set(range(start,start+21)) | (set(range(938,953)) if start==900 else set()))
    evidence=pd.read_csv(run/'evidence'/'segmentation_evidence.csv',low_memory=False)
    local=pd.read_csv(args.comparison/'HUMAN_MANUAL_INIT_local.csv')
    obs=evidence.set_index(['time_sec','local_id']);states=local.set_index(['time_sec','local_id']);previous={};reference={};rows=[]
    cap=open_video(str(args.video))
    try:
        for t in times:
            frame=read_frame_at(cap,t);warped={cam:crop_roi(frame,roi) for cam,roi in rois.items()}
            for s in selected:
                lid=s['local_id'];gid=str(s['global_id']);img=warped[s['cctv']];x,y=map(int,s.get('manual_point',s['point']));h,w=img.shape[:2]
                crop=img[max(0,y-80):min(h,y+80),max(0,x-80):min(w,x+80)].copy()
                if crop.size==0:raise ValueError('Invalid crop')
                if lid not in reference:reference[lid]=crop
                # Previous requested frame is not always the previous second across gaps.
                prev=previous.get(lid);delta=normalized_visual_diff(crop,prev[1]) if prev is not None else 0.0
                row=obs.loc[(float(t),lid)];state=states.loc[(float(t),lid)]
                peers=evidence[(evidence.time_sec==t)&(evidence.global_id==gid)&(evidence.local_id!=lid)]
                rows.append({'time_sec':t,'global_id':gid,'local_id':lid,'cctv':s['cctv'],'state':state['state'],'phase':state['phase'],
                    'exit_gate':state['exit_gate_ok'],'window_hit_ratio':state['window_hit_ratio'],'recent_hit_ratio':state['recent_hit_ratio'],
                    'full_detected':row['full_detected'],'full_conf':row['full_det_conf'],'crop_detected':row['crop_detected'],'crop_conf':row['crop_det_conf'],
                    'full_speed_px_s':row['full_track_speed_px_s'],'track_span_px':row['full_track_motion_span_px'],
                    'delta_from_start':normalized_visual_diff(crop,reference[lid]),'delta_from_previous_sample':delta,
                    'previous_sample_sec':prev[0] if prev else None,'other_camera_count':len(peers),'other_camera_full_detected':int(peers.full_detected.sum()),
                    'departure_evidence':'unverified_visual_motion_not_a_ground_truth_label'})
                previous[lid]=(t,crop)
    finally:cap.release()
    if args.output.exists():raise FileExistsError(args.output)
    pd.DataFrame(rows).to_csv(args.output,index=False)


if __name__=='__main__':main()
