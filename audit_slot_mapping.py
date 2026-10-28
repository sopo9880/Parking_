"""Record structural slot mapping and exact-time frame sampling without changing labels."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from human_slot_review import sha
from manual_initialization import load_snapshot
from utils import crop_roi, open_video, read_frame_at, load_ground_truth


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--video',type=Path,required=True)
    p.add_argument('--gt',type=Path,required=True);p.add_argument('--snapshot-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    slots=json.loads((args.fit/'slots.json').read_text(encoding='utf-8'))['slots'];rois=json.loads((args.fit/'worker_input.json').read_text(encoding='utf-8'))['rois']
    df=pd.DataFrame(slots);ids=set(df.global_id.astype(str));gt=load_ground_truth(args.gt)
    args.output.mkdir(parents=True,exist_ok=False); rows=[];times=[]
    cap=open_video(args.video)
    try:
        for sec in (0,900):
            snapshot=args.snapshot_dir/f'human_{sec:04d}.csv';states=load_snapshot(snapshot,sec,ids)
            if states is None:raise ValueError('Snapshot is missing exact timestamp')
            frame=read_frame_at(cap,sec);times.append({'requested_sec':sec,'reported_position_sec':cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                'frame_index_after_read':cap.get(cv2.CAP_PROP_POS_FRAMES),'human_occupied':sum(x=='OCCUPIED' for x in states.values()),
                'gt_occupied':int(gt.loc[gt.time_sec==sec,'ground_truth_occupied_space_count'].iloc[0]),'snapshot_sha256':sha(snapshot)})
            for cam in sorted(df.cctv.unique()):
                image=crop_roi(frame,rois[cam]);subset=df[df.cctv==cam]
                for s in subset.to_dict('records'):
                    x,y=map(float,s.get('manual_point',s['point']));peer=[]
                    for other in subset.to_dict('records'):
                        if other['local_id']==s['local_id']:continue
                        ox,oy=map(float,other.get('manual_point',other['point']));peer.append((float(np.hypot(x-ox,y-oy)),other['global_id']))
                    nearest=min(peer) if peer else (None,None)
                    rows.append({'time_sec':sec,'global_id':s['global_id'],'local_id':s['local_id'],'camera':cam,'x':x,'y':y,'human_state':states[str(s['global_id'])],
                                 'nearest_point_px':nearest[0],'nearest_global_id':nearest[1]})
                    cv2.circle(image,(int(x),int(y)),4,(0,0,255),-1);cv2.putText(image,str(s['global_id']),(int(x)+4,int(y)-4),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
                cv2.imwrite(str(args.output/f'private_{cam}_{sec}.jpg'),image)
    finally:cap.release()
    pd.DataFrame(rows).to_csv(args.output/'local_global_mapping.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps({'local_slots':len(df),'global_slots':len(ids),'same_camera_duplicate_global':int(df.duplicated(['cctv','global_id']).sum()),
        'duplicate_local_id':int(df.local_id.duplicated().sum()),'time_checks':times,'video_sha256':sha(args.video),'gt_sha256':sha(args.gt),
        'scope':'structural checks only; proximity and camera overlap do not establish physical identity'},indent=2),encoding='utf-8')
    (args.output/'README.txt').write_text('Private CCTV mapping overlays. Do not publish. Structural consistency does not verify physical slot identity.\n',encoding='utf-8')


if __name__=='__main__':main()
