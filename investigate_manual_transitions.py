"""Local diagnostic evidence for first manual-init occupied-to-empty transitions."""
import argparse
import json
from pathlib import Path
import cv2
import pandas as pd
from human_slot_review import sha
from utils import crop_roi, open_video, read_frame_at


def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    m=json.loads((args.fresh/'manifest.json').read_text(encoding='utf-8'))
    if m['status']!='completed' or sha(args.video)!=m['video_sha256']:raise ValueError('Video does not match completed fresh experiment')
    comp=json.loads((args.comparison/'manifest.json').read_text(encoding='utf-8'))
    if comp['status']!='completed' or comp['fresh_manifest_sha256']!=sha(args.fresh/'manifest.json'):raise ValueError('Comparison mismatch')
    start=int(m['start_sec']);fit=args.fresh/'fit_copy';run=Path(m['result'])/f'restart_{start}'
    slots=json.loads((fit/'slots.json').read_text(encoding='utf-8'))['slots']
    rois=json.loads((fit/'worker_input.json').read_text(encoding='utf-8'))['rois']
    changes=pd.read_csv(args.comparison/'HUMAN_MANUAL_INIT_transitions.csv')
    changes=changes[(changes.scope=='GLOBAL')&(changes.from_state=='OCCUPIED')&(changes.to_state=='EMPTY')].sort_values('time_sec').groupby('global_id',as_index=False).first()
    local=pd.read_csv(args.comparison/'HUMAN_MANUAL_INIT_local.csv');e=pd.read_csv(run/'evidence'/'segmentation_evidence.csv',low_memory=False)
    args.output.mkdir(parents=True,exist_ok=False); images=args.output/'private_images';images.mkdir()
    rows=[];cap=open_video(str(args.video))
    try:
        for event in changes.itertuples():
            gid=str(event.global_id);t=float(event.time_sec)
            if t>start+120:continue
            for slot in (s for s in slots if str(s['global_id'])==gid):
                lid=slot['local_id'];data=e[(e.local_id==lid)&e.time_sec.between(start,t)]
                last=local[(local.local_id==lid)&(local.time_sec==t)]
                rows.append({'global_id':gid,'local_id':lid,'cctv':slot['cctv'],'start_sec':start,'first_global_clear_sec':t,
                             'rows_before_clear':len(data),'full_detection_count':int(data.full_detected.sum()),
                             'max_full_confidence':float(data.full_det_conf.max()) if len(data) else None,
                             'local_state_at_global_clear':last.state.iloc[0] if len(last) else None,
                             'window_hit_ratio_at_clear':float(last.window_hit_ratio.iloc[0]) if len(last) else None,
                             'recent_hit_ratio_at_clear':float(last.recent_hit_ratio.iloc[0]) if len(last) else None,
                             'exit_gate_at_clear':bool(last.exit_gate_ok.iloc[0]) if len(last) else None})
                for when in sorted({start,max(start,t-1),min(t+1,1230)}):
                    frame=read_frame_at(cap,when);image=crop_roi(frame,rois[slot['cctv']]);x,y=map(int,slot.get('manual_point',slot['point']))
                    h,w=image.shape[:2];cv2.circle(image,(x,y),7,(0,0,255),2);crop=image[max(0,y-140):min(h,y+140),max(0,x-140):min(w,x+140)]
                    if not crop.size or not cv2.imwrite(str(images/f'{gid}_{lid}_{int(when)}s.jpg'),crop):raise RuntimeError('Unable to save crop')
    finally:cap.release()
    pd.DataFrame(rows).to_csv(args.output/'transition_evidence.csv',index=False)
    (args.output/'README.txt').write_text('Local private CCTV crops and state/YOLO diagnostics. A state transition is not a confirmed error without later per-slot GT.\n',encoding='utf-8')


if __name__=='__main__':main()
