"""Create local, private image panels for unverified UNKNOWN initial slots."""
import argparse
import json
from pathlib import Path
import cv2
from utils import crop_roi, open_video, read_frame_at


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path, help='fresh experiment output directory')
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed': raise ValueError('Fresh run must be complete')
    start=manifest['start_sec']; fit=args.run/'fit_copy'; folder=Path(manifest['result'])/f'restart_{start}'
    audit=json.loads((folder/'manual_init_audit.json').read_text(encoding='utf-8'))
    slots=json.loads((fit/'slots.json').read_text(encoding='utf-8'))['slots']
    rois=json.loads((fit/'worker_input.json').read_text(encoding='utf-8'))['rois']
    ids={gid for gid,state in audit['states'].items() if state=='UNKNOWN'}
    args.output.mkdir(parents=True,exist_ok=False)
    cap=open_video(str(args.video))
    try:
        for t in (start,min(start+30,1230),min(start+120,1230)):
            frame=read_frame_at(cap,t)
            if frame is None: raise RuntimeError(f'Cannot read {t}s')
            for s in slots:
                if str(s['global_id']) not in ids: continue
                image=crop_roi(frame,rois[s['cctv']]); x,y=map(int,s.get('manual_point',s['point']))
                h,w=image.shape[:2];x1=max(0,x-100);x2=min(w,x+100);y1=max(0,y-100);y2=min(h,y+100)
                cv2.circle(image,(x,y),5,(0,0,255),-1)
                crop=image[y1:y2,x1:x2]
                if crop.size and not cv2.imwrite(str(args.output/f"{s['global_id']}_{s['local_id']}_{t}s.jpg"),crop):
                    raise OSError('Failed to write crop')
    finally: cap.release()
    (args.output/'README.txt').write_text('Private CCTV crops for visual inspection. UNKNOWN is not a demonstrated error. Do not publish.\n',encoding='utf-8')


if __name__=='__main__': main()
