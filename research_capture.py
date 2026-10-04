# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from utils import crop_roi, ensure_dir, format_timestamp, load_json, open_video, read_frame_at
from slot_engine import load_slots


def _letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    canvas = np.full((h, w, 3), 28, np.uint8)
    if img is None or img.size == 0:
        return canvas
    ih, iw = img.shape[:2]
    scale = min(w / max(1, iw), h / max(1, ih))
    nw, nh = max(1, int(round(iw*scale))), max(1, int(round(ih*scale)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x, y = (w-nw)//2, (h-nh)//2
    canvas[y:y+nh, x:x+nw] = r
    return canvas


def _panel_grid(panels: List[np.ndarray], cols: int = 2) -> np.ndarray:
    if not panels:
        return np.full((720,1280,3),28,np.uint8)
    h,w=panels[0].shape[:2]; cols=max(1,int(cols)); rows=int(math.ceil(len(panels)/cols))
    padded=list(panels)
    while len(padded)<rows*cols: padded.append(np.full((h,w,3),28,np.uint8))
    return np.vstack([np.hstack(padded[r*cols:(r+1)*cols]) for r in range(rows)])


def _nearest_row(df: pd.DataFrame, t: float, tol: float = 0.51) -> pd.DataFrame:
    if df is None or df.empty or 'time_sec' not in df.columns:
        return pd.DataFrame()
    vals = df['time_sec'].astype(float)
    d = (vals-float(t)).abs()
    if d.min() > tol:
        return pd.DataFrame()
    return df[d <= d.min()+1e-9].copy()


def _draw_warped(img: np.ndarray, cctv: str, slots: Sequence[Dict], state_rows: pd.DataFrame, evidence_rows: pd.DataFrame) -> np.ndarray:
    out = img.copy()
    state_map = {str(r.local_id): r for r in state_rows.itertuples()} if not state_rows.empty else {}
    ev_map = {str(r.local_id): r for r in evidence_rows.itertuples()} if not evidence_rows.empty else {}
    for s in slots:
        if str(s.get('cctv')) != str(cctv):
            continue
        lid = str(s['local_id']); gid = str(s.get('global_id', lid)); px, py = [int(round(v)) for v in s['point']]
        sr = state_map.get(lid); er = ev_map.get(lid)
        state = str(getattr(sr, 'state', s.get('initial_state','UNKNOWN'))).upper() if sr else str(s.get('initial_state','UNKNOWN')).upper()
        phase = str(getattr(sr, 'phase', state)).upper() if sr else state
        if phase == 'MANEUVERING': color = (0,165,255)
        elif state == 'OCCUPIED': color = (0,0,255)
        else: color = (0,200,0)
        if er is not None:
            try:
                x1,y1,x2,y2 = int(er.crop_x1),int(er.crop_y1),int(er.crop_x2),int(er.crop_y2)
                cv2.rectangle(out,(x1,y1),(x2,y2),(150,150,150),1)
            except Exception:
                pass
        cv2.circle(out,(px,py),8,color,-1); cv2.circle(out,(px,py),11,(255,255,255),2)
        full = int(getattr(er,'full_detected',0)) if er is not None else 0
        crop = int(getattr(er,'aux_crop_detected',getattr(er,'crop_detected',0))) if er is not None else 0
        support = int(getattr(er,'aux_support_scales',0)) if er is not None else 0
        seg = int(getattr(er,'seg_detected',0)) if er is not None else 0
        label = f'{gid} {state[0]} F{full} A{crop} SEG{seg} S{support}'
        cv2.putText(out,label,(max(0,px+9),max(20,py-7)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),4,cv2.LINE_AA); cv2.putText(out,label,(max(0,px+9),max(20,py-7)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
    cv2.rectangle(out,(6,6),(180,40),(15,15,15),-1); cv2.putText(out,cctv.upper(),(14,32),cv2.FONT_HERSHEY_SIMPLEX,0.78,(255,255,255),2,cv2.LINE_AA)
    return out


def _count_info(count_df: pd.DataFrame, t: float) -> Tuple[str,str,str]:
    sub = _nearest_row(count_df,t,tol=5.1)
    if sub.empty:
        return ('?','?','?')
    r=sub.iloc[0]
    gt=str(int(r['ground_truth_occupied_space_count'])) if 'ground_truth_occupied_space_count' in r else '?'
    pred=str(int(r['occupied_pred'])) if 'occupied_pred' in r else '?'
    err=str(int(r['error'])) if 'error' in r else '?'
    return gt,pred,err


def _select_times(count_df: pd.DataFrame, slot_df: pd.DataFrame, max_error: int = 3, max_success: int = 3, max_maneuver: int = 2) -> List[Tuple[str,float]]:
    chosen: List[Tuple[str,float]] = []
    if not count_df.empty:
        times = count_df['time_sec'].astype(float).tolist()
        for label,target in [('overview_start',10.0),('overview_dev_mid',450.0),('overview_test_start',900.0),('overview_end',1230.0)]:
            if times:
                t=min(times,key=lambda x:abs(x-target)); chosen.append((label,float(t)))
        errs=count_df[(count_df.get('split','')=='TEST') & (count_df.get('error',0)!=0)].copy()
        if not errs.empty:
            errs['abs_sort']=errs['error'].abs();errs=errs.sort_values(['abs_sort','time_sec'],ascending=[False,True])
            for i,r in enumerate(errs.head(max_error).itertuples(),1):chosen.append((f'error_{i}',float(r.time_sec)))
        oks=count_df[(count_df.get('split','')=='TEST') & (count_df.get('error',0)==0)].copy()
        if not oks.empty:
            idx=np.linspace(0,len(oks)-1,min(max_success,len(oks))).round().astype(int)
            for i,j in enumerate(idx,1):chosen.append((f'success_{i}',float(oks.iloc[int(j)]['time_sec'])))
    if slot_df is not None and not slot_df.empty and 'phase' in slot_df.columns:
        man=slot_df[slot_df['phase'].astype(str).str.upper()=='MANEUVERING']['time_sec'].drop_duplicates().astype(float).tolist()
        if man:
            idx=np.linspace(0,len(man)-1,min(max_maneuver,len(man))).round().astype(int)
            for i,j in enumerate(idx,1):chosen.append((f'maneuver_{i}',float(man[int(j)])))
    out=[];seen=set()
    for label,t in chosen:
        key=round(float(t),2)
        if key in seen:continue
        seen.add(key);out.append((label,float(t)))
    return out[:12]



def _slot_lookup(slots: Sequence[Dict]) -> Dict[str, Dict]:
    return {str(s.get('local_id')): s for s in slots}


def _focused_slot_candidates(slot_df: pd.DataFrame, transitions_df: pd.DataFrame, t: float, error: int, limit: int) -> List[str]:
    rows = _nearest_row(slot_df, t, tol=0.51)
    if rows.empty:
        return []
    ordered: List[str] = []
    def add(lid):
        lid=str(lid)
        if lid and lid not in ordered:
            ordered.append(lid)

    # AUX-only additions are the first suspects for over-counting.
    if 'evidence_source' in rows.columns:
        aux = rows[rows['evidence_source'].astype(str).str.upper().isin(['AUX','AUX_ADD'])]
        for lid in aux.get('local_id', pd.Series(dtype=str)).tolist(): add(lid)

    # State changes just before an error are also highly informative.
    if transitions_df is not None and not transitions_df.empty:
        recent = transitions_df[(transitions_df.get('scope','').astype(str)=='LOCAL') &
                                (transitions_df['time_sec'].astype(float) <= float(t)+1e-6) &
                                (transitions_df['time_sec'].astype(float) >= float(t)-12.0)]
        recent = recent.sort_values('time_sec', ascending=False)
        for lid in recent.get('id', pd.Series(dtype=str)).tolist(): add(lid)

    # Fall back to weakest occupied states for +errors or suspicious empty states for -errors.
    if error >= 0:
        cand = rows[rows.get('state','').astype(str).str.upper()=='OCCUPIED'].copy()
        if 'mean_detection_confidence' in cand.columns:
            cand = cand.sort_values(['mean_detection_confidence','window_hit_ratio'] if 'window_hit_ratio' in cand.columns else ['mean_detection_confidence'])
    else:
        cand = rows[rows.get('state','').astype(str).str.upper()=='EMPTY'].copy()
        if 'recent_hit_ratio' in cand.columns:
            cand = cand.sort_values('recent_hit_ratio',ascending=False)
    for lid in cand.get('local_id', pd.Series(dtype=str)).tolist(): add(lid)
    return ordered[:max(1,int(limit))]


def _make_focused_error_figure(frame: np.ndarray, t: float, error: int, gt: str, pred: str, rois: Dict,
                               slots: Sequence[Dict], slot_df: pd.DataFrame, ev_df: pd.DataFrame,
                               transitions_df: pd.DataFrame, limit: int = 4) -> np.ndarray | None:
    lids = _focused_slot_candidates(slot_df, transitions_df, t, error, limit)
    if not lids:
        return None
    lookup = _slot_lookup(slots)
    sr_all = _nearest_row(slot_df, t, tol=0.51)
    er_all = _nearest_row(ev_df, t, tol=0.51)
    panels=[]
    for lid in lids:
        s=lookup.get(lid)
        if not s: continue
        cctv=str(s.get('cctv',''))
        if cctv not in rois: continue
        warped=crop_roi(frame,rois[cctv])
        er=er_all[er_all['local_id'].astype(str)==lid] if (not er_all.empty and 'local_id' in er_all.columns) else pd.DataFrame()
        sr=sr_all[sr_all['local_id'].astype(str)==lid] if (not sr_all.empty and 'local_id' in sr_all.columns) else pd.DataFrame()
        px,py=[int(round(v)) for v in s.get('point',[warped.shape[1]/2,warped.shape[0]/2])]
        if not er.empty and all(k in er.columns for k in ['crop_x1','crop_y1','crop_x2','crop_y2']):
            rr=er.iloc[0];x1,y1,x2,y2=[int(rr[k]) for k in ['crop_x1','crop_y1','crop_x2','crop_y2']]
        else:
            halfw,halfh=160,130;x1,y1,x2,y2=px-halfw,py-halfh,px+halfw,py+halfh
        # Expand a little so the paper figure shows context around the auxiliary crop.
        ww=max(1,x2-x1);hh=max(1,y2-y1);x1-=int(0.12*ww);x2+=int(0.12*ww);y1-=int(0.12*hh);y2+=int(0.12*hh)
        x1=max(0,x1);y1=max(0,y1);x2=min(warped.shape[1],x2);y2=min(warped.shape[0],y2)
        z=warped[y1:y2,x1:x2].copy()
        if z.size==0: continue
        cv2.circle(z,(max(0,min(z.shape[1]-1,px-x1)),max(0,min(z.shape[0]-1,py-y1))),7,(0,255,255),2)
        state='?';source='?';mconf=0.0;phase='?'
        if not sr.empty:
            r=sr.iloc[0];state=str(r.get('state','?'));source=str(r.get('evidence_source','?'));phase=str(r.get('phase','?'));mconf=float(r.get('mean_detection_confidence',0.0))
        full=aux=support=seg=0;fconf=aconf=segconf=0.0;segcore=0.0
        if not er.empty:
            r=er.iloc[0];full=int(r.get('full_detected',0));aux=int(r.get('aux_crop_detected',r.get('crop_detected',0)));support=int(r.get('aux_support_scales',0));fconf=float(r.get('full_det_conf',0));aconf=float(r.get('aux_crop_det_conf',r.get('crop_det_conf',0)));seg=int(r.get('seg_detected',0));segconf=float(r.get('seg_conf',0.0));segcore=float(r.get('seg_core_overlap_ratio',0.0))
        panel=_letterbox(z,1100,620)
        bar=np.full((125,1100,3),20,np.uint8)
        gid=str(s.get('global_id',lid))
        cv2.putText(bar,f'{gid} / {lid} / {cctv}',(16,38),cv2.FONT_HERSHEY_SIMPLEX,0.82,(245,245,245),2,cv2.LINE_AA)
        cv2.putText(bar,f'{state} {phase} | F={full}({fconf:.2f}) A={aux}({aconf:.2f}) SEG={seg}({segconf:.2f}) core={segcore:.2f}',(16,82),cv2.FONT_HERSHEY_SIMPLEX,0.67,(235,235,235),2,cv2.LINE_AA)
        cv2.putText(bar,f'source={source} | AUX scales={support} | mean FULL conf={mconf:.2f}',(16,113),cv2.FONT_HERSHEY_SIMPLEX,0.58,(220,220,220),2,cv2.LINE_AA)
        panels.append(np.vstack([bar,panel]))
    if not panels: return None
    blank=np.full_like(panels[0],28)
    while len(panels)<4: panels.append(blank.copy())
    body=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:4])])
    top=np.full((110,body.shape[1],3),14,np.uint8)
    cv2.putText(top,f'Focused error | t={format_timestamp(t)} | GT={gt} PRED={pred} ERR={error}',(22,70),cv2.FONT_HERSHEY_SIMPLEX,1.05,(250,250,250),3,cv2.LINE_AA)
    return np.vstack([top,body])


def generate_research_screenshots(video_path: str, rois: Dict, slots_path: str, settings: Dict, run_dir: str | Path,
                                  learn_dir: str | Path, evidence_dir: str | Path) -> Path:
    run_dir=Path(run_dir);learn_dir=Path(learn_dir);evidence_dir=Path(evidence_dir)
    root=ensure_dir(run_dir/'screenshots');paper=ensure_dir(root/'paper_ready');setup=ensure_dir(root/'setup')
    count_path=run_dir/'selected_count_timeseries.csv';slot_path=run_dir/'selected_slot_timeseries.csv';ev_path=evidence_dir/'slot_evidence.csv'
    count_df=pd.read_csv(count_path,encoding='utf-8-sig') if count_path.exists() else pd.DataFrame()
    slot_df=pd.read_csv(slot_path,encoding='utf-8-sig') if slot_path.exists() else pd.DataFrame()
    ev_df=pd.read_csv(ev_path,encoding='utf-8-sig') if ev_path.exists() else pd.DataFrame()
    params=load_json(run_dir/'selected_state_params.json',{}) or {}
    mode=str(params.get('evidence_mode','?'))
    slots=load_slots(slots_path)

    # Preserve setup figures that are useful in methodology sections.
    copied=[]
    for c in sorted(rois.keys()):
        for suffix in ['manual_voronoi_preview.jpg','slot_overlap_preview.jpg','candidate_preview.jpg','slot_configuration_diagnostics.jpg']:
            src=learn_dir/f'{c}_{suffix}'
            if src.exists():
                dst=setup/src.name;shutil.copy2(src,dst);copied.append(dst)

    times=_select_times(count_df,slot_df)
    cap=open_video(video_path);index_lines=['Parking Slot Engine v16 automatic research screenshots','',f'Selected evidence mode: {mode}','']
    try:
        for n,(kind,t) in enumerate(times,1):
            frame=read_frame_at(cap,t)
            gt,pred,err=_count_info(count_df,t)
            panels=[_letterbox(frame,1280,720)]
            for c in sorted(rois.keys()):
                if c not in rois:
                    panels.append(np.full((720,1280,3),28,np.uint8));continue
                warped=crop_roi(frame,rois[c])
                sr=_nearest_row(slot_df[(slot_df['cctv'].astype(str)==c)] if (not slot_df.empty and 'cctv' in slot_df.columns) else pd.DataFrame(),t)
                er=_nearest_row(ev_df[(ev_df['cctv'].astype(str)==c)] if (not ev_df.empty and 'cctv' in ev_df.columns) else pd.DataFrame(),t)
                ann=_draw_warped(warped,c,slots,sr,er)
                panels.append(_letterbox(ann,1280,720))
            canvas=_panel_grid(panels,2)
            bar=np.full((105,canvas.shape[1],3),18,np.uint8)
            text=f'{kind} | t={format_timestamp(t)} | GT={gt} PRED={pred} ERR={err} | mode={mode} | F=FULL, A=multi-scale AUX, S=scale support'
            cv2.putText(bar,text,(22,68),cv2.FONT_HERSHEY_SIMPLEX,0.92,(245,245,245),3,cv2.LINE_AA)
            final=np.vstack([bar,canvas])
            fn=f'paper_{n:02d}_{kind}_{int(round(t)):04d}s_gt{gt}_pred{pred}.jpg'
            cv2.imwrite(str(paper/fn),final,[int(cv2.IMWRITE_JPEG_QUALITY),92])
            index_lines.append(f'{fn}: {kind}, {format_timestamp(t)}, GT={gt}, prediction={pred}, error={err}')
    finally:
        cap.release()
    # Focused error figures: zoom in on the slots most likely responsible for the error.
    focused=ensure_dir(root/'focused_errors')
    transitions_path=run_dir/'selected_state_transitions.csv'
    transitions_df=pd.read_csv(transitions_path,encoding='utf-8-sig') if transitions_path.exists() else pd.DataFrame()
    max_focused=int(settings.get('research_screenshots',{}).get('focused_error_count',3))
    focused_slots=int(settings.get('research_screenshots',{}).get('focused_slots_per_error',4))
    errs=count_df[(count_df.get('split','')=='TEST') & (count_df.get('error',0)!=0)].copy() if not count_df.empty else pd.DataFrame()
    if not errs.empty:
        errs['abs_sort']=errs['error'].abs();errs=errs.sort_values(['abs_sort','time_sec'],ascending=[False,True]).head(max_focused)
        cap2=open_video(video_path)
        try:
            for i,r in enumerate(errs.itertuples(),1):
                t=float(r.time_sec);frame=read_frame_at(cap2,t);gt,pred,err=_count_info(count_df,t)
                fig=_make_focused_error_figure(frame,t,int(err),gt,pred,rois,slots,slot_df,ev_df,transitions_df,focused_slots)
                if fig is None: continue
                fn=f'focused_error_{i:02d}_{int(round(t)):04d}s_gt{gt}_pred{pred}.jpg'
                cv2.imwrite(str(focused/fn),fig,[int(cv2.IMWRITE_JPEG_QUALITY),94])
                index_lines.append(f'{fn}: focused TEST error figure, {format_timestamp(t)}, GT={gt}, prediction={pred}, error={err}')
        finally:
            cap2.release()
    index_lines += ['', 'Setup figures:'] + [p.name for p in copied]
    (root/'PAPER_SCREENSHOTS_INDEX.txt').write_text('\n'.join(index_lines),encoding='utf-8')
    return root

# -----------------------------------------------------------------------------
# v15.2 ground-truth visual review package
# -----------------------------------------------------------------------------

def _draw_text_box(img: np.ndarray, lines: List[str], origin=(20,20), font_scale: float = 0.78, line_h: int = 34) -> np.ndarray:
    out=img.copy();x,y=origin
    widths=[]
    for line in lines:
        (tw,th),_=cv2.getTextSize(str(line),cv2.FONT_HERSHEY_SIMPLEX,font_scale,2);widths.append((tw,th))
    bw=max([w for w,_ in widths]+[200])+28;bh=max(50,line_h*len(lines)+18)
    cv2.rectangle(out,(x,y),(min(out.shape[1]-1,x+bw),min(out.shape[0]-1,y+bh)),(12,12,12),-1)
    yy=y+line_h
    for line in lines:
        cv2.putText(out,str(line),(x+14,yy),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(0,0,0),5,cv2.LINE_AA)
        cv2.putText(out,str(line),(x+14,yy),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(255,255,255),2,cv2.LINE_AA)
        yy+=line_h
    return out


def _raw_roi_bbox(frame: np.ndarray, roi: Dict) -> np.ndarray:
    pts=np.asarray((roi or {}).get('points',[]),dtype=np.float32)
    if pts.shape!=(4,2):
        return frame.copy()
    x1=max(0,int(np.floor(pts[:,0].min())));x2=min(frame.shape[1],int(np.ceil(pts[:,0].max()))+1)
    y1=max(0,int(np.floor(pts[:,1].min())));y2=min(frame.shape[0],int(np.ceil(pts[:,1].max()))+1)
    return frame[y1:y2,x1:x2].copy()


def _contact_sheet(entries: List[Tuple[Path,str]], cols: int, rows: int, tw: int, th: int) -> np.ndarray:
    cell_h=th+48; canvas=np.full((rows*cell_h,cols*tw,3),24,np.uint8)
    for i,(path,label) in enumerate(entries[:cols*rows]):
        img=cv2.imread(str(path));
        if img is None:continue
        thumb=_letterbox(img,tw,th);rr=i//cols;cc=i%cols
        canvas[rr*cell_h:rr*cell_h+th,cc*tw:(cc+1)*tw]=thumb
        y=rr*cell_h+th+31
        cv2.putText(canvas,label,(cc*tw+8,y),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(canvas,label,(cc*tw+8,y),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2,cv2.LINE_AA)
    return canvas


def _crop_around(img: np.ndarray, point: Tuple[float,float], half_w: int=180, half_h: int=145) -> np.ndarray:
    px,py=map(int,map(round,point));x1=max(0,px-half_w);x2=min(img.shape[1],px+half_w);y1=max(0,py-half_h);y2=min(img.shape[0],py+half_h)
    crop=img[y1:y2,x1:x2].copy()
    if crop.size:
        cv2.circle(crop,(px-x1,py-y1),10,(0,165,255),3)
    return crop


def generate_ground_truth_review(video_path: str, rois: Dict, gt_path: str, slots_path: str, settings: Dict,
                                 run_dir: str | Path, learn_dir: str | Path) -> Path:
    """Capture every GT timestamp locally and build a slim upload-oriented review subset.

    Full raw/annotated/CCTV-split frames stay in the local run folder for human GT audit.
    UPLOAD_TO_CHATGPT.zip can include only contact sheets, event/change frames and a
    periodic high-resolution review subset, keeping the upload much smaller.
    """
    from utils import load_ground_truth
    run_dir=Path(run_dir);learn_dir=Path(learn_dir);root=ensure_dir(run_dir/'ground_truth_review')
    raw_dir=ensure_dir(root/'raw');ann_dir=ensure_dir(root/'annotated')
    split_raw=ensure_dir(root/'cctv_split'/'raw');split_warp=ensure_dir(root/'cctv_split'/'warped')
    contact_dir=ensure_dir(root/'contact_sheets');event_dir=ensure_dir(root/'events');upload_dir=ensure_dir(root/'upload_review')
    cand_dir=ensure_dir(run_dir/'candidate_review');dup_dir=ensure_dir(run_dir/'duplicate_review')
    cfg=settings.get('ground_truth_review',{}) or {};quality=int(cfg.get('jpeg_quality',92));upload_stride=float(cfg.get('upload_review_stride_sec',30.0))
    dev_end=float(settings.get('dev_end_sec',900.0))
    gt=load_ground_truth(gt_path);slots=load_slots(slots_path)
    cap=open_video(video_path);index=[];raw_entries=[];upload_index=[]
    prev_occ=None;prev_unique=None;prev_cam=None
    try:
        for i,r in gt.iterrows():
            t=float(r['time_sec']);frame=read_frame_at(cap,t);sec=int(round(t));ts=str(r.get('timestamp',format_timestamp(t)))
            occ=r.get('ground_truth_occupied_space_count','?');uniq=r.get('ground_truth_unique_vehicle_count','?')
            base=f'{sec:04d}_{ts.replace(":","m")}s'
            raw_name=f'{base}.jpg';cv2.imwrite(str(raw_dir/raw_name),frame,[int(cv2.IMWRITE_JPEG_QUALITY),quality])
            camera_line=' | '.join(f'{c.upper()} {r.get(c + "_count","?")}' for c in sorted(rois.keys()))
            lines=[f'TIME {ts}',f'GT OCCUPIED {occ}',f'GT UNIQUE {uniq}',camera_line,
                   f'TAGS {r.get("tags","")} | OVERLAP {r.get("overlap_cctv","")}']
            ann=_draw_text_box(frame,lines,(22,22),1.02,44);ann_name=f'{base}_GT{occ}.jpg';cv2.imwrite(str(ann_dir/ann_name),ann,[int(cv2.IMWRITE_JPEG_QUALITY),quality])
            row={'time_sec':t,'timestamp':ts,'raw_file_local':f'ground_truth_review/raw/{raw_name}','annotated_file_local':f'ground_truth_review/annotated/{ann_name}',
                 'ground_truth_occupied_space_count':occ,'ground_truth_unique_vehicle_count':uniq,'tags':r.get('tags',''),'overlap_cctv':r.get('overlap_cctv',''),'notes':r.get('notes','')}
            for c in sorted(rois.keys()): row[f'{c}_count']=r.get(f'{c}_count','')
            warped_names=[]
            for cctv,roi in rois.items():
                rr=_raw_roi_bbox(frame,roi);ww=crop_roi(frame,roi)
                rn=f'{base}_{cctv.upper()}_RAW.jpg';wn=f'{base}_{cctv.upper()}_WARPED.jpg'
                cv2.imwrite(str(split_raw/rn),rr,[int(cv2.IMWRITE_JPEG_QUALITY),quality]);cv2.imwrite(str(split_warp/wn),ww,[int(cv2.IMWRITE_JPEG_QUALITY),quality])
                row[f'{cctv}_raw_file_local']=f'ground_truth_review/cctv_split/raw/{rn}';row[f'{cctv}_warped_file_local']=f'ground_truth_review/cctv_split/warped/{wn}'
                warped_names.append((cctv,wn))
            index.append(row);raw_entries.append((raw_dir/raw_name,f'{ts} | GT {occ}'))

            tag_text=str(r.get('tags',''));tag_tokens=set(x.strip() for x in tag_text.replace(',',';').split(';') if x.strip())
            cams=tuple(r.get(f'{c}_count','') for c in sorted(rois.keys()))
            count_change = (prev_occ is not None and str(occ)!=str(prev_occ)) or (prev_unique is not None and str(uniq)!=str(prev_unique)) or (prev_cam is not None and tuple(map(str,cams))!=tuple(map(str,prev_cam)))
            tagged=bool(tag_tokens.intersection({'21','22','23','31'}))
            event=tagged or bool(count_change)
            if event:
                reason=[]
                if tagged:reason.append('TAG_'+'_'.join(sorted(tag_tokens)))
                if count_change:reason.append('COUNT_CHANGE')
                event_name=f'{base}_EVENT_{"_".join(reason)}_GT{occ}.jpg';shutil.copy2(ann_dir/ann_name,event_dir/event_name)
            # Upload subset: every change/event plus one high-res checkpoint every N seconds in TEST.
            periodic_test = bool(t >= dev_end-1e-6 and upload_stride>0 and abs((t-dev_end) % upload_stride) < 0.51)
            if event or periodic_test:
                up_ann=f'{base}_GT{occ}.jpg';shutil.copy2(ann_dir/ann_name,upload_dir/up_ann)
                up_row={'time_sec':t,'timestamp':ts,'reason':'EVENT_OR_CHANGE' if event else 'PERIODIC_TEST','annotated_file':f'ground_truth_review/upload_review/{up_ann}'}
                # Include warped CCTV splits for these selected frames so GT can be checked at readable scale.
                for cctv,wn in warped_names:
                    src=split_warp/wn;dst=upload_dir/wn;shutil.copy2(src,dst);up_row[f'{cctv}_warped_file']=f'ground_truth_review/upload_review/{wn}'
                upload_index.append(up_row)
            prev_occ,prev_unique,prev_cam=occ,uniq,cams
    finally:
        cap.release()
    pd.DataFrame(index).to_csv(root/'gt_review_index.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(upload_index).to_csv(root/'gt_review_upload_index.csv',index=False,encoding='utf-8-sig')

    cols=max(2,int(cfg.get('contact_sheet_columns',5)));rows=max(2,int(cfg.get('contact_sheet_rows',6)));tw=int(cfg.get('contact_thumb_width',360));th=int(cfg.get('contact_thumb_height',203));per=cols*rows
    for st in range(0,len(raw_entries),per):
        chunk=raw_entries[st:st+per];sheet=_contact_sheet(chunk,cols,rows,tw,th)
        start_ts=index[st]['timestamp'];end_ts=index[min(len(index)-1,st+len(chunk)-1)]['timestamp']
        fn=f'contact_{st//per+1:02d}_{str(start_ts).replace(":","m")}_{str(end_ts).replace(":","m")}.jpg'
        cv2.imwrite(str(contact_dir/fn),sheet,[int(cv2.IMWRITE_JPEG_QUALITY),quality])

    # Candidate and confirmed duplicate review images use only operator/candidate metadata, never model count predictions.
    cand_path=learn_dir/'candidate_slots.csv'
    if cand_path.exists():
        cdf=pd.read_csv(cand_path,encoding='utf-8-sig')
        review_t=float(cfg.get('candidate_review_time_sec',450.0));cap2=open_video(video_path)
        try:
            frame=read_frame_at(cap2,review_t)
            for rr in cdf.itertuples():
                cctv=str(rr.cctv)
                if cctv not in rois:continue
                warped=crop_roi(frame,rois[cctv]);point=(float(rr.x),float(rr.y));crop=_crop_around(warped,point)
                if crop.size==0:continue
                label=f'{rr.candidate_id} | {getattr(rr,"candidate_kind","NEW")} | {cctv} | t={format_timestamp(review_t)}'
                crop=_draw_text_box(crop,[label],(8,8),0.55,28)
                cv2.imwrite(str(cand_dir/f'{rr.candidate_id}_{str(getattr(rr,"candidate_kind","NEW")).lower()}.jpg'),crop,[int(cv2.IMWRITE_JPEG_QUALITY),quality])
        finally:cap2.release()

        dups=cdf[cdf.get('candidate_kind',pd.Series(dtype=str)).astype(str).str.upper()=='DUPLICATE'] if not cdf.empty else pd.DataFrame()
        slot_lookup={str(s.get('local_id')):s for s in slots};times=[float(x) for x in cfg.get('duplicate_review_times_sec',[0,450,900,1230])]
        if not dups.empty:
            cap3=open_video(video_path)
            try:
                for dr in dups.itertuples():
                    target=slot_lookup.get(str(getattr(dr,'target_local_id','')))
                    if target is None:continue
                    for t in times:
                        frame=read_frame_at(cap3,t);src_cam=str(dr.cctv);dst_cam=str(target.get('cctv',''))
                        if src_cam not in rois or dst_cam not in rois:continue
                        a=_crop_around(crop_roi(frame,rois[dst_cam]),tuple(map(float,target.get('point',[0,0]))),180,145)
                        b=_crop_around(crop_roi(frame,rois[src_cam]),(float(dr.x),float(dr.y)),180,145)
                        a=_letterbox(a,760,500);b=_letterbox(b,760,500);canvas=np.hstack([a,b]);bar=np.full((105,canvas.shape[1],3),16,np.uint8)
                        cv2.putText(bar,f"SAME PHYSICAL SLOT REVIEW | {target.get('local_id')} ({dst_cam})  <->  {dr.candidate_id} ({src_cam}) | t={format_timestamp(t)}",(20,42),cv2.FONT_HERSHEY_SIMPLEX,0.74,(0,0,0),5,cv2.LINE_AA)
                        cv2.putText(bar,f"SAME PHYSICAL SLOT REVIEW | {target.get('local_id')} ({dst_cam})  <->  {dr.candidate_id} ({src_cam}) | t={format_timestamp(t)}",(20,42),cv2.FONT_HERSHEY_SIMPLEX,0.74,(255,255,255),2,cv2.LINE_AA)
                        cv2.putText(bar,'Operator-confirmed DUPLICATE: global occupied count must not +1; review local CCTV counts only.',(20,82),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,0,0),4,cv2.LINE_AA)
                        cv2.putText(bar,'Operator-confirmed DUPLICATE: global occupied count must not +1; review local CCTV counts only.',(20,82),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2,cv2.LINE_AA)
                        final=np.vstack([bar,canvas]);fn=f'{dr.candidate_id}_vs_{target.get("local_id")}_{int(round(t)):04d}s.jpg';cv2.imwrite(str(dup_dir/fn),final,[int(cv2.IMWRITE_JPEG_QUALITY),quality])
            finally:cap3.release()

    (root/'README_GT_REVIEW.txt').write_text(
        'Full local review set: raw/, annotated/, cctv_split/raw/, cctv_split/warped/.\n'
        'annotated/ shows only human-entered GT values; model predictions are intentionally excluded to avoid review bias.\n'
        'events/ is generated from GT tags 21/22/23/31 OR any count change between adjacent GT rows.\n'
        'upload_review/ is a smaller high-resolution subset: all events/changes plus periodic TEST checkpoints.\n'
        'gt_review_index.csv references LOCAL full-review files. gt_review_upload_index.csv references only the slim upload subset.\n',encoding='utf-8')
    return root
