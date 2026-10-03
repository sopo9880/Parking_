# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from detector import VehicleDetector
from segmentation import (
    VehicleSegmenter,
    choose_best_segment,
    conditioned_parking_crop,
    image_quality_metrics,
    build_relative_quality_thresholds,
    classify_quality_condition,
)
from utils import box_center, crop_roi, ensure_dir, format_timestamp, open_video, read_frame_at


def _clip_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def degrade_low_res(img: np.ndarray, scale: float = 0.35) -> np.ndarray:
    h, w = img.shape[:2]
    sw = max(12, int(round(w * float(scale))))
    sh = max(12, int(round(h * float(scale))))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_sun_glare(img: np.ndarray, strength: float = 0.50) -> np.ndarray:
    """Synthetic overexposure/glare stress. This is a stress test, not a real-light GT label."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = 0.62 * w, 0.28 * h
    rx, ry = max(1.0, 0.62 * w), max(1.0, 0.55 * h)
    g = np.exp(-(((xx-cx)/rx)**2 + ((yy-cy)/ry)**2) * 2.2)
    gain = 1.0 + float(strength) * g[..., None]
    bias = 80.0 * float(strength) * g[..., None]
    out = img.astype(np.float32) * gain + bias
    return _clip_u8(out)


def degrade_monitor_stripes(img: np.ndarray, strength: float = 0.22, period: int = 5, phase: float = 0.0) -> np.ndarray:
    """Synthetic re-recorded-monitor stripe/moire-like stress with controllable phase."""
    h, w = img.shape[:2]
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    p = max(3, int(period))
    stripe = np.sin(2.0 * np.pi * y / p + float(phase)) + 0.45 * np.sin(2.0 * np.pi * x / (p + 2) + 0.63 * float(phase))
    stripe = stripe[..., None]
    out = img.astype(np.float32) + stripe * (255.0 * float(strength) * 0.28)
    return _clip_u8(out)


def _crop_geometry(row: pd.Series, slot: Dict, warped: np.ndarray) -> Tuple[int, int, int, int, Tuple[float, float], float]:
    try:
        x1 = int(row.get('crop_x1')); y1 = int(row.get('crop_y1')); x2 = int(row.get('crop_x2')); y2 = int(row.get('crop_y2'))
    except Exception:
        px, py = map(float, slot.get('point', [warped.shape[1]/2, warped.shape[0]/2]))
        x1, x2 = int(px-120), int(px+120); y1, y2 = int(py-140), int(py+140)
    x1 = max(0, min(warped.shape[1]-1, x1)); x2 = max(x1+1, min(warped.shape[1], x2))
    y1 = max(0, min(warped.shape[0]-1, y1)); y2 = max(y1+1, min(warped.shape[0], y2))
    px, py = map(float, slot.get('point', [0, 0]))
    lp = (px-x1, py-y1)
    rad = float(row.get('crop_core_radius_px', max(10.0, 0.15*min(y2-y1, x2-x1))))
    return x1, y1, x2, y2, lp, rad


def _det_support(dets, point: Tuple[float, float], radius: float) -> Tuple[int, float]:
    if not dets:
        return 0, 0.0
    px, py = point
    best = 0.0
    for d in dets:
        cx, cy = box_center(d.box)
        dist = float(np.hypot(cx-px, cy-py))
        inside = (d.x1 <= px <= d.x2 and d.y1 <= py <= d.y2)
        if inside or dist <= max(18.0, 1.35*radius):
            best = max(best, float(d.conf))
    return int(best > 0.0), float(best)


def _seg_support(dets, min_core: float, positive_conf: float, positive_core: float):
    best = choose_best_segment(dets, min_core)
    if best is None:
        return 0, 0.0, 0.0, 0.0, 0
    pos = bool(float(best.conf) >= positive_conf and (int(best.point_covered) or float(best.core_overlap_ratio) >= positive_core))
    return int(pos), float(best.conf), float(best.mask_ratio), float(best.core_overlap_ratio), int(best.point_covered)


def _pick_hard_slots(evidence: pd.DataFrame, slots: Sequence[Dict], cfg: Dict) -> List[str]:
    """Pick bounded hard slots, with operator-forced slots guaranteed to survive max_slots truncation."""
    per_cam = max(1, int(cfg.get('hard_slots_per_cctv', 2)))
    valid = {str(s.get('local_id','')) for s in slots}
    forced = [str(x) for x in cfg.get('force_slots', []) if str(x) in valid]
    # Forced slots are first-class experimental controls. Never append them after truncation.
    out: List[str] = []
    for lid in forced:
        if lid not in out:
            out.append(lid)
    max_slots = max(len(out), max(1, int(cfg.get('max_slots', max(len(out), 6)))))
    if evidence.empty:
        for s in slots:
            lid=str(s.get('local_id',''))
            if lid and lid not in out:
                out.append(lid)
            if len(out)>=max_slots: break
        return out[:max_slots]
    tmp = evidence.copy()
    tmp['full_detected'] = pd.to_numeric(tmp.get('full_detected', 0), errors='coerce').fillna(0)
    tmp['full_det_conf'] = pd.to_numeric(tmp.get('full_det_conf', 0.0), errors='coerce').fillna(0.0)
    g = tmp.groupby(['cctv','local_id'], as_index=False).agg(full_hit_ratio=('full_detected','mean'), mean_conf=('full_det_conf','mean'))
    g['difficulty'] = (1.0-g['full_hit_ratio']) + 0.35*(1.0-g['mean_conf'].clip(0,1))
    for cctv in sorted(g['cctv'].astype(str).unique()):
        part = g[g['cctv'].astype(str)==cctv].sort_values(['difficulty','full_hit_ratio'], ascending=[False, True])
        for lid in part['local_id'].astype(str).tolist():
            if lid not in out:
                out.append(lid)
            if len(out)>=max_slots: break
        if len(out)>=max_slots: break
    return out[:max_slots]


def _pick_times(evidence: pd.DataFrame, cfg: Dict) -> List[float]:
    max_times = max(4, int(cfg.get('max_timestamps', 12)))
    if evidence.empty:
        return []
    ts = sorted(set(float(x) for x in evidence['time_sec'].dropna().tolist()))
    if len(ts) <= max_times:
        return ts
    idx = np.linspace(0, len(ts)-1, max_times).round().astype(int)
    return sorted({ts[int(i)] for i in idx})



def _diverse_take(pool: pd.DataFrame, n: int, score_col: str | None = None, descending: bool = True) -> pd.DataFrame:
    """Deterministic, slot-diverse sampling across time. No randomness -> reproducible cache signatures."""
    if n <= 0 or pool is None or pool.empty:
        return pd.DataFrame(columns=list(pool.columns) if pool is not None else [])
    p = pool.copy().sort_values(['cctv','local_id','time_sec']).reset_index(drop=True)
    groups = []
    for _, g in p.groupby(['cctv','local_id'], sort=True):
        g = g.sort_values('time_sec').reset_index(drop=True)
        groups.append(g)
    if not groups:
        return p.head(0)
    per = max(1, int(np.ceil(float(n) / float(len(groups)))))
    picks=[]
    for g in groups:
        k=min(per,len(g))
        idx=np.linspace(0,len(g)-1,k).round().astype(int)
        picks.append(g.iloc[sorted(set(int(x) for x in idx))])
    cand=pd.concat(picks,ignore_index=True).drop_duplicates(['time_sec','local_id'])
    if score_col and score_col in cand.columns:
        cand=cand.sort_values([score_col,'time_sec'],ascending=[not descending,True])
    if len(cand)>n:
        # Preserve temporal diversity when trimming.
        idx=np.linspace(0,len(cand)-1,n).round().astype(int)
        cand=cand.iloc[sorted(set(int(x) for x in idx))]
    if len(cand)<n:
        used=set((round(float(r.time_sec),6),str(r.local_id)) for r in cand.itertuples())
        rest=p[~p.apply(lambda r:(round(float(r['time_sec']),6),str(r['local_id'])) in used,axis=1)]
        if score_col and score_col in rest.columns:
            rest=rest.sort_values([score_col,'time_sec'],ascending=[not descending,True])
        cand=pd.concat([cand,rest.head(n-len(cand))],ignore_index=True)
    return cand.head(n).copy()


def _pick_balanced_samples(evidence: pd.DataFrame, slots: Sequence[Dict], cfg: Dict) -> pd.DataFrame:
    """Build a GT-review set that intentionally contains likely OCCUPIED and likely EMPTY samples.

    This only controls *what the human labels*. Pseudo hints are never used as GT.
    Forced hard slots are retained, but they no longer dominate the whole dataset.
    """
    if evidence is None or evidence.empty:
        return pd.DataFrame(columns=['time_sec','cctv','local_id','sampling_stratum'])
    df=evidence.copy()
    for c in ['full_detected','full_det_conf','crop_detected','crop_det_conf','recent_full_hit_ratio','visual_diff_initial']:
        if c not in df.columns: df[c]=0.0
        df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
    target=max(24,int(cfg.get('balanced_gt_samples',96)))
    forced_each=max(1,int(cfg.get('forced_samples_per_slot',3)))
    forced_ids=[str(x) for x in cfg.get('force_slots',[]) if str(x)]
    selected=[]
    used=set()
    def add(part: pd.DataFrame, stratum: str):
        nonlocal selected,used
        if part is None or part.empty: return
        for _,r in part.iterrows():
            key=(round(float(r['time_sec']),6),str(r['local_id']))
            if key in used: continue
            d=r.to_dict(); d['sampling_stratum']=stratum
            selected.append(d); used.add(key)
    # Preserve representative rows from operator-important slots.
    for lid in forced_ids:
        g=df[df['local_id'].astype(str)==lid].sort_values('time_sec')
        if g.empty: continue
        idx=np.linspace(0,len(g)-1,min(forced_each,len(g))).round().astype(int)
        add(g.iloc[sorted(set(int(x) for x in idx))], 'FORCED_HARD_SLOT')
    remaining=max(0,target-len(selected))
    # Count forced pseudo-balance so extras compensate rather than worsen imbalance.
    forced_occ=0;forced_empty=0
    for r in selected:
        if int(r.get('full_detected',0))>0 or (int(r.get('crop_detected',0))>0 and float(r.get('crop_det_conf',0))>=0.12): forced_occ+=1
        elif int(r.get('full_detected',0))==0 and int(r.get('crop_detected',0))==0: forced_empty+=1
    half=target//2
    need_occ=max(0,half-forced_occ)
    need_empty=max(0,half-forced_empty)
    # Strong occupied candidates: confident full detector or supported crop evidence.
    occ=df[((df['full_detected']>0)&(df['full_det_conf']>=float(cfg.get('gt_occ_full_conf',0.25)))) |
           ((df['crop_detected']>0)&(df['crop_det_conf']>=float(cfg.get('gt_occ_crop_conf',0.25))))].copy()
    occ['sample_score']=np.maximum(occ['full_det_conf'],occ['crop_det_conf'])
    # Strong empty candidates: no detector support, no recent full history, preferably slots that began EMPTY.
    empty=df[(df['full_detected']<=0)&(df['crop_detected']<=0)&
             (df['recent_full_hit_ratio']<=float(cfg.get('gt_empty_recent_hit_max',0.10)))].copy()
    empty['sample_score']=1.0-empty['visual_diff_initial'].clip(0,1)
    empty['initial_empty_priority']=(empty['initial_state'].astype(str).str.upper()=='EMPTY').astype(int)
    empty=empty.sort_values(['initial_empty_priority','sample_score'],ascending=[False,False])
    if selected:
        mask_occ=~occ.apply(lambda r:(round(float(r['time_sec']),6),str(r['local_id'])) in used,axis=1);occ=occ[mask_occ]
        mask_emp=~empty.apply(lambda r:(round(float(r['time_sec']),6),str(r['local_id'])) in used,axis=1);empty=empty[mask_emp]
    add(_diverse_take(occ,need_occ,'sample_score',True),'LIKELY_OCCUPIED')
    add(_diverse_take(empty,need_empty,'sample_score',True),'LIKELY_EMPTY')
    # Fill any shortfall without changing deterministic behavior.
    remaining=max(0,target-len(selected))
    if remaining:
        rest=df[~df.apply(lambda r:(round(float(r['time_sec']),6),str(r['local_id'])) in used,axis=1)].copy()
        add(_diverse_take(rest,remaining,None,True),'DIVERSITY_FILL')
    out=pd.DataFrame(selected)
    if out.empty: return out
    return out.sort_values(['time_sec','cctv','local_id']).head(target).reset_index(drop=True)

def _augment_samples_for_human_balance(evidence: pd.DataFrame, base_samples: pd.DataFrame, cfg: Dict, gt_labels: pd.DataFrame) -> pd.DataFrame:
    """v16.2 preserve existing human GT and add only the missing class.

    v16 initially sampled 50/50 by model hints, but human review showed 81 occupied / 15 empty.
    Model hints are not trustworthy enough to declare balance. Once human labels exist, keep
    those exact sample IDs and supplement likely candidates from the minority class until the
    *human* class ratio approaches the requested target. Existing labels are never discarded or
    relabeled automatically.
    """
    if evidence is None or evidence.empty or gt_labels is None or gt_labels.empty:
        return base_samples
    df=evidence.copy()
    for c in ['full_detected','full_det_conf','crop_detected','crop_det_conf','recent_full_hit_ratio','visual_diff_initial']:
        if c not in df.columns: df[c]=0.0
        df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
    df['_sample_id']=[f"{str(c)}_{str(l)}_{int(round(float(t))):04d}s" for c,l,t in zip(df['cctv'],df['local_id'],df['time_sec'])]
    old=gt_labels.copy(); old['occupied_gt']=old['occupied_gt'].map(_canonical_gt_label).astype('string')
    old_ids=set(old['sample_id'].astype(str))
    persisted=df[df['_sample_id'].astype(str).isin(old_ids)].copy()
    # One row per saved sample ID; evidence normally already has that property.
    persisted=persisted.drop_duplicates('_sample_id')
    persisted['sampling_stratum']='PERSISTED_HUMAN_GT'
    if persisted.empty:
        return base_samples

    matched_ids=set(persisted['_sample_id'].astype(str))
    matched_old=old[old['sample_id'].astype(str).isin(matched_ids)].copy()
    labels_num=_gt_label_numeric_series(matched_old['occupied_gt'])
    occ=int((labels_num==1).sum()); emp=int((labels_num==0).sum()); known=occ+emp
    target_minor=float(cfg.get('human_balance_target_minority_fraction',0.40))
    target_minor=max(0.25,min(0.49,target_minor)); max_new=max(0,int(cfg.get('human_balance_max_new_per_run',48)))
    supplement=pd.DataFrame()
    needed=0; minority='NONE'
    if known>0 and min(occ,emp)/known < target_minor:
        if emp<occ:
            minority='EMPTY'
            # Solve (emp+n)/(known+n) >= target_minor.
            needed=int(np.ceil(max(0.0,(target_minor*known-emp)/max(1e-9,1.0-target_minor))))
            pool=df[(df['full_detected']<=0)&(df['crop_detected']<=0)&
                    (df['recent_full_hit_ratio']<=float(cfg.get('gt_empty_recent_hit_max',0.10)))].copy()
            pool['sample_score']=(1.0-pool['visual_diff_initial'].clip(0,1))
            if 'initial_state' in pool.columns:
                pool['initial_empty_priority']=(pool['initial_state'].astype(str).str.upper()=='EMPTY').astype(int)
                pool=pool.sort_values(['initial_empty_priority','sample_score'],ascending=[False,False])
        else:
            minority='OCCUPIED'
            needed=int(np.ceil(max(0.0,(target_minor*known-occ)/max(1e-9,1.0-target_minor))))
            pool=df[((df['full_detected']>0)&(df['full_det_conf']>=float(cfg.get('gt_occ_full_conf',0.25)))) |
                    ((df['crop_detected']>0)&(df['crop_det_conf']>=float(cfg.get('gt_occ_crop_conf',0.25))))].copy()
            pool['sample_score']=np.maximum(pool['full_det_conf'],pool['crop_det_conf'])
            pool=pool.sort_values(['sample_score','time_sec'],ascending=[False,True])
        needed=min(max_new,needed)
        used_ids=set(persisted['_sample_id'].astype(str))
        pool=pool[~pool['_sample_id'].astype(str).isin(used_ids)].copy()
        if needed>0 and not pool.empty:
            supplement=_diverse_take(pool,needed,'sample_score',True)
            supplement['sampling_stratum']=f'HUMAN_BALANCE_REPAIR_{minority}'
    out=pd.concat([persisted.drop(columns=['_sample_id'],errors='ignore'),supplement.drop(columns=['_sample_id'],errors='ignore')],ignore_index=True,sort=False)
    if out.empty:
        return base_samples
    out=out.drop_duplicates(['time_sec','local_id']).sort_values(['time_sec','cctv','local_id']).reset_index(drop=True)
    out.attrs['human_balance_before']={'occupied':occ,'empty':emp,'known':known,'minority':minority,'supplement_requested':needed,'supplement_added':int(len(supplement))}
    return out


def _draw_sample_panel(images: Dict[str, np.ndarray], title: str, out_path: Path, quality: int = 92):
    keys = list(images)
    if not keys:
        return
    thumbs=[]
    for k in keys:
        img=images[k]
        h,w=img.shape[:2]; target_w=340; target_h=max(190,int(round(h*target_w/max(1,w))))
        x=cv2.resize(img,(target_w,target_h),interpolation=cv2.INTER_CUBIC)
        bar=np.full((54,target_w,3),18,np.uint8)
        cv2.putText(bar,k,(10,36),cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,255,255),2,cv2.LINE_AA)
        thumbs.append(np.vstack([bar,x]))
    mh=max(x.shape[0] for x in thumbs)
    thumbs=[cv2.copyMakeBorder(x,0,mh-x.shape[0],0,0,cv2.BORDER_CONSTANT,value=(18,18,18)) for x in thumbs]
    canvas=np.hstack(thumbs)
    top=np.full((72,canvas.shape[1],3),10,np.uint8)
    cv2.putText(top,title,(18,46),cv2.FONT_HERSHEY_SIMPLEX,0.84,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(str(out_path),np.vstack([top,canvas]),[int(cv2.IMWRITE_JPEG_QUALITY),int(quality)])


def _auto_real_condition(q: Dict[str,float], cfg: Dict, thresholds: Dict[str,float] | None = None) -> str:
    if thresholds:
        return classify_quality_condition(q, thresholds)[0]
    # Legacy fallback only; v15.4 normally supplies per-CCTV percentile thresholds.
    if float(q.get('highlight_ratio',0.0)) >= float(cfg.get('auto_highlight_ratio',0.10)):
        return 'SUN_GLARE'
    if float(q.get('stripe_energy_ratio',0.0)) >= float(cfg.get('auto_stripe_energy_ratio',0.43)):
        return 'MONITOR_STRIPES'
    if float(q.get('sharpness_laplacian_var',9999.0)) <= float(cfg.get('auto_low_sharpness',55.0)):
        return 'LOW_RES'
    return 'CLEAN'


def _read_slot_history(cap, t: float, cctv: str, rois: Dict, rect: Tuple[int,int,int,int], history_frames: int, history_step_sec: float) -> List[np.ndarray]:
    x1,y1,x2,y2=rect
    frames=[]
    # causal only: oldest -> current, never future frames
    for k in reversed(range(max(1,int(history_frames)))):
        tt=max(0.0,float(t)-float(k)*float(history_step_sec))
        fr=read_frame_at(cap,tt)
        if cctv not in rois:
            continue
        warped=crop_roi(fr,rois[cctv])
        xx1=max(0,min(warped.shape[1]-1,x1));xx2=max(xx1+1,min(warped.shape[1],x2))
        yy1=max(0,min(warped.shape[0]-1,y1));yy2=max(yy1+1,min(warped.shape[0],y2))
        crop=warped[yy1:yy2,xx1:xx2].copy()
        if crop.size:
            frames.append(crop)
    return frames


def _write_gt_review_image(crop: np.ndarray, point: Tuple[float,float], sample_id: str, title: str, out_path: Path, quality: int):
    img=crop.copy(); px=int(round(point[0]));py=int(round(point[1]))
    cv2.circle(img,(max(0,min(img.shape[1]-1,px)),max(0,min(img.shape[0]-1,py))),8,(0,255,255),2,cv2.LINE_AA)
    scale=max(1.0, 900/max(1,img.shape[1])); view=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    bar=np.full((92,view.shape[1],3),16,np.uint8)
    cv2.putText(bar,sample_id,(16,36),cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(bar,title,(16,72),cv2.FONT_HERSHEY_SIMPLEX,0.60,(235,235,235),2,cv2.LINE_AA)
    cv2.imwrite(str(out_path),np.vstack([bar,view]),[int(cv2.IMWRITE_JPEG_QUALITY),int(quality)])


def _canonical_gt_label(value) -> str:
    """Return the persisted human label O/E/U while accepting legacy 1/0/blank files."""
    if value is None or pd.isna(value):
        return 'U'
    text=str(value).strip().upper()
    if text in {'1','1.0','O','OCCUPIED','TRUE','T','YES','Y'}:
        return 'O'
    if text in {'0','0.0','E','EMPTY','FALSE','F','NO','N'}:
        return 'E'
    if text in {'','U','UNKNOWN','?','<NA>','NAN','NONE'}:
        return 'U'
    return 'U'


def _gt_label_numeric_series(series: pd.Series) -> pd.Series:
    """Map O/E to 1/0 and U to NaN. Used only for metrics, never for persistence."""
    def conv(value):
        label=_canonical_gt_label(value)
        if label=='O': return 1.0
        if label=='E': return 0.0
        return np.nan
    return series.map(conv).astype(float)


def _load_gt_labels(path: str | Path | None) -> pd.DataFrame:
    p=Path(path) if path else None
    if not p or not p.is_file():
        return pd.DataFrame(columns=['sample_id','occupied_gt','notes'])
    try:
        df=pd.read_csv(p,encoding='utf-8-sig')
    except Exception:
        return pd.DataFrame(columns=['sample_id','occupied_gt','notes'])
    if 'sample_id' not in df.columns:
        return pd.DataFrame(columns=['sample_id','occupied_gt','notes'])
    if 'occupied_gt' not in df.columns: df['occupied_gt']='U'
    if 'notes' not in df.columns: df['notes']=''
    df['occupied_gt']=df['occupied_gt'].map(_canonical_gt_label).astype('string')
    df['notes']=df['notes'].fillna('').astype('string')
    return df[['sample_id','occupied_gt','notes']].copy()


def _binary_metric_row(condition: str, preprocess: str, method: str, gt: pd.Series, pred: pd.Series) -> Dict:
    gt=gt.astype(int); pred=pred.astype(int)
    tp=int(((gt==1)&(pred==1)).sum()); tn=int(((gt==0)&(pred==0)).sum())
    fp=int(((gt==0)&(pred==1)).sum()); fn=int(((gt==1)&(pred==0)).sum())
    precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn); specificity=tn/max(1,tn+fp)
    f1=(2*precision*recall/max(1e-9,precision+recall)) if (tp+fp+fn)>0 else 0.0
    return {
        'condition':condition,'preprocess':preprocess,'method':method,'N_labeled':len(gt),
        'N_occupied':int((gt==1).sum()),'N_empty':int((gt==0).sum()),
        'accuracy':float((pred==gt).mean()),'precision':float(precision),'f1':float(f1),
        'false_empty_rate':float(fn/max(1,int((gt==1).sum()))),
        'false_occupied_rate':float(fp/max(1,int((gt==0).sum()))),
        'occupied_recall':float(recall) if int((gt==1).sum()) else np.nan,
        'empty_specificity':float(specificity) if int((gt==0).sum()) else np.nan,
        'TP':tp,'TN':tn,'FP':fp,'FN':fn,
    }


def _write_ground_truth_metrics(sdf: pd.DataFrame, gt_labels: pd.DataFrame, out_dir: Path):
    if gt_labels.empty:
        pd.DataFrame([{'N_labeled':0,'note':'Label the balanced robustness GT with the GUI, then rerun. Synthetic variants inherit CLEAN sample GT.'}]).to_csv(out_dir/'degradation_ground_truth_metrics.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{'N_labeled':0,'note':'No balanced GT yet.'}]).to_csv(out_dir/'degradation_fusion_metrics.csv',index=False,encoding='utf-8-sig')
        return
    lab=gt_labels.copy(); lab['occupied_gt']=_gt_label_numeric_series(lab['occupied_gt'])
    lab=lab[lab['occupied_gt'].isin([0,1])]
    if lab.empty:
        pd.DataFrame([{'N_labeled':0,'note':'No 0/1 robustness GT labels yet.'}]).to_csv(out_dir/'degradation_ground_truth_metrics.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{'N_labeled':0,'note':'No balanced GT yet.'}]).to_csv(out_dir/'degradation_fusion_metrics.csv',index=False,encoding='utf-8-sig')
        return
    merged=sdf.merge(lab[['sample_id','occupied_gt']],on='sample_id',how='left')
    merged['fusion_positive']=((merged['detector_positive'].astype(int)>0)|(merged['seg_positive'].astype(int)>0)).astype(int)
    rows=[]
    for (cond,prep),part in merged.groupby(['condition','preprocess']):
        p=part[_gt_label_numeric_series(part['occupied_gt']).isin([0,1])].copy()
        if p.empty: continue
        gt=p['occupied_gt'].astype(int)
        for method,col in [('YOLO','detector_positive'),('SEG','seg_positive'),('FUSION_OR','fusion_positive')]:
            rows.append(_binary_metric_row(str(cond),str(prep),method,gt,p[col].astype(int)))
    pd.DataFrame(rows).to_csv(out_dir/'degradation_ground_truth_metrics.csv',index=False,encoding='utf-8-sig')

    # Cross-preprocess fusion. This is the table used to decide whether extra sensors
    # actually improve recall *without* creating too many false occupied slots.
    frows=[]
    for cond,part in merged.groupby('condition'):
        p=part[_gt_label_numeric_series(part['occupied_gt']).isin([0,1])].copy()
        if p.empty: continue
        piv=p.pivot_table(index=['sample_id','occupied_gt'],columns='preprocess',values=['detector_positive','seg_positive'],aggfunc='max',fill_value=0).reset_index()
        if piv.empty: continue
        def col(metric,prep):
            key=(metric,prep)
            return piv[key].astype(int) if key in piv.columns else pd.Series(np.zeros(len(piv),dtype=int),index=piv.index)
        gt=piv[('occupied_gt','')].astype(int) if ('occupied_gt','') in piv.columns else piv['occupied_gt'].astype(int)
        dr=col('detector_positive','RAW'); da=col('detector_positive','ADAPTIVE')
        sr=col('seg_positive','RAW'); sa=col('seg_positive','ADAPTIVE')
        methods={
            'YOLO_RAW_OR_ADAPTIVE': ((dr>0)|(da>0)).astype(int),
            'SEG_RAW_OR_ADAPTIVE': ((sr>0)|(sa>0)).astype(int),
            'FUSION_ANY_4': ((dr+da+sr+sa)>0).astype(int),
            'FUSION_2_OF_4': ((dr+da+sr+sa)>=2).astype(int),
            'FUSION_CONSERVATIVE': ((dr>0)|(da>0)|((sr>0)&(sa>0))).astype(int),
        }
        # Stripe rule deliberately excludes temporal-clean SEG; temporal median is a YOLO recovery path.
        if str(cond).upper()=='MONITOR_STRIPES':
            methods['FUSION_CONDITION_RULED']=((dr>0)|(da>0)|(sr>0)).astype(int)
        elif str(cond).upper()=='CLEAN':
            methods['FUSION_CONDITION_RULED']=((dr>0)|(sr>0)).astype(int)
        else:
            methods['FUSION_CONDITION_RULED']=((dr>0)|(da>0)|((sr>0)&(sa>0))).astype(int)
        for name,pred in methods.items():
            frows.append(_binary_metric_row(str(cond),'CROSS_PREPROCESS',name,gt,pred))
    pd.DataFrame(frows).to_csv(out_dir/'degradation_fusion_metrics.csv',index=False,encoding='utf-8-sig')

def run_degradation_robustness_experiment(video_path: str, rois: Dict[str,Sequence[int]], slots_path: str,
                                          settings: Dict, evidence: pd.DataFrame, output_dir: str,
                                          progress=None) -> Dict:
    """v16 balanced degradation robustness experiment.

    * Human GT review set is intentionally balanced toward likely occupied/empty samples.
    * Synthetic degradation is still a stress test; only the human CLEAN label is GT.
    * Stripe temporal median is a YOLO recovery path. SEG keeps the RAW stripe image.
    * Low-resolution adaptive input uses causal multi-frame aggregation before conservative upsample.
    * Raw and adaptive inputs remain parallel evidence; adaptive never replaces the raw view.
    """
    from slot_engine import load_slots

    cfg=dict(settings.get('robustness_experiment',{}) or {})
    out_dir=ensure_dir(output_dir)
    if not bool(cfg.get('enabled',True)):
        pd.DataFrame([{'enabled':False}]).to_csv(out_dir/'degradation_robustness_summary.csv',index=False,encoding='utf-8-sig')
        return {'enabled':False,'rows':0}

    slots=load_slots(slots_path); slot_lookup={str(s['local_id']):s for s in slots}
    sample_df=_pick_balanced_samples(evidence,slots,cfg)
    existing_gt=_load_gt_labels(cfg.get('gt_file'))
    if not existing_gt.empty:
        sample_df=_augment_samples_for_human_balance(evidence,sample_df,cfg,existing_gt)
    balance_plan=dict(sample_df.attrs.get('human_balance_before',{})) if hasattr(sample_df,'attrs') else {}
    if sample_df.empty:
        pd.DataFrame([{'enabled':True,'rows':0,'note':'no balanced samples'}]).to_csv(out_dir/'degradation_robustness_summary.csv',index=False,encoding='utf-8-sig')
        return {'enabled':True,'rows':0}
    sample_df[['time_sec','timestamp','cctv','local_id','sampling_stratum'] if 'timestamp' in sample_df.columns else ['time_sec','cctv','local_id','sampling_stratum']].to_csv(
        out_dir/'robustness_sample_manifest.csv',index=False,encoding='utf-8-sig')
    if balance_plan:
        (out_dir/'HUMAN_GT_BALANCE_PLAN.txt').write_text(
            'Parking Slot Engine v16.2 human-GT balance repair\n\n' +
            '\n'.join(f'{k}={v}' for k,v in balance_plan.items()) +
            '\n\nExisting O/E labels are preserved. Only newly selected minority-class candidates need review.\n',encoding='utf-8')

    seg_cfg=dict(settings.get('segmentation_assist',{}) or {})
    segmenter=VehicleSegmenter(seg_cfg); segmenter._load()
    det_cfg=dict(cfg.get('detector',{}) or {}) or {'model':'yolov8m.pt','imgsz':640,'conf':0.05,'vehicle_class_ids':[2,3,5,7],'tiling':False}
    detector=VehicleDetector(det_cfg); detector._load()

    min_core=float(seg_cfg.get('min_core_overlap_ratio',0.08)); pos_conf=float(seg_cfg.get('positive_conf',0.10)); pos_core=float(seg_cfg.get('positive_core_overlap_ratio',0.12))
    batch=max(1,int(cfg.get('batch_size',12))); quality=int(cfg.get('jpeg_quality',92)); panel_limit=max(0,int(cfg.get('sample_panels',6)))
    history_frames=max(3,int(cfg.get('temporal_clean_frames',5))); history_step=float(cfg.get('temporal_clean_step_sec',1.0))
    conditions=['CLEAN','LOW_RES','SUN_GLARE','MONITOR_STRIPES']

    cap=open_video(video_path)
    samples=[];det_jobs=[];seg_jobs=[];meta=[];panels=[];gt_rows=[]
    gt_review_dir=ensure_dir(out_dir/'gt_review'); clean_dir=ensure_dir(gt_review_dir/'clean_samples')
    grouped={float(t):g.copy() for t,g in sample_df.groupby('time_sec',sort=True)}
    total=max(1,len(sample_df));n=0
    try:
        for t,gsel in grouped.items():
            frame=read_frame_at(cap,float(t)); warped_cache={c:crop_roi(frame,rois[c]) for c in rois}
            for _,sel in gsel.iterrows():
                lid=str(sel['local_id']);slot=slot_lookup.get(lid)
                if slot is None: continue
                cctv=str(slot.get('cctv',''))
                if cctv not in warped_cache: continue
                warped=warped_cache[cctv]; row=pd.Series(sel)
                x1,y1,x2,y2,point,radius=_crop_geometry(row,slot,warped);crop=warped[y1:y2,x1:x2].copy()
                if crop.size==0: continue
                sample_id=f'{cctv}_{lid}_{int(round(float(t))):04d}s';clean_name=f'{sample_id}.jpg';clean_path=clean_dir/clean_name
                if not clean_path.exists():
                    _write_gt_review_image(crop,point,sample_id,f't={format_timestamp(t)} | O/1=occupied, E/0=empty, U=unknown',clean_path,quality)
                gt_rows.append({'sample_id':sample_id,'time_sec':float(t),'timestamp':format_timestamp(t),'cctv':cctv,'local_id':lid,
                                'sampling_stratum':str(sel.get('sampling_stratum','')),'occupied_gt':'','notes':'',
                                'image_path':str(Path('clean_samples')/clean_name).replace('\\','/')})

                hist=_read_slot_history(cap,float(t),cctv,rois,(x1,y1,x2,y2),history_frames,history_step)
                base={
                    'CLEAN':crop,
                    'LOW_RES':degrade_low_res(crop,float(cfg.get('low_res_scale',0.35))),
                    'SUN_GLARE':degrade_sun_glare(crop,float(cfg.get('sun_glare_strength',0.50))),
                    'MONITOR_STRIPES':degrade_monitor_stripes(crop,float(cfg.get('stripe_strength',0.22)),int(cfg.get('stripe_period_px',5)),phase=0.0),
                }
                low_hist=[degrade_low_res(h,float(cfg.get('low_res_scale',0.35))) for h in hist]
                stripe_hist=[degrade_monitor_stripes(h,float(cfg.get('stripe_strength',0.22)),int(cfg.get('stripe_period_px',5)),phase=0.9*hi) for hi,h in enumerate(hist)]
                det_ad={
                    'CLEAN':crop,
                    'LOW_RES':conditioned_parking_crop(base['LOW_RES'],'LOW_RES',temporal_frames=low_hist),
                    'SUN_GLARE':conditioned_parking_crop(base['SUN_GLARE'],'SUN_GLARE'),
                    'MONITOR_STRIPES':conditioned_parking_crop(base['MONITOR_STRIPES'],'MONITOR_STRIPES',temporal_frames=stripe_hist),
                }
                seg_ad={
                    'CLEAN':crop,
                    'LOW_RES':conditioned_parking_crop(base['LOW_RES'],'LOW_RES',temporal_frames=low_hist),
                    'SUN_GLARE':conditioned_parking_crop(base['SUN_GLARE'],'SUN_GLARE'),
                    # Temporal median helped YOLO but destroyed detailed masks; keep RAW stripe for SEG.
                    'MONITOR_STRIPES':base['MONITOR_STRIPES'],
                }
                det_methods={'CLEAN':'RAW','LOW_RES':'CAUSAL_TEMPORAL_MEDIAN+LOWRES_RESAMPLE','SUN_GLARE':'GLARE_TONE_COMPRESSION','MONITOR_STRIPES':'CAUSAL_TEMPORAL_MEDIAN_DESTRIPE'}
                seg_methods={'CLEAN':'RAW','LOW_RES':'CAUSAL_TEMPORAL_MEDIAN+LOWRES_RESAMPLE','SUN_GLARE':'GLARE_TONE_COMPRESSION','MONITOR_STRIPES':'RAW_STRIPE_SEG'}
                if len(panels)<panel_limit:
                    panels.append((f'{lid} {cctv} t={format_timestamp(t)}',{
                        'CLEAN':crop,'LOW_RAW':base['LOW_RES'],'LOW_MULTI':det_ad['LOW_RES'],
                        'SUN_RAW':base['SUN_GLARE'],'SUN_TONE':det_ad['SUN_GLARE'],
                        'STRIPE_RAW':base['MONITOR_STRIPES'],'STRIPE_TEMP_YOLO':det_ad['MONITOR_STRIPES'],
                    }))
                for cond in conditions:
                    for prep in ['RAW','ADAPTIVE']:
                        dimg=base[cond] if prep=='RAW' else det_ad[cond]
                        simg=base[cond] if prep=='RAW' else seg_ad[cond]
                        det_jobs.append(dimg);seg_jobs.append(simg)
                        meta.append((sample_id,float(t),cctv,lid,cond,prep,point,radius,
                                     det_methods[cond] if prep=='ADAPTIVE' else 'RAW',seg_methods[cond] if prep=='ADAPTIVE' else 'RAW',
                                     image_quality_metrics(dimg),image_quality_metrics(simg)))
                n+=1
                if progress and n%max(1,total//20)==0:
                    progress(min(0.25,n/total*0.25),f'Balanced robustness crops {n}/{total}')
    finally:
        cap.release()

    gt_template=pd.DataFrame(gt_rows).drop_duplicates('sample_id').sort_values(['time_sec','cctv','local_id'])
    gt_template.to_csv(gt_review_dir/'robustness_gt_template.csv',index=False,encoding='utf-8-sig')
    if not gt_template.empty:
        gt_template['sampling_stratum'].value_counts().rename_axis('sampling_stratum').reset_index(name='N').to_csv(gt_review_dir/'sampling_balance.csv',index=False,encoding='utf-8-sig')

    det_results=[];seg_results=[]
    for st in range(0,len(det_jobs),batch):
        det_results.extend(detector.detect_batch(det_jobs[st:st+batch]))
        pts=[meta[i][6] for i in range(st,min(st+batch,len(meta)))];rads=[meta[i][7] for i in range(st,min(st+batch,len(meta)))]
        seg_results.extend(segmenter.segment_batch(seg_jobs[st:st+batch],pts,rads))
        if progress:
            progress(0.25+0.70*min(1.0,(st+min(batch,len(det_jobs)-st))/max(1,len(det_jobs))),f'Balanced robustness inference {min(st+batch,len(det_jobs))}/{len(det_jobs)}')

    for m,dets,segs in zip(meta,det_results,seg_results):
        sample_id,t,cctv,lid,cond,prep,point,radius,det_method,seg_method,qd,qs=m
        dp,dc=_det_support(dets,point,radius);sp,sc,sm,sco,spoint=_seg_support(segs,min_core,pos_conf,pos_core)
        samples.append({'sample_id':sample_id,'time_sec':t,'timestamp':format_timestamp(t),'cctv':cctv,'local_id':lid,
                        'condition':cond,'preprocess':prep,'detector_adaptive_method':det_method,'seg_adaptive_method':seg_method,
                        'synthetic_stress':int(cond!='CLEAN'),'detector_positive':dp,'detector_conf':dc,'seg_positive':sp,'seg_conf':sc,
                        'seg_mask_ratio':sm,'seg_core_overlap_ratio':sco,'seg_point_covered':spoint,
                        **qd,'seg_input_highlight_ratio':float(qs.get('highlight_ratio',0.0)),
                        'seg_input_sharpness':float(qs.get('sharpness_laplacian_var',0.0)),
                        'seg_input_stripe_energy_ratio':float(qs.get('stripe_energy_ratio',0.0))})
    sdf=pd.DataFrame(samples);sdf.to_csv(out_dir/'degradation_robustness_samples.csv',index=False,encoding='utf-8-sig')

    clean_raw=sdf[(sdf['condition']=='CLEAN')&(sdf['preprocess']=='RAW')].copy();qrows=[]
    for r in clean_raw.itertuples():
        qrows.append({'cctv':str(r.cctv),'highlight_ratio':float(r.highlight_ratio),'stripe_energy_ratio':float(r.stripe_energy_ratio),'sharpness_laplacian_var':float(r.sharpness_laplacian_var)})
    rel_thr=build_relative_quality_thresholds(qrows,cfg);th_rows=[]
    for cam,th in rel_thr.items():
        if cam!='__GLOBAL__': th_rows.append({'cctv':cam,**th})
    pd.DataFrame(th_rows).to_csv(out_dir/'real_condition_thresholds.csv',index=False,encoding='utf-8-sig')
    if not clean_raw.empty:
        crows=[]
        for r in clean_raw.itertuples():
            cond,score=classify_quality_condition({'highlight_ratio':float(r.highlight_ratio),'stripe_energy_ratio':float(r.stripe_energy_ratio),'sharpness_laplacian_var':float(r.sharpness_laplacian_var)},rel_thr.get(str(r.cctv),rel_thr.get('__GLOBAL__',{})))
            crows.append({'sample_id':str(r.sample_id),'cctv':str(r.cctv),'local_id':str(r.local_id),'time_sec':float(r.time_sec),'relative_condition':cond,'relative_condition_score':score})
        pd.DataFrame(crows).to_csv(out_dir/'real_condition_classification.csv',index=False,encoding='utf-8-sig')

    base_clean=sdf[(sdf['condition']=='CLEAN')&(sdf['preprocess']=='RAW')][['sample_id','detector_positive','seg_positive']].rename(columns={'detector_positive':'clean_det','seg_positive':'clean_seg'})
    merged=sdf.merge(base_clean,on='sample_id',how='left');summary=[]
    for (cond,prep),part in merged.groupby(['condition','preprocess']):
        det_ref=part[part['clean_det']==1];seg_ref=part[part['clean_seg']==1]
        summary.append({'condition':cond,'preprocess':prep,'N':len(part),'detector_positive_rate':float(part['detector_positive'].mean()) if len(part) else 0.0,
                        'seg_positive_rate':float(part['seg_positive'].mean()) if len(part) else 0.0,
                        'detector_retention_vs_clean_positive':float(det_ref['detector_positive'].mean()) if len(det_ref) else np.nan,
                        'seg_retention_vs_clean_positive':float(seg_ref['seg_positive'].mean()) if len(seg_ref) else np.nan,
                        'mean_highlight_ratio':float(part['highlight_ratio'].mean()) if len(part) else 0.0,
                        'mean_sharpness':float(part['sharpness_laplacian_var'].mean()) if len(part) else 0.0,
                        'mean_stripe_energy_ratio':float(part['stripe_energy_ratio'].mean()) if len(part) else 0.0,
                        'metric_note':'retention_vs_clean is a stress metric; use degradation_ground_truth_metrics.csv after human labels'})
    pd.DataFrame(summary).to_csv(out_dir/'degradation_robustness_summary.csv',index=False,encoding='utf-8-sig')

    piv=merged.pivot_table(index=['sample_id','condition'],columns='preprocess',values=['detector_positive','seg_positive'],aggfunc='max').reset_index();rec=[]
    for cond in conditions:
        p=piv[piv['condition']==cond]
        if p.empty: continue
        try:
            dr=p[('detector_positive','RAW')];da=p[('detector_positive','ADAPTIVE')];sr=p[('seg_positive','RAW')];sa=p[('seg_positive','ADAPTIVE')]
            rec.append({'condition':cond,'N':len(p),'detector_adaptive_recovery_rate':float(((dr==0)&(da==1)).mean()),
                        'seg_adaptive_recovery_rate':float(((sr==0)&(sa==1)).mean()),'detector_adaptive_loss_rate':float(((dr==1)&(da==0)).mean()),
                        'seg_adaptive_loss_rate':float(((sr==1)&(sa==0)).mean())})
        except Exception: pass
    pd.DataFrame(rec).to_csv(out_dir/'degradation_adaptive_recovery.csv',index=False,encoding='utf-8-sig')

    gt_labels=_load_gt_labels(cfg.get('gt_file'));_write_ground_truth_metrics(sdf,gt_labels,out_dir)
    if not gt_labels.empty: gt_labels.to_csv(out_dir/'robustness_gt_labels_used.csv',index=False,encoding='utf-8-sig')
    panels_dir=ensure_dir(out_dir/'sample_panels')
    for i,(title,imgs) in enumerate(panels,1): _draw_sample_panel(imgs,title,panels_dir/f'robustness_sample_{i:02d}.jpg',quality)

    (gt_review_dir/'README.txt').write_text(
        'Balanced robustness GT labeling\n\n'
        'The review set intentionally mixes likely occupied and likely empty samples; the hint is sampling-only and is never used as GT.\n'
        'Label each CLEAN crop 1=OCCUPIED, 0=EMPTY, or UNKNOWN. The same label is inherited by synthetic LOW_RES / SUN_GLARE / MONITOR_STRIPES variants.\n'
        'If the final human labels are still strongly imbalanced, the next run/report will say so explicitly.\n',encoding='utf-8')
    (out_dir/'ROBUSTNESS_README.txt').write_text(
        'Parking Slot Engine v16 balanced degradation robustness experiment\n\n'
        'LOW_RES: causal multi-frame aggregation + conservative resample.\n'
        'SUN_GLARE: RAW and tone-compressed inputs run in parallel.\n'
        'MONITOR_STRIPES: temporal median is YOLO-only; SEG keeps RAW stripe input because temporal cleaning damaged masks in v15.4.\n'
        'Use degradation_ground_truth_metrics.csv and degradation_fusion_metrics.csv after balanced human labels.\n'
        'All temporal preprocessing is causal and never uses future frames.\n',encoding='utf-8')
    if progress: progress(1.0,'Balanced robustness experiment complete')
    return {'enabled':True,'rows':len(sdf),'slots':sorted(set(sample_df['local_id'].astype(str))),'times':sorted(set(sample_df['time_sec'].astype(float))),'gt_samples':len(gt_template)}

def label_robustness_ground_truth(review_dir: str | Path, output_csv: str | Path) -> Dict:
    """Keyboard UI: O/1=occupied, E/0=empty, U=unknown, B=back, S/Q=save+quit.

    v16.1 stores labels as strings (O/E/U) so UNKNOWN never collides with a float64
    column. Metrics convert O/E to 1/0 and automatically exclude U. Every label
    key is persisted immediately so an interrupted labeling session is recoverable.
    """
    review=Path(review_dir); template=review/'robustness_gt_template.csv'
    if not template.is_file():
        raise FileNotFoundError(f'robustness_gt_template.csv not found: {template}')
    df=pd.read_csv(template,encoding='utf-8-sig')
    out=Path(output_csv); out.parent.mkdir(parents=True,exist_ok=True)
    old=_load_gt_labels(out)
    old_map={str(r.sample_id):r for r in old.itertuples()}
    if 'occupied_gt' not in df.columns: df['occupied_gt']='U'
    if 'notes' not in df.columns: df['notes']=''
    # Force an explicit string dtype before any keyboard edit. This is the fix for
    # pandas TypeError: Invalid value '' for dtype 'float64' when U was pressed.
    df['occupied_gt']=df['occupied_gt'].map(_canonical_gt_label).astype('string')
    df['notes']=df['notes'].fillna('').astype('string')
    for i,r in df.iterrows():
        prev=old_map.get(str(r['sample_id']))
        if prev is not None:
            df.at[i,'occupied_gt']=_canonical_gt_label(getattr(prev,'occupied_gt','U'))
            df.at[i,'notes']=str(getattr(prev,'notes','') or '')

    def save_now():
        labeled=df[['sample_id','occupied_gt','notes']].copy()
        labeled['occupied_gt']=labeled['occupied_gt'].map(_canonical_gt_label).astype('string')
        labeled.to_csv(out,index=False,encoding='utf-8-sig')
        return labeled

    idxs=list(range(len(df))); start=len(idxs)
    for j,i in enumerate(idxs):
        if _canonical_gt_label(df.at[i,'occupied_gt'])=='U':
            start=j; break
    if start>=len(idxs):
        labeled=save_now()
        vals=_gt_label_numeric_series(labeled['occupied_gt'])
        nl=int(vals.isin([0,1]).sum()); occ=int((vals==1).sum()); emp=int((vals==0).sum())
        minority=min(occ,emp)/max(1,occ+emp)
        return {'path':str(out),'total':len(labeled),'labeled':nl,'occupied':occ,'empty':emp,
                'unknown':int((labeled['occupied_gt'].map(_canonical_gt_label)=='U').sum()),
                'balance_ok':bool(nl==0 or minority>=0.25),'minority_fraction':float(minority)}

    pos=start; window='Robustness GT | O/1 occupied | E/0 empty | U unknown | B back | S/Q save+quit'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    try:
        while 0 <= pos < len(idxs):
            i=idxs[pos]; r=df.loc[i]; ip=review/str(r['image_path'])
            img=cv2.imread(str(ip))
            if img is None:
                pos+=1; continue
            val=_canonical_gt_label(r.get('occupied_gt','U'))
            bar=np.full((86,img.shape[1],3),12,np.uint8)
            cv2.putText(bar,f'{pos+1}/{len(idxs)} {r.sample_id}  current={val}',(14,34),cv2.FONT_HERSHEY_SIMPLEX,0.70,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(bar,'O/1 occupied   E/0 empty   U unknown   B back   S/Q save+quit',(14,70),cv2.FONT_HERSHEY_SIMPLEX,0.56,(230,230,230),2,cv2.LINE_AA)
            cv2.imshow(window,np.vstack([bar,img])); k=cv2.waitKey(0)&0xFF
            changed=False
            if k in (ord('o'),ord('1')):
                df.at[i,'occupied_gt']='O'; pos+=1; changed=True
            elif k in (ord('e'),ord('0')):
                df.at[i,'occupied_gt']='E'; pos+=1; changed=True
            elif k==ord('u'):
                df.at[i,'occupied_gt']='U'; pos+=1; changed=True
            elif k in (ord('b'),81):
                pos=max(0,pos-1)
            elif k in (ord('s'),ord('q'),27):
                save_now(); break
            if changed:
                save_now()
    finally:
        cv2.destroyWindow(window)
    labeled=save_now()
    vals=_gt_label_numeric_series(labeled['occupied_gt'])
    nl=int(vals.isin([0,1]).sum()); occ=int((vals==1).sum()); emp=int((vals==0).sum())
    minority=min(occ,emp)/max(1,occ+emp)
    return {'path':str(out),'total':len(labeled),'labeled':nl,'occupied':occ,'empty':emp,
            'unknown':int((labeled['occupied_gt'].map(_canonical_gt_label)=='U').sum()),
            'balance_ok':bool(nl==0 or minority>=0.25),'minority_fraction':float(minority)}
