# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import threading
import traceback
import hashlib
import shutil
import time
import webbrowser
from datetime import datetime
from pathlib import Path
import tkinter as tk
from localization import tr, set_language, get_language, refresh_tree
from tkinter import filedialog, messagebox, ttk

import cv2
import pandas as pd
from PIL import Image, ImageTk

from detector import VehicleDetector, draw_detections
from detector_tuner import run_warped_detector_sweep
from research_capture import generate_research_screenshots, generate_ground_truth_review
from degradation_robustness import run_degradation_robustness_experiment, label_robustness_ground_truth
from research_history import open_history_window, append_run
from update_manager import current_version, check_latest, prepare_update, launch_apply, load_update_config
from transition_refiner_v164 import run as run_v164_candidate, RefinerConfig
from validation_center import (
    open_validation_center as launch_validation_center,
    configured_camera_names,
    write_episode_metrics,
    write_repeat_stability,
    write_window_evaluation,
)

from slot_engine import (
    extract_evidence,
    extract_segmentation_assist,
    learn_slots_and_candidates,
    load_slots,
    next_global_id,
    next_local_id,
    save_slots,
    search_state_parameters,
    make_slot_gt_template,
    write_camera_detector_metrics,
)
from utils import (
    crop_roi,
    make_perspective_roi,
    ensure_dir,
    load_ground_truth,
    load_json,
    open_video,
    read_frame_at,
    safe_copy,
    save_json,
    zip_paths,
)

APP_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = APP_DIR / 'settings.json'
ROIS_PATH = APP_DIR / 'rois.json'
SLOTS_PATH = APP_DIR / 'slots.json'
WORK_DIR = ensure_dir(APP_DIR / 'work')
LEARN_DIR = ensure_dir(WORK_DIR / 'learned')
OUTPUT_ROOT = ensure_dir(APP_DIR / 'output')
DETECTOR_TUNE_DIR = ensure_dir(WORK_DIR / 'detector_tuning')
UI_STATE_PATH = APP_DIR / 'ui_state.json'
LAST_ERROR_PATH = APP_DIR / 'LAST_ERROR.txt'
# Cross-version cache: keep expensive detector/evidence results outside the version folder.
# On Windows this lives under %LOCALAPPDATA% so Desktop/OneDrive copies do not duplicate it.
_cache_base = Path(os.environ.get('LOCALAPPDATA', str(WORK_DIR))) / 'ParkingSlotEngine'
CACHE_ROOT = ensure_dir(_cache_base / 'cache')
LABEL_ROOT = ensure_dir(_cache_base / 'labels')
ROBUSTNESS_GT_PATH = LABEL_ROOT / 'robustness_gt.csv'
APP_VERSION = current_version()
GITHUB_RELEASES_URL = 'https://github.com/sopo9880/Parking_/releases'


def _file_token(path_text):
    p = Path(str(path_text or '')).expanduser()
    if not p.is_file():
        return {'path': str(p), 'exists': False}
    st = p.stat()
    return {'path': str(p.resolve()), 'exists': True, 'size': int(st.st_size), 'mtime_ns': int(st.st_mtime_ns)}


def _manual_slot_payload(slots):
    out=[]
    for s in slots:
        out.append({
            'local_id':str(s.get('local_id','')),'global_id':str(s.get('global_id','')),'cctv':str(s.get('cctv','')),
            'point':[round(float(x),4) for x in s.get('point',[0,0])],
            'initial_state':str(s.get('initial_state','UNKNOWN')),
        })
    return sorted(out,key=lambda x:(x['cctv'],x['local_id']))


def _stable_signature(kind, video, gt_path, rois, slots, settings):
    common={
        'kind':kind,'cache_version':str(settings.get('pipeline_cache',{}).get('cache_version','v13')),
        'video':_file_token(video),'rois':rois,'slots':_manual_slot_payload(slots),
        'dev_end_sec':settings.get('dev_end_sec'),'eval_end_sec':settings.get('eval_end_sec'),
    }
    if kind=='detector':
        common.update({'gt':_file_token(gt_path),'detector':settings.get('detector',{}),'sweep':settings.get('detector_sweep',{})})
    elif kind=='learning':
        common.update({'detector_by_cctv':settings.get('detector_by_cctv',{}),'matching':settings.get('matching',{}),'learn_sample_sec':settings.get('learn_sample_sec')})
    elif kind=='evidence':
        common.update({
            'detector_by_cctv':settings.get('detector_by_cctv',{}),'matching':settings.get('matching',{}),
            'temporal_tracker':{k:v for k,v in settings.get('temporal',{}).items() if k.startswith('tracker_')},
            'selective_aux':settings.get('selective_aux',{}),'slot_crop_detector':settings.get('slot_crop_detector',{}),
            'evidence_sample_sec':settings.get('evidence_sample_sec'),
        })
    elif kind=='segmentation':
        common['cache_version']=str(settings.get('pipeline_cache',{}).get('segmentation_cache_version','v15.4-adaptive-seg-1'))
        common.update({
            'segmentation_assist':settings.get('segmentation_assist',{}),
            'evidence_signature':_stable_signature('evidence',video,gt_path,rois,slots,settings),
        })
    elif kind=='robustness':
        common['cache_version']=str(settings.get('pipeline_cache',{}).get('robustness_cache_version','v15.4-temporal-robustness-gt-1'))
        rob_cfg=settings.get('robustness_experiment',{}) or {}
        common.update({
            'segmentation_assist':settings.get('segmentation_assist',{}),
            'robustness_experiment':rob_cfg,
            'robustness_gt':_file_token(rob_cfg.get('gt_file','')) if rob_cfg.get('gt_file') else {'exists':False},
            'evidence_signature':_stable_signature('evidence',video,gt_path,rois,slots,settings),
        })
    raw=json.dumps(common,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _cache_sig_matches(path, signature):
    d=load_json(path,{}) or {}
    return str(d.get('signature',''))==str(signature)


def _write_cache_sig(path, signature, kind):
    save_json(path,{'signature':signature,'kind':kind,'created_at':datetime.now().isoformat(timespec='seconds')})


def _copy_cache_files(src_dir, dst_dir, names):
    src=Path(src_dir); dst=ensure_dir(dst_dir)
    copied=[]
    for name in names:
        sp=src/name
        if sp.is_file():
            safe_copy(sp,dst/name); copied.append(name)
    return copied


# v16.2: slot-learning cache must restore not only template image files but also the
# learned per-slot geometry/template metadata that points at those files. Earlier
# v16 builds copied work/learned from shared cache while leaving a fresh package
# slots.json untouched. That made template_initial=None for every slot and
# normalized_visual_diff() returned its missing-template sentinel (1.0) for every
# slot-frame. The result looked like a real appearance change and could pin a slot
# OCCUPIED forever. Keep the administrator's manual identity fields immutable and
# restore only learned geometry metadata.
_LEARNED_SLOT_FIELDS = {
    'anchor_center','anchor_center_raw','anchor_trusted','anchor_reason','anchor_clamped',
    'anchor_std_xy_px','anchor_std_px','anchor_shift_px','anchor_max_shift_px',
    'anchor_sample_gate_px','nearest_manual_slot_distance_px','nearest_slot_distance_px',
    'expected_box_wh','assignment_radius_px','region','learned_box_samples',
    'learned_center_samples_raw','learned_center_samples','manual_point','template_initial','geometry_version',
}
_LEARNED_SNAPSHOT_NAME = 'learned_slots_snapshot.json'


def _write_learned_slot_snapshot(slots_path, learned_dir):
    slots=load_slots(slots_path)
    payload=[]
    for s in slots:
        row={'local_id':str(s.get('local_id','')),'cctv':str(s.get('cctv',''))}
        for k in _LEARNED_SLOT_FIELDS:
            if k in s:
                row[k]=s[k]
        payload.append(row)
    save_json(Path(learned_dir)/_LEARNED_SNAPSHOT_NAME,{'slots':payload})
    return payload


def _merge_learned_slot_metadata(current_slots, learned_slots):
    learned_by={(str(s.get('cctv','')),str(s.get('local_id',''))):s for s in learned_slots}
    merged=[]
    for cur in current_slots:
        d=dict(cur); src=learned_by.get((str(cur.get('cctv','')),str(cur.get('local_id',''))),{})
        for k in _LEARNED_SLOT_FIELDS:
            if k in src:
                d[k]=src[k]
        # The manual point is the immutable slot identity even if an old cache contains it.
        if 'point' in cur: d['point']=cur['point']
        if 'global_id' in cur: d['global_id']=cur['global_id']
        if 'initial_state' in cur: d['initial_state']=cur['initial_state']
        if 'source' in cur: d['source']=cur['source']
        merged.append(d)
    return merged


def _learned_metadata_health(slots, learned_dir):
    learned_dir=Path(learned_dir); total=max(1,len(slots)); template_ok=0; region_ok=0; geom_ok=0
    missing_templates=[]; missing_regions=[]
    for s in slots:
        rel=s.get('template_initial')
        if rel and (learned_dir/str(rel)).is_file(): template_ok+=1
        else: missing_templates.append(str(s.get('local_id','?')))
        reg=s.get('region')
        if isinstance(reg,(list,tuple)) and len(reg)==4:
            try:
                vals=[float(x) for x in reg]
                if vals[2]>vals[0] and vals[3]>vals[1]: region_ok+=1
                else: missing_regions.append(str(s.get('local_id','?')))
            except Exception: missing_regions.append(str(s.get('local_id','?')))
        else: missing_regions.append(str(s.get('local_id','?')))
        if str(s.get('geometry_version',''))=='v12_manual_voronoi': geom_ok+=1
    stats={'slot_count':len(slots),'template_ok':template_ok,'region_ok':region_ok,'geometry_ok':geom_ok,
           'template_fraction':template_ok/total,'region_fraction':region_ok/total,'geometry_fraction':geom_ok/total,
           'missing_templates':missing_templates[:12],'missing_regions':missing_regions[:12]}
    return bool(template_ok==len(slots) and region_ok==len(slots) and geom_ok==len(slots)),stats


def _restore_learning_metadata_from_cache(slots_path, learned_dir, cache_dir=None):
    """Restore learned slot metadata while preserving manual point/link/state fields.

    Returns (ok, source, stats). Old v15/v16 caches may have templates but no snapshot;
    in that case try a compatible sibling runtime slots.json once, otherwise force one
    fresh learning pass rather than silently creating invalid appearance evidence.
    """
    current=load_slots(slots_path); learned_dir=Path(learned_dir); candidates=[]
    for base in [learned_dir, Path(cache_dir) if cache_dir else None]:
        if not base: continue
        sp=base/_LEARNED_SNAPSHOT_NAME
        if sp.is_file():
            data=load_json(sp,{}) or {}; candidates.append(('snapshot',list(data.get('slots',[]))))
    # Migration path for older runtime folders: their slots.json was modified in-place
    # by successful slot learning even though the shared cache had no metadata snapshot.
    try:
        for sibling_slots in APP_DIR.parent.glob('parking_slot_engine_v*/slots.json'):
            if sibling_slots.resolve()==Path(slots_path).resolve(): continue
            try: ss=load_slots(sibling_slots)
            except Exception: continue
            if _manual_slot_payload(ss)!=_manual_slot_payload(current): continue
            learned_count=sum(1 for x in ss if x.get('template_initial') and x.get('region') and x.get('geometry_version')=='v12_manual_voronoi')
            if learned_count:
                candidates.append((f'sibling:{sibling_slots.parent.name}',ss))
    except Exception:
        pass
    # Prefer the candidate with the most complete learned metadata.
    best=None
    for source,ls in candidates:
        score=sum(1 for x in ls if x.get('template_initial') and x.get('region') and x.get('geometry_version')=='v12_manual_voronoi')
        if best is None or score>best[0]: best=(score,source,ls)
    if best is not None:
        merged=_merge_learned_slot_metadata(current,best[2]); save_slots(slots_path,merged)
        ok,stats=_learned_metadata_health(merged,learned_dir)
        if ok:
            _write_learned_slot_snapshot(slots_path,learned_dir)
            if cache_dir:
                safe_copy(learned_dir/_LEARNED_SNAPSHOT_NAME,Path(cache_dir)/_LEARNED_SNAPSHOT_NAME)
            return True,best[1],stats
    ok,stats=_learned_metadata_health(current,learned_dir)
    return ok,'current_slots',stats


def _evidence_sanity(evidence, slots, learned_dir):
    """Reject the specific v16 corrupted appearance cache and other impossible evidence."""
    stats={'rows':int(len(evidence)) if evidence is not None else 0}
    meta_ok,meta_stats=_learned_metadata_health(slots,learned_dir); stats.update({f'learning_{k}':v for k,v in meta_stats.items() if not isinstance(v,list)})
    if evidence is None or evidence.empty:
        return False,'evidence is empty',stats
    if 'visual_diff_initial' not in evidence.columns:
        return False,'visual_diff_initial column missing',stats
    vd=pd.to_numeric(evidence['visual_diff_initial'],errors='coerce')
    valid=vd.dropna(); stats['visual_diff_valid']=int(len(valid))
    if valid.empty:
        return False,'visual_diff_initial has no numeric values',stats
    stats['visual_diff_mean']=float(valid.mean()); stats['visual_diff_std']=float(valid.std(ddof=0)); stats['visual_diff_min']=float(valid.min()); stats['visual_diff_max']=float(valid.max())
    stats['visual_diff_exact_1_fraction']=float((valid.sub(1.0).abs()<=1e-12).mean())
    stats['visual_diff_unique_rounded_6']=int(valid.round(6).nunique())
    if not meta_ok:
        return False,'learned slot template/region metadata incomplete',stats
    if len(valid)>=100 and stats['visual_diff_exact_1_fraction']>=0.95:
        return False,'visual_diff_initial is almost entirely the missing-template sentinel 1.0',stats
    if len(valid)>=1000 and stats['visual_diff_unique_rounded_6']<=2 and stats['visual_diff_mean']>=0.95:
        return False,'visual_diff_initial distribution is implausibly constant near 1.0',stats
    return True,'OK',stats


def _write_evidence_sanity_report(path, ok, reason, stats, action):
    lines=['Parking Slot Engine v16.2 evidence sanity check','',f'ok={bool(ok)}',f'reason={reason}',f'action={action}']
    for k in sorted(stats): lines.append(f'{k}={stats[k]}')
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def _gt_occupancy_fingerprint(gt):
    if gt is None or gt.empty or 'ground_truth_occupied_space_count' not in gt.columns:
        return ''
    parts=[]
    for r in gt.itertuples():
        ts=str(getattr(r,'timestamp','')).strip() if hasattr(r,'timestamp') else f"{float(getattr(r,'time_sec',0.0)):.3f}"
        try: occ=int(round(float(getattr(r,'ground_truth_occupied_space_count'))))
        except Exception: return ''
        parts.append(f'{ts}|{occ}')
    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()


def _write_non_regression_check(run_dir, gt, result, settings):
    cfg=dict(settings.get('non_regression_reference',{}) or {}); path=Path(run_dir)/'NON_REGRESSION_CHECK.txt'
    if not bool(cfg.get('enabled',True)):
        path.write_text('disabled\n',encoding='utf-8'); return {'applicable':False,'passed':True}
    actual_fp=_gt_occupancy_fingerprint(gt); expected_fp=str(cfg.get('gt_occupancy_fingerprint',''))
    applicable=bool(expected_fp and actual_fp==expected_fp)
    lines=['Parking Slot Engine v16.2 non-regression audit','',f'applicable={applicable}',f'actual_gt_fingerprint={actual_fp}',f'expected_gt_fingerprint={expected_fp}']
    if not applicable:
        lines.append('status=SKIPPED_DIFFERENT_GT'); path.write_text('\n'.join(lines)+'\n',encoding='utf-8'); return {'applicable':False,'passed':True}
    board=result.get('leaderboard') if isinstance(result,dict) else None
    if board is None or getattr(board,'empty',True) or 'temporal_variant' not in board.columns:
        lines.append('status=FAILED_NO_LEADERBOARD'); path.write_text('\n'.join(lines)+'\n',encoding='utf-8'); return {'applicable':True,'passed':False}
    safe=board[board['temporal_variant'].astype(str)=='SAFE_BASELINE']
    if safe.empty:
        lines.append('status=FAILED_NO_SAFE_BASELINE'); path.write_text('\n'.join(lines)+'\n',encoding='utf-8'); return {'applicable':True,'passed':False}
    r=safe.iloc[0]; exact=float(r.get('test_exact_rate',float('nan'))); mae=float(r.get('test_MAE',float('nan')))
    ex_ref=float(cfg.get('expected_test_exact_rate',0.8529411764705882)); mae_ref=float(cfg.get('expected_test_mae',0.14705882352941177))
    ex_tol=float(cfg.get('exact_rate_tolerance',0.005)); mae_tol=float(cfg.get('mae_tolerance',0.01))
    passed=bool(abs(exact-ex_ref)<=ex_tol and abs(mae-mae_ref)<=mae_tol)
    lines += [f'safe_baseline_test_exact_rate={exact}',f'reference_test_exact_rate={ex_ref}',f'safe_baseline_test_mae={mae}',f'reference_test_mae={mae_ref}',f'status={"PASS" if passed else "WARNING_NON_REGRESSION_MISMATCH"}',
              'note=TEST remains evaluation-only; this audit never selects or tunes a model.']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'applicable':True,'passed':passed,'test_exact_rate':exact,'test_mae':mae}


def _nearest_slot(slots, cctv, x, y, max_dist=32):
    best = None
    best_d = 1e9
    for s in slots:
        if s['cctv'] != cctv:
            continue
        px, py = s['point']
        d = ((px-x)**2 + (py-y)**2) ** 0.5
        if d < best_d:
            best_d = d
            best = s
    if best_d <= max_dist:
        return best
    return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(tr(f'Parking Research Agent {APP_VERSION} | v16.2 SAFE + v16.4 CANDIDATE'))
        sw=max(1024,int(self.winfo_screenwidth())); sh=max(720,int(self.winfo_screenheight()))
        init_w=max(980,min(1180,sw-80)); init_h=max(700,min(900,sh-100))
        self.geometry(f'{init_w}x{init_h}')
        self.minsize(900, 650)

        self.settings = load_json(SETTINGS_PATH, {})
        set_language((self.settings.get('ui', {}) or {}).get('language', 'ko'))
        self.language_var = tk.StringVar(value=get_language())
        ui_state = load_json(UI_STATE_PATH, {}) or {}
        last_video = str(ui_state.get('video_path', '') or '').strip()
        last_gt = str(ui_state.get('gt_path', '') or '').strip()
        last_slot_gt = str(ui_state.get('slot_gt_path', '') or '').strip()
        if not last_gt or not Path(last_gt).is_file():
            sample_gt = APP_DIR / 'sample_ground_truth.csv'
            last_gt = str(sample_gt) if sample_gt.is_file() else ''
        if last_video and not Path(last_video).is_file():
            last_video = ''
        if last_slot_gt and not Path(last_slot_gt).is_file():
            last_slot_gt = ''
        self.video_var = tk.StringVar(value=last_video)
        self.gt_var = tk.StringVar(value=last_gt)
        self.slot_gt_var = tk.StringVar(value=last_slot_gt)
        self.status_var = tk.StringVar(value=tr('Ready'))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.timing_var = tk.StringVar(value='Total Elapsed -- | Step Elapsed -- | Step ETA -- | Total Remaining --')
        self.cache_status_var = tk.StringVar(value='CACHE | not audited yet')
        self._pipeline_started_at = None
        self._step_started_at = None
        self._step_key = None

        self._build_ui()
        refresh_tree(self)
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        cfg = load_update_config(self.settings)
        if cfg.get('enabled', True) and cfg.get('check_on_startup', True):
            self.after(1200, lambda: self.check_for_updates(manual=False))

    def _build_ui(self):
        pad = {'padx': 10, 'pady': 7}

        agent = ttk.Frame(self)
        agent.pack(fill='x', padx=10, pady=(10, 2))
        left = ttk.Frame(agent)
        left.pack(side='top', fill='x', expand=True)
        ttk.Label(left, text=tr('Parking Research Agent'), font=('Segoe UI', 15, 'bold')).pack(side='left')
        ttk.Label(left, text=tr(f'{APP_VERSION}  |  v16.2 SAFE + v16.4 Candidate + External Validation')).pack(side='left', padx=(12, 0))
        right = ttk.Frame(agent)
        right.pack(side='top', anchor='e', pady=(5, 0))
        ttk.Button(right, text=tr('Settings'), command=self.open_settings).pack(side='left', padx=3)
        ttk.Button(right, text=tr('Validation Center'), command=self.open_validation_center).pack(side='left', padx=3)
        ttk.Button(right, text=tr('Research History'), command=self.open_research_history).pack(side='left', padx=3)
        ttk.Button(right, text=tr('Check Update'), command=lambda: self.check_for_updates(manual=True)).pack(side='left', padx=3)
        ttk.Button(right, text=tr('Releases'), command=lambda: webbrowser.open(GITHUB_RELEASES_URL)).pack(side='left', padx=3)
        top = ttk.LabelFrame(self, text=tr('1. Input'))
        top.pack(fill='x', **pad)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text=tr('Video')).grid(row=0, column=0, sticky='w', padx=8, pady=6)
        ttk.Entry(top, textvariable=self.video_var).grid(row=0, column=1, sticky='ew', padx=8, pady=6)
        ttk.Button(top, text=tr('Browse'), command=self.pick_video).grid(row=0, column=2, padx=8, pady=6)

        ttk.Label(top, text=tr('Ground Truth CSV')).grid(row=1, column=0, sticky='w', padx=8, pady=6)
        ttk.Entry(top, textvariable=self.gt_var).grid(row=1, column=1, sticky='ew', padx=8, pady=6)
        ttk.Button(top, text=tr('Browse'), command=self.pick_gt).grid(row=1, column=2, padx=8, pady=6)

        ttk.Label(top, text=tr('Slot GT events (optional)')).grid(row=2, column=0, sticky='w', padx=8, pady=6)
        ttk.Entry(top, textvariable=self.slot_gt_var).grid(row=2, column=1, sticky='ew', padx=8, pady=6)
        ttk.Button(top, text=tr('Browse'), command=self.pick_slot_gt).grid(row=2, column=2, padx=8, pady=6)

        setup = ttk.LabelFrame(self, text=tr('2. Initial setup'))
        setup.pack(fill='x', **pad)
        buttons = [
            ('Set 4-point CCTV ROI', self.setup_rois),
            ('Preview warped CCTV', self.preview_rois),
            ('Set slot points', self.setup_slots),
            ('Visual duplicate-slot linker', self.open_linker),
            ('Export slot-GT event template', self.export_slot_gt_template),
            ('Open settings.json', self.open_settings),
        ]
        for i, (txt, cmd) in enumerate(buttons):
            row, col = divmod(i, 3)
            ttk.Button(setup, text=tr(txt), command=cmd).grid(row=row, column=col, padx=8, pady=6, sticky='ew')
        for col in range(3):
            setup.columnconfigure(col, weight=1)

        learn = ttk.LabelFrame(self, text=tr('3. DEV tuning + slot learning (TEST is untouched)'))
        learn.pack(fill='x', **pad)
        ttk.Button(learn, text=tr('Tune YOLO on warped CCTV (DEV)'), command=self.start_detector_tuning).grid(row=0, column=0, padx=8, pady=8, sticky='ew')
        ttk.Button(learn, text=tr('Learn manual-point slots'), command=self.start_learning).grid(row=0, column=1, padx=8, pady=8, sticky='ew')
        ttk.Button(learn, text=tr('Review learning diagnostics'), command=self.review_learning_diagnostics).grid(row=0, column=2, padx=8, pady=8, sticky='ew')
        ttk.Button(learn, text=tr('Review candidate slots'), command=self.review_candidates).grid(row=1, column=0, padx=8, pady=8, sticky='ew')
        ttk.Button(learn, text=tr('Review slot overlap warnings'), command=self.review_overlap_warnings).grid(row=1, column=1, padx=8, pady=8, sticky='ew')
        ttk.Button(learn, text=tr('Open detector tuning folder'), command=self.open_detector_tuning_folder).grid(row=1, column=2, padx=8, pady=8, sticky='ew')
        learn.columnconfigure(0, weight=1)
        learn.columnconfigure(1, weight=1)
        learn.columnconfigure(2, weight=1)

        run = ttk.LabelFrame(self, text=tr('4. After setup: unattended pipeline'))
        run.pack(fill='x', **pad)
        self.all_in_one_btn = tk.Button(
            run, text=tr('ALL-IN-ONE  |  Cache -> SAFE -> v16.4 Candidate -> episode/repeat validation -> ZIP'),
            command=self.start_all_in_one, font=('Segoe UI', 11, 'bold'),
            bg='#16784a', fg='white', activebackground='#12653e', activeforeground='white',
            relief='raised', padx=10, pady=10
        )
        self.all_in_one_btn.pack(fill='x', padx=8, pady=(10,6))
        ttk.Label(
            run,
            text=tr('Once started, no more clicks are required. You can minimize the window; the final UPLOAD_TO_CHATGPT.zip is created automatically.')
        ).pack(anchor='w', padx=8, pady=(0,6))
        ttk.Button(run, text=tr('Manual: RUN FINAL EXPERIMENT ONLY'), command=self.start_experiment).pack(fill='x', padx=8, pady=(0,5))
        ttk.Button(run, text=tr('Label / repair robustness GT balance (existing labels are kept)'), command=self.label_robustness_gt).pack(fill='x', padx=8, pady=(0,8))
        ttk.Label(run, text=tr('DEV selects detector/state parameters. TEST 15:00-20:30 is evaluated only after selection. First 10s are warm-up.')).pack(anchor='w', padx=8, pady=(0,8))

        # Keep progress/timing visible from the first launch even as help text grows.
        # The bottom status area is packed before the expandable Controls panel so it always reserves screen space.
        bottom = ttk.Frame(self)
        bottom.pack(side='bottom', fill='x', padx=10, pady=8)
        ttk.Progressbar(bottom, variable=self.progress_var, maximum=100).pack(fill='x', side='top')
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor='w', pady=(4,0))
        ttk.Label(bottom, textvariable=self.timing_var).pack(anchor='w', pady=(2,0))
        ttk.Label(bottom, textvariable=self.cache_status_var, font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(2,0))

        info = ttk.LabelFrame(self, text=tr('Controls'))
        info.pack(fill='both', expand=True, **pad)
        text = (
            'ROI editor: click 4 corners in any order, drag to adjust, right click to remove, ENTER/S to confirm.\n'
            'Slot point editor: Left empty = add, Left existing + drag = move, Right = delete, T = toggle, U = undo current CCTV, S = save/next, Q = cancel\n'
            'Candidate review: Left click = accept candidate, Right click = reject candidate, S = save/next\n'
            'Visual duplicate-slot linker: click matching slots directly on all configured CCTV views, then Link selected. Camera count is set in Validation Center.\n\n'
            'Each tilted CCTV is first perspective-corrected from a 4-point quadrilateral into a rectangular image. '
            'The manual slot point stays fixed as the slot identity. Optional learned anchors are constrained inside that manual geometry, and Voronoi gating prevents a detection from hopping to a neighboring slot. The program keeps full-CCTV YOLO as the primary detector. Expensive per-slot crop YOLO is called only when FULL is weak, suddenly missing, appearance is suspicious, or a sparse audit is due. AUX can recover a missed FULL detection but can never delete a FULL detection. Selective instance segmentation now compares RAW with a condition-specific ADAPTIVE input (low-res recovery / glare tone compression / stripe-safe cleanup); it never creates occupancy by itself. Detector/evidence and segmentation caches are separated. Long-stationary candidate slots remain advisory only, '
            'merges user-linked duplicate slots, tracks vehicles online, and uses a rolling 10-second history to distinguish parking maneuvers from stable occupancy. '
            'The first 10 seconds are warm-up; every later decision uses only current/past frames, never future frames.\n'
            'After ROI/slot/link setup, the green ALL-IN-ONE button automatically reuses valid cache, runs the protected v16.2 SAFE baseline, automatically evaluates the v16.4 transition candidate, adds episode/cut and repeated-source stability metrics when configured, captures GT review figures, and builds the ZIP without further input.\n'
            'At the end, upload output/.../UPLOAD_TO_CHATGPT.zip to ChatGPT for analysis. The ZIP also contains automatically selected paper-ready screenshots.'
        )
        help_wrap = ttk.Frame(info)
        help_wrap.pack(fill='both', expand=True, padx=8, pady=8)
        help_text = tk.Text(help_wrap, height=9, wrap='word', relief='flat', borderwidth=0)
        help_scroll = ttk.Scrollbar(help_wrap, orient='vertical', command=help_text.yview)
        help_text.configure(yscrollcommand=help_scroll.set)
        help_text.insert('1.0', tr(text))
        help_text.configure(state='disabled')
        help_scroll.pack(side='right', fill='y')
        help_text.pack(side='left', fill='both', expand=True)

    def open_validation_center(self):
        launch_validation_center(self)

    def open_research_history(self):
        open_history_window(self)

    def check_for_updates(self, manual=False):
        cfg = load_update_config(self.settings)
        if not cfg.get('enabled', True):
            if manual:
                messagebox.showinfo(tr('Updates disabled'), tr('Auto Update is disabled in settings.json.'))
            return

        def job():
            try:
                info = check_latest(self.settings)
            except Exception as exc:
                if manual:
                    self.after(0, lambda: messagebox.showerror(tr('Update check failed'), tr(f'{type(exc).__name__}: {exc}')))
                return

            def finish():
                if not info.get('available'):
                    if manual:
                        messagebox.showinfo(tr('Up to date'), tr(f"Current version: {info.get('current')}\nLatest release: {info.get('latest') or 'none'}"))
                    return
                notes = str(info.get('body') or '').strip()
                preview = notes[:1200] + ('...' if len(notes) > 1200 else '')
                msg = (f"New version available: {info.get('latest')}\n"
                       f"Current: {info.get('current')}\n\n{preview}\n\n"
                       'Download, verify and install this update? The Agent will close and restart.')
                ask = bool(cfg.get('ask_before_install', True))
                if ask and not messagebox.askyesno(tr('Parking Research Agent Update'), tr(msg)):
                    return
                self.status_var.set(tr(f"Downloading update {info.get('latest')}..."))
                self._download_and_apply_update(info)
            self.after(0, finish)

        threading.Thread(target=job, daemon=True).start()

    def _download_and_apply_update(self, info):
        def job():
            try:
                zpath = prepare_update(info)
                self.after(0, lambda: self.status_var.set(tr('Update verified. Restarting to install...')))
                launch_apply(zpath, restart=True)
                self.after(250, self.destroy)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror(tr('Update failed'), tr(f'{type(exc).__name__}: {exc}')))
                self.after(0, lambda: self.status_var.set(tr('Update failed; current version was not changed.')))
        threading.Thread(target=job, daemon=True).start()

    def _save_ui_state(self):
        save_json(UI_STATE_PATH, {
            'video_path': self.video_var.get().strip(),
            'gt_path': self.gt_var.get().strip(),
            'slot_gt_path': self.slot_gt_var.get().strip(),
        })

    def pick_video(self):
        p = filedialog.askopenfilename(filetypes=[('Video files','*.mp4 *.mov *.avi *.mkv'), ('All files','*.*')])
        if p:
            self.video_var.set(p)
            self._save_ui_state()

    def pick_gt(self):
        p = filedialog.askopenfilename(filetypes=[('CSV','*.csv'), ('All files','*.*')])
        if p:
            self.gt_var.set(p)
            self._save_ui_state()

    def pick_slot_gt(self):
        p = filedialog.askopenfilename(filetypes=[('CSV','*.csv'), ('All files','*.*')])
        if p:
            self.slot_gt_var.set(p)
            self._save_ui_state()

    def export_slot_gt_template(self):
        slots = load_slots(SLOTS_PATH)
        if not slots:
            messagebox.showwarning(tr('No slots'), tr('Set slot points and duplicate links first.'))
            return
        p = filedialog.asksaveasfilename(defaultextension='.csv', initialfile='slot_gt_events.csv', filetypes=[('CSV','*.csv')])
        if not p:
            return
        make_slot_gt_template(slots, p)
        self.slot_gt_var.set(p)
        messagebox.showinfo(tr('Saved'), tr('Slot GT event template saved. Keep the 0:00 rows and add a new row only when a GLOBAL SLOT changes state.'))

    def _validate_inputs(self, need_gt=True):
        # Path('') resolves to the current directory and .exists() returns True.
        # v10 therefore allowed a blank Video field to reach cv2.VideoCapture('').
        # Require an explicit non-empty regular file before any background job starts.
        video_text = self.video_var.get().strip()
        if not video_text:
            messagebox.showerror(tr('Video required'), tr('Video is empty. Click Browse and select the test video first.'))
            return False
        video = Path(video_text)
        if not video.is_file():
            messagebox.showerror(tr('Invalid video'), tr(f'Video file was not found:\n{video_text}'))
            return False
        if need_gt:
            gt_text = self.gt_var.get().strip()
            if not gt_text:
                messagebox.showerror(tr('Ground Truth required'), tr('Ground Truth CSV is empty. Select the GT CSV first.'))
                return False
            gt = Path(gt_text)
            if not gt.is_file():
                messagebox.showerror(tr('Invalid Ground Truth'), tr(f'Ground Truth CSV was not found:\n{gt_text}'))
                return False
        self._save_ui_state()
        return True

    def _load_first_frame(self):
        cap = open_video(self.video_var.get().strip())
        try:
            return read_frame_at(cap, 0.0)
        finally:
            cap.release()

    def _select_quad(self, frame, name):
        """Interactive 4-point selector. Points can be clicked in any order."""
        h, w = frame.shape[:2]
        try:
            sw = max(900, int(self.winfo_screenwidth()) - 120)
            sh = max(600, int(self.winfo_screenheight()) - 180)
        except Exception:
            sw, sh = 1500, 850
        scale = min(1.0, sw / max(1, w), sh / max(1, h))
        if scale < 1.0:
            display_base = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            display_base = frame.copy()

        points = []
        drag_index = [-1]
        preview_window = f'Perspective preview - {name}'
        window = f'4-point ROI - {name}'

        def disp_point_to_original(x, y):
            return [float(x) / scale, float(y) / scale]

        def original_to_disp(pt):
            return int(round(float(pt[0]) * scale)), int(round(float(pt[1]) * scale))

        def nearest_idx(x, y, max_dist=22):
            best, best_d = -1, 1e9
            for i, p in enumerate(points):
                dx, dy = original_to_disp(p)
                d = ((dx - x) ** 2 + (dy - y) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = i, d
            return best if best_d <= max_dist else -1

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = nearest_idx(x, y)
                if idx >= 0:
                    drag_index[0] = idx
                elif len(points) < 4:
                    points.append(disp_point_to_original(x, y))
                    drag_index[0] = len(points) - 1
            elif event == cv2.EVENT_MOUSEMOVE and drag_index[0] >= 0 and (flags & cv2.EVENT_FLAG_LBUTTON):
                points[drag_index[0]] = disp_point_to_original(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drag_index[0] = -1
            elif event == cv2.EVENT_RBUTTONDOWN:
                idx = nearest_idx(x, y, max_dist=35)
                if idx >= 0:
                    points.pop(idx)
                elif points:
                    points.pop()

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse)
        confirmed = False
        while True:
            canvas = display_base.copy()
            dpts = [original_to_disp(p) for p in points]
            if len(dpts) >= 2:
                for i in range(len(dpts) - 1):
                    cv2.line(canvas, dpts[i], dpts[i + 1], (0, 220, 255), 2, cv2.LINE_AA)
            if len(dpts) == 4:
                cv2.line(canvas, dpts[3], dpts[0], (0, 220, 255), 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(dpts):
                cv2.circle(canvas, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(canvas, str(i + 1), (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            help1 = f'{name}: click 4 screen corners in ANY order. Drag point to adjust. Right click = remove.'
            help2 = 'ENTER/S = confirm   R = reset   Q/ESC = cancel'
            cv2.putText(canvas, help1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, help2, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, canvas)

            if len(points) == 4:
                try:
                    cfg = make_perspective_roi(points)
                    warped = crop_roi(frame, cfg)
                    pw = min(1000, max(320, warped.shape[1]))
                    ph = max(1, int(round(warped.shape[0] * pw / max(1, warped.shape[1]))))
                    if pw != warped.shape[1]:
                        shown = cv2.resize(warped, (pw, ph), interpolation=cv2.INTER_AREA)
                    else:
                        shown = warped
                    preview = shown.copy()
                    cv2.putText(preview, f'{name} perspective preview', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow(preview_window, preview)
                except Exception:
                    pass
            else:
                try:
                    cv2.destroyWindow(preview_window)
                except Exception:
                    pass

            key = cv2.waitKey(30) & 0xFF
            if key in (13, ord('s')) and len(points) == 4:
                confirmed = True
                break
            if key == ord('r'):
                points.clear()
            if key in (27, ord('q')):
                break

        try:
            cv2.destroyWindow(window)
        except Exception:
            pass
        try:
            cv2.destroyWindow(preview_window)
        except Exception:
            pass
        if not confirmed:
            return None
        return make_perspective_roi(points)

    def _save_roi_previews(self, frame, rois):
        preview_dir = ensure_dir(WORK_DIR / 'roi_previews')
        for name, roi in rois.items():
            try:
                img = crop_roi(frame, roi)
                cv2.imwrite(str(preview_dir / f'{name}.jpg'), img)
            except Exception:
                pass

    def setup_rois(self):
        if not self._validate_inputs(need_gt=False):
            return
        frame = self._load_first_frame()
        self.settings = load_json(SETTINGS_PATH, self.settings) or self.settings
        existing = load_json(ROIS_PATH, {}) or {}
        names = configured_camera_names(self.settings, existing)
        rois = dict(existing)
        messagebox.showinfo(
            tr('4-point perspective ROI'),
            tr('For each CCTV, click the FOUR visible corners of that CCTV screen.\n\n'
            'The screen may be tilted or trapezoidal. Clicks can be in any order.\n'
            'Drag a point to fine-tune it, then press ENTER or S.\n'
            'The program will perspective-correct it into a rectangular CCTV image.')
        )
        for name in names:
            cfg = self._select_quad(frame, name)
            if cfg is None:
                cv2.destroyAllWindows()
                messagebox.showwarning(tr('Canceled'), tr(f'{name} selection canceled. Previous ROI settings were not overwritten.'))
                return
            rois[name] = cfg
        cv2.destroyAllWindows()
        save_json(ROIS_PATH, rois)
        self._save_roi_previews(frame, rois)
        messagebox.showinfo(tr('Saved'), tr(f'4-point perspective ROIs saved to {ROIS_PATH.name}.\nUse Preview warped CCTV to verify them.'))

    def preview_rois(self):
        if not self._validate_inputs(need_gt=False):
            return
        rois = load_json(ROIS_PATH, {})
        if not rois:
            messagebox.showerror(tr('Error'), tr('Set 4-point CCTV ROI first.'))
            return
        frame = self._load_first_frame()
        for name in configured_camera_names(self.settings, rois):
            if name not in rois:
                continue
            img = crop_roi(frame, rois[name])
            shown = img
            max_w = 1100
            if img.shape[1] > max_w:
                nh = max(1, int(round(img.shape[0] * max_w / img.shape[1])))
                shown = cv2.resize(img, (max_w, nh), interpolation=cv2.INTER_AREA)
            cv2.namedWindow(f'Warped {name}', cv2.WINDOW_NORMAL)
            cv2.imshow(f'Warped {name}', shown)
        messagebox.showinfo(tr('Preview'), tr('Warped CCTV windows are open. Check that no important parking area is cut off.\nPress any key inside an OpenCV window to close previews.'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _first_frame_detections(self, cctv, image):
        cfg = self.settings.get('detector_by_cctv', {}).get(cctv, self.settings['detector'])
        det = VehicleDetector(cfg)
        return det.detect(image)

    def setup_slots(self):
        if not self._validate_inputs(need_gt=False):
            return
        rois=load_json(ROIS_PATH,{})
        if not rois:
            messagebox.showerror(tr('Error'),tr('Set CCTV ROI first.'))
            return
        frame=self._load_first_frame();slots=load_slots(SLOTS_PATH)

        learned_keys={
            'anchor_center','anchor_center_raw','anchor_trusted','anchor_reason','anchor_clamped','anchor_std_xy_px','anchor_std_px',
            'anchor_shift_px','anchor_max_shift_px','anchor_sample_gate_px','nearest_manual_slot_distance_px','nearest_slot_distance_px',
            'expected_box_wh','assignment_radius_px','region','learned_box_samples','learned_center_samples_raw','learned_center_samples',
            'manual_point','template_initial','geometry_version',
        }
        def invalidate_geometry(slot):
            for k in learned_keys:slot.pop(k,None)
            slot['manual_point_edited']=True

        for cctv,roi in rois.items():
            image=crop_roi(frame,roi);self.status_var.set(tr(f'Loading YOLO for {cctv}...'));self.update_idletasks()
            try:dets=self._first_frame_detections(cctv,image)
            except Exception as e:
                messagebox.showerror(tr('YOLO error'),tr(str(e)));return
            last_mouse=[0,0];canceled=[False];drag_slot=[None]
            snapshot=[dict(x) for x in slots if x['cctv']==cctv]
            window=f'Slot editor - {cctv}'

            def nearest(x,y,max_dist=28):
                return _nearest_slot(slots,cctv,x,y,max_dist=max_dist)

            def draw():
                canvas=draw_detections(image,dets)
                for sl in [z for z in slots if z['cctv']==cctv]:
                    px,py=map(int,sl['point']);state=str(sl.get('initial_state','UNKNOWN')).upper();color=(0,255,0) if state=='OCCUPIED' else (255,128,0)
                    cv2.circle(canvas,(px,py),9,color,-1);cv2.circle(canvas,(px,py),12,(255,255,255),2)
                    cv2.putText(canvas,f"{sl['local_id']} {state[0]}",(px+12,max(18,py-10)),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,0,0),4,cv2.LINE_AA)
                    cv2.putText(canvas,f"{sl['local_id']} {state[0]}",(px+12,max(18,py-10)),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2,cv2.LINE_AA)
                cv2.rectangle(canvas,(4,4),(min(canvas.shape[1]-4,900),38),(15,15,15),-1)
                cv2.putText(canvas,'L empty:add | L existing:DRAG | R:delete | T:toggle | U:undo camera | S:save/next | Q:cancel',(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
                return canvas

            def mouse(event,x,y,flags,param):
                last_mouse[:]=[x,y]
                if event==cv2.EVENT_LBUTTONDOWN:
                    sl=nearest(x,y,30)
                    if sl is not None:
                        drag_slot[0]=sl
                    else:
                        initial='EMPTY'
                        for d in dets:
                            if d.x1<=x<=d.x2 and d.y1<=y<=d.y2:initial='OCCUPIED';break
                        sl={'local_id':next_local_id(slots,cctv),'cctv':cctv,'point':[int(x),int(y)],'initial_state':initial,'global_id':next_global_id(slots),'source':'MANUAL'}
                        slots.append(sl);drag_slot[0]=sl
                elif event==cv2.EVENT_MOUSEMOVE and drag_slot[0] is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
                    drag_slot[0]['point']=[int(max(0,min(image.shape[1]-1,x))),int(max(0,min(image.shape[0]-1,y)))];invalidate_geometry(drag_slot[0])
                elif event==cv2.EVENT_LBUTTONUP:
                    if drag_slot[0] is not None:
                        invalidate_geometry(drag_slot[0]);drag_slot[0]=None
                elif event==cv2.EVENT_RBUTTONDOWN:
                    sl=nearest(x,y,38)
                    if sl is not None:slots.remove(sl)

            cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.setMouseCallback(window,mouse)
            while True:
                cv2.imshow(window,draw());key=cv2.waitKey(30)&0xFF
                if key in (ord('s'),13):break
                if key in (ord('q'),27):canceled[0]=True;break
                if key==ord('t'):
                    sl=nearest(last_mouse[0],last_mouse[1],60)
                    if sl:sl['initial_state']='EMPTY' if str(sl.get('initial_state')).upper()=='OCCUPIED' else 'OCCUPIED'
                if key==ord('u'):
                    slots[:]=[x for x in slots if x['cctv']!=cctv]+[dict(x) for x in snapshot]
            cv2.destroyWindow(window)
            if canceled[0]:cv2.destroyAllWindows();return
        save_slots(SLOTS_PATH,slots);self.status_var.set(tr(f'Saved {len(slots)} local slots.'))
        messagebox.showinfo(tr('Saved'),tr(f'{len(slots)} local slots saved.\nManual points are never moved by learning. If you dragged a point, run Learn/ALL-IN-ONE again.'))

    def open_linker(self):
        """Visual multi-CCTV linker for duplicate physical parking spaces."""
        if not self._validate_inputs(need_gt=False):
            return
        slots = load_slots(SLOTS_PATH)
        if not slots:
            messagebox.showwarning(tr('No slots'), tr('Set slot points first.'))
            return
        rois = load_json(ROIS_PATH, {}) or {}
        if not rois:
            messagebox.showwarning(tr('No ROI'), tr('Set 4-point CCTV ROI first.'))
            return

        frame = self._load_first_frame()
        cctvs = [c for c in configured_camera_names(self.settings, rois) if c in rois]
        images = {}
        for c in cctvs:
            try:
                images[c] = crop_roi(frame, rois[c])
            except Exception as e:
                messagebox.showerror(tr('ROI error'), tr(f'{c}: {e}'))
                return

        overlap_warning_rows = []
        warn_path = LEARN_DIR / 'slot_overlap_warnings.csv'
        if warn_path.exists():
            try:
                import pandas as pd
                overlap_warning_rows = pd.read_csv(warn_path, encoding='utf-8-sig').to_dict('records')
            except Exception:
                overlap_warning_rows = []

        win = tk.Toplevel(self)
        win.title(tr('Visual duplicate-slot linker'))
        try:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            win.geometry(f'{max(1100, min(sw-80, 1700))}x{max(720, min(sh-100, 980))}')
        except Exception:
            win.geometry('1500x900')
        win.minsize(980, 650)

        selected = set()
        canvas_meta = {}
        photo_refs = {}
        linked_palette = [
            '#2E86DE','#E67E22','#27AE60','#8E44AD','#C0392B','#16A085',
            '#D4AC0D','#5D6D7E','#CA6F1E','#1ABC9C','#AF7AC5','#2E4053',
            '#7DCEA0','#EC7063','#5DADE2','#F4D03F','#A569BD','#48C9B0'
        ]

        def members_by_gid():
            groups = {}
            for s in slots:
                gid = str(s.get('global_id') or s['local_id'])
                groups.setdefault(gid, []).append(s)
            return groups

        def color_for_gid(gid, count):
            if count <= 1:
                return '#8A8A8A'
            import hashlib
            idx = int(hashlib.md5(gid.encode('utf-8')).hexdigest()[:8], 16) % len(linked_palette)
            return linked_palette[idx]

        header = ttk.Frame(win)
        header.pack(fill='x', padx=10, pady=(8,4))
        ttk.Label(
            header,
            text=tr('Click matching physical parking spaces to link them. Same color=same GLOBAL SLOT, yellow border=selected, orange/red dashed line=learned overlap warning.'),
            font=('', 10, 'bold')
        ).pack(side='left', anchor='w')
        status = tk.StringVar(value=tr('Click slots to select them.'))
        ttk.Label(header, textvariable=status).pack(side='right', anchor='e')

        views_outer = ttk.Frame(win)
        views_outer.pack(fill='both', expand=True, padx=8, pady=4)
        for col in range(len(cctvs)):
            views_outer.columnconfigure(col, weight=1, uniform='cctv')
        views_outer.rowconfigure(0, weight=1)
        canvases = {}

        def slot_short_name(local_id):
            t = str(local_id)
            return 'S' + t.split('_S',1)[1] if '_S' in t else t

        def fit_image(cctv, target_w, target_h):
            img = images[cctv]
            h, w = img.shape[:2]
            usable_w = max(220, target_w - 10)
            usable_h = max(180, target_h - 10)
            scale = min(usable_w / max(1,w), usable_h / max(1,h), 1.0)
            dw = max(1, int(round(w * scale)))
            dh = max(1, int(round(h * scale)))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((dw, dh), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(pil), scale, dw, dh

        def nearest_slot_on_canvas(cctv, x, y, max_px=26):
            meta = canvas_meta.get(cctv)
            if not meta:
                return None
            scale = meta['scale']
            ox, oy = meta['offset']
            best, best_d = None, 1e9
            for s in slots:
                if s['cctv'] != cctv:
                    continue
                px, py = s['point']
                sx, sy = ox + float(px)*scale, oy + float(py)*scale
                d = ((sx-x)**2 + (sy-y)**2)**0.5
                if d < best_d:
                    best, best_d = s, d
            return best if best_d <= max_px else None

        def refresh_table():
            for iid in tree.get_children():
                tree.delete(iid)
            groups = members_by_gid()
            for gid in sorted(groups):
                mem = groups[gid]
                by_cam = {c: [] for c in cctvs}
                for s in mem:
                    by_cam.setdefault(s['cctv'], []).append(slot_short_name(s['local_id']))
                vals = [gid] + [', '.join(sorted(by_cam.get(c, []))) or '-' for c in cctvs] + [str(len(mem))]
                tree.insert('', 'end', iid=gid, values=vals, tags=('linked',) if len(mem) >= 2 else ('single',))

        def redraw_view(cctv):
            canvas = canvases[cctv]
            cw = max(260, canvas.winfo_width())
            ch = max(240, canvas.winfo_height())
            photo, scale, dw, dh = fit_image(cctv, cw, ch)
            photo_refs[cctv] = photo
            ox, oy = max(0,(cw-dw)//2), max(0,(ch-dh)//2)
            canvas_meta[cctv] = {'scale': scale, 'offset': (ox,oy)}
            canvas.delete('all')
            canvas.create_rectangle(0,0,cw,ch,fill='#171717',outline='')
            canvas.create_image(ox,oy,image=photo,anchor='nw')
            groups = members_by_gid()
            for s in [z for z in slots if z['cctv'] == cctv]:
                gid = str(s.get('global_id') or s['local_id'])
                color = color_for_gid(gid, len(groups.get(gid, [])))
                px, py = s['point']
                x, y = ox + float(px)*scale, oy + float(py)*scale
                sel = s['local_id'] in selected
                r = 9 if sel else 7
                canvas.create_oval(x-r,y-r,x+r,y+r,fill=color,outline='#FFD400' if sel else '#FFFFFF',width=4 if sel else 2)
                label = f"{slot_short_name(s['local_id'])} / {gid}"
                tid = canvas.create_text(x+11,y-11,text=tr(label),anchor='sw',fill='white',font=('Consolas',9,'bold'))
                bbox = canvas.bbox(tid)
                if bbox:
                    bg = canvas.create_rectangle(bbox[0]-2,bbox[1]-1,bbox[2]+2,bbox[3]+1,fill='#111111',outline='')
                    canvas.tag_lower(bg, tid)
            # After learning, visualize slot-overlap warnings directly in the linker.
            by_lid = {z['local_id']: z for z in slots if z['cctv'] == cctv}
            for wr in overlap_warning_rows:
                if str(wr.get('cctv','')) != cctv:
                    continue
                a = by_lid.get(str(wr.get('slot_a',''))); b = by_lid.get(str(wr.get('slot_b','')))
                if not a or not b:
                    continue
                ap = a.get('anchor_center') or a['point']; bp = b.get('anchor_center') or b['point']
                ax, ay = ox + float(ap[0])*scale, oy + float(ap[1])*scale
                bx, by = ox + float(bp[0])*scale, oy + float(bp[1])*scale
                sev = str(wr.get('severity','WARNING')).upper()
                wc = '#FF3030' if sev == 'CRITICAL' else '#FF9F1A'
                canvas.create_line(ax, ay, bx, by, fill=wc, width=3, dash=(5,3))
                canvas.create_text((ax+bx)/2, (ay+by)/2, text=tr('! '+sev), fill=wc, font=('Consolas',8,'bold'))
            canvas.create_text(10,10,text=tr(cctv.upper()),anchor='nw',fill='white',font=('',12,'bold'))

        def redraw_all():
            for c in cctvs:
                redraw_view(c)
            refresh_table()
            status.set(tr('Selected: ' + ', '.join(sorted(selected)) if selected else 'Click slots to select them.'))

        def on_canvas_click(cctv, event):
            s = nearest_slot_on_canvas(cctv, event.x, event.y)
            if s is None:
                return
            lid = s['local_id']
            if lid in selected:
                selected.remove(lid)
            else:
                selected.add(lid)
            redraw_all()

        def select_group_from_tree(event=None):
            gids = tree.selection()
            if not gids:
                return
            groups = members_by_gid()
            selected.clear()
            for gid in gids:
                for s in groups.get(gid, []):
                    selected.add(s['local_id'])
            redraw_all()

        def clear_selection():
            selected.clear()
            try:
                tree.selection_remove(tree.selection())
            except Exception:
                pass
            redraw_all()

        def link_selected():
            chosen = [s for s in slots if s['local_id'] in selected]
            if len(chosen) < 2:
                messagebox.showwarning(tr('Selection required'), tr('Select at least two slots representing the same physical parking space.'), parent=win)
                return
            groups = members_by_gid()
            gids = {str(s.get('global_id') or s['local_id']) for s in chosen}
            all_members = []
            seen = set()
            for gid in gids:
                for s in groups.get(gid, []):
                    if s['local_id'] not in seen:
                        all_members.append(s)
                        seen.add(s['local_id'])
            valid = sorted(g for g in gids if g.startswith('G') and g[1:].isdigit())
            gid = valid[0] if valid else next_global_id(slots)
            for s in all_members:
                s['global_id'] = gid
            save_slots(SLOTS_PATH, slots)
            selected.clear()
            redraw_all()
            status.set(tr(f'{gid}: {len(all_members)}개 로컬 슬롯을 같은 실제 주차면으로 연결했습니다.'))

        def unlink_selected():
            chosen = [s for s in slots if s['local_id'] in selected]
            if not chosen:
                messagebox.showwarning(tr('Selection required'), tr('Select slots to unlink.'), parent=win)
                return
            for s in chosen:
                s['global_id'] = next_global_id(slots)
            save_slots(SLOTS_PATH, slots)
            n = len(chosen)
            selected.clear()
            redraw_all()
            status.set(tr(f'{n} slots을 각각 독립 GLOBAL SLOT으로 분리했습니다.'))

        def delete_selected():
            chosen = [s for s in slots if s['local_id'] in selected]
            if not chosen:
                messagebox.showwarning(tr('Selection required'), tr('Select slots to delete.'), parent=win)
                return
            names = ', '.join(s['local_id'] for s in chosen)
            question = f"선택한 {len(chosen)}개 로컬 슬롯을 삭제할까요?\n\n{names}"
            if not messagebox.askyesno(tr('Delete slots'), tr(question), parent=win):
                return
            lids = {s['local_id'] for s in chosen}
            slots[:] = [s for s in slots if s['local_id'] not in lids]
            save_slots(SLOTS_PATH, slots)
            selected.clear()
            redraw_all()
            status.set(tr(f'{len(lids)} slots deleted.'))

        def select_unlinked():
            groups = members_by_gid()
            selected.clear()
            for mem in groups.values():
                if len(mem) == 1:
                    selected.add(mem[0]['local_id'])
            redraw_all()

        for ci, cctv in enumerate(cctvs):
            panel = ttk.LabelFrame(views_outer, text=tr(cctv.upper()))
            panel.grid(row=0,column=ci,sticky='nsew',padx=4,pady=2)
            panel.rowconfigure(0,weight=1)
            panel.columnconfigure(0,weight=1)
            cv = tk.Canvas(panel,background='#171717',highlightthickness=0,cursor='hand2')
            cv.grid(row=0,column=0,sticky='nsew')
            cv.bind('<Button-1>', lambda e,c=cctv: on_canvas_click(c,e))
            cv.bind('<Configure>', lambda e,c=cctv: redraw_view(c))
            canvases[cctv] = cv

        lower = ttk.Frame(win)
        lower.pack(fill='x',padx=10,pady=(4,8))
        actions = ttk.Frame(lower)
        actions.pack(fill='x',pady=(0,5))
        ttk.Button(actions,text=tr('Link selected slots'),command=link_selected).pack(side='left',padx=3)
        ttk.Button(actions,text=tr('Unlink selected slots'),command=unlink_selected).pack(side='left',padx=3)
        ttk.Button(actions,text=tr('Delete selected slots'),command=delete_selected).pack(side='left',padx=3)
        ttk.Button(actions,text=tr('Clear selection'),command=clear_selection).pack(side='left',padx=3)
        ttk.Button(actions,text=tr('Select all unlinked slots'),command=select_unlinked).pack(side='left',padx=3)
        ttk.Button(actions,text=tr('Save and close'),command=lambda:(save_slots(SLOTS_PATH,slots),win.destroy())).pack(side='right',padx=3)

        columns = ['global'] + cctvs + ['count']
        tree_frame = ttk.Frame(lower)
        tree_frame.pack(fill='x')
        tree = ttk.Treeview(tree_frame,columns=columns,show='headings',height=7,selectmode='extended')
        tree.heading('global',text=tr('GLOBAL SLOT'))
        tree.column('global',width=100,anchor='center',stretch=False)
        for c in cctvs:
            tree.heading(c,text=tr(c.upper()))
            tree.column(c,width=180,anchor='center')
        tree.heading('count',text=tr('Views'))
        tree.column('count',width=65,anchor='center',stretch=False)
        tree.tag_configure('linked',background='#E8F6F3')
        tree.tag_configure('single',foreground='#707070')
        ysb = ttk.Scrollbar(tree_frame,orient='vertical',command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side='left',fill='x',expand=True)
        ysb.pack(side='right',fill='y')
        tree.bind('<<TreeviewSelect>>',select_group_from_tree)
        win.after(150,redraw_all)

    def review_overlap_warnings(self):
        csv_path = LEARN_DIR / 'slot_overlap_warnings.csv'
        if not csv_path.exists():
            messagebox.showinfo(tr('Overlap warnings'), tr('Run slot learning first.'))
            return
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except Exception as e:
            messagebox.showerror(tr('Error'), tr(str(e)))
            return
        if df.empty:
            messagebox.showinfo(tr('Overlap warnings'), tr('No slot-overlap warnings were found.'))
        else:
            crit = int((df['severity'].astype(str) == 'CRITICAL').sum()) if 'severity' in df else 0
            warn = len(df) - crit
            lines = []
            for _, r in df.head(12).iterrows():
                lines.append(f"{r.get('cctv','')} {r.get('slot_a','')} <-> {r.get('slot_b','')} | IoU={float(r.get('region_iou',0)):.2f} | {r.get('severity','')}")
            messagebox.showwarning(tr('Overlap warnings'), tr(f'CRITICAL={crit}, WARNING={warn}\n\n' + '\n'.join(lines) + ('\n...' if len(df)>12 else '')))
        camera_names = configured_camera_names(self.settings, load_json(ROIS_PATH,{}) or {})
        for cctv in camera_names:
            p = LEARN_DIR / f'{cctv}_slot_overlap_preview.jpg'
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    cv2.namedWindow(f'Overlap preview - {cctv}', cv2.WINDOW_NORMAL)
                    cv2.imshow(f'Overlap preview - {cctv}', img)
        if any((LEARN_DIR / f'{c}_slot_overlap_preview.jpg').exists() for c in camera_names):
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _change_language(self, event=None):
        language = self.language_var.get()
        # Read the latest settings so a Validation Center save is not lost.
        settings = load_json(SETTINGS_PATH, self.settings) or self.settings
        settings.setdefault('ui', {})['language'] = language
        save_json(SETTINGS_PATH, settings)
        self.settings = settings
        set_language(language)
        refresh_tree(self)

    def open_settings(self):
        win = tk.Toplevel(self)
        win.title(tr('Settings'))
        win.geometry('420x150')
        ttk.Label(win, text=tr('Language / 언어')).pack(padx=16, pady=(16, 8))
        selector = ttk.Combobox(win, textvariable=self.language_var,
                                values=('ko', 'en'), state='readonly', width=15)
        selector.pack()
        selector.bind('<<ComboboxSelected>>', self._change_language)
        ttk.Button(win, text=tr('Open advanced settings.json'),
                   command=self.open_advanced_settings).pack(pady=12)

    def open_advanced_settings(self):
        try:
            os.startfile(str(SETTINGS_PATH))
        except Exception:
            messagebox.showinfo(tr('Settings'), str(SETTINGS_PATH))

    def label_robustness_gt(self):
        runs=sorted(OUTPUT_ROOT.glob('run_*_ALL_IN_ONE'), key=lambda x:x.stat().st_mtime if x.exists() else 0, reverse=True)
        review=None
        for r in runs:
            cand=r/'robustness'/'gt_review'
            if (cand/'robustness_gt_template.csv').is_file():
                review=cand;break
        if review is None:
            messagebox.showinfo(tr('Robustness GT'), tr('Run ALL-IN-ONE once first. The robustness sample crops/template are created after Stage 5.'))
            return
        try:
            result=label_robustness_ground_truth(review,ROBUSTNESS_GT_PATH)
            messagebox.showinfo(tr('Robustness GT saved'), (
                tr(f"Saved to:\n{result['path']}\n\n"
                f"Labeled {result['labeled']}/{result['total']} | occupied={result['occupied']} | empty={result['empty']} | unknown={result.get('unknown',0)}\n"
                + ('Balance looks usable.\n\n' if result.get('balance_ok', True) else 'Warning: labels are still strongly class-imbalanced; v16.2 metrics will flag this.\n\n')
                + 'Run ALL-IN-ONE again. Only the robustness stage is invalidated; detector/evidence caches remain reusable.')
            ))
        except Exception as e:
            messagebox.showerror(tr('Robustness GT error'), tr(f'{type(e).__name__}: {e}'))

    @staticmethod
    def _fmt_duration(sec):
        if sec is None or sec != sec or sec < 0:
            return '--:--:--'
        sec=int(round(float(sec)));h=sec//3600;m=(sec%3600)//60;s=sec%60
        return f'{h:02d}:{m:02d}:{s:02d}'

    def _progress(self, ratio, text, step_elapsed=None, step_eta=None):
        now=time.monotonic(); overall=max(0.0,min(1.0,float(ratio)))
        total_elapsed=(now-self._pipeline_started_at) if self._pipeline_started_at else None
        # Track actual current-stage elapsed even for direct CACHE HIT / stage messages.
        m=re.match(r'\s*(\d+/\d+)',str(text))
        step_key=m.group(1) if m else str(text).split('|',1)[0].strip()
        if step_key != self._step_key:
            self._step_key=step_key; self._step_started_at=now
        if step_elapsed is None and self._step_started_at is not None:
            step_elapsed=max(0.0,now-self._step_started_at)
        if self._pipeline_started_at and step_eta is None and ('CACHE HIT' in str(text) or 'REUSED' in str(text) or overall>=0.999):
            step_eta=0.0
        total_remaining=None
        if total_elapsed is not None and overall>=0.015 and overall<0.999:
            total_remaining=max(0.0,total_elapsed*(1.0-overall)/overall)
        elif overall>=0.999:
            total_remaining=0.0
        def update():
            self.progress_var.set(overall*100); self.status_var.set(tr(text))
            self.timing_var.set(
                tr(f'Total Elapsed {self._fmt_duration(total_elapsed)} | '
                f'Step Elapsed {self._fmt_duration(step_elapsed)} | '
                f'Step ETA {self._fmt_duration(step_eta)} | '
                f'Total Remaining {self._fmt_duration(total_remaining)}')
            )
        self.after(0, update)

    def _run_thread(self, fn, done_message=None):
        def worker():
            try:
                fn()
                if done_message:
                    self.after(0, lambda msg=done_message: messagebox.showinfo(tr('Done'), tr(msg)))
            except Exception as e:
                # Exception variables are cleared when the except block exits in Python 3.
                # Capture strings now instead of closing over `e` in a later Tk callback.
                err_msg = f'{type(e).__name__}: {e}'
                err_trace = traceback.format_exc()
                try:
                    LAST_ERROR_PATH.write_text(err_trace, encoding='utf-8')
                except Exception:
                    pass
                self.after(0, lambda msg=err_msg: messagebox.showerror(
                    tr('Error'),
                    tr(f'{msg}\n\nFull traceback saved to:\n{LAST_ERROR_PATH}')
                ))
            finally:
                self.after(0, lambda: self.progress_var.set(0))
        threading.Thread(target=worker, daemon=True).start()

    def start_detector_tuning(self):
        if not self._validate_inputs(need_gt=True):
            return
        rois = load_json(ROIS_PATH, {})
        if not rois:
            messagebox.showerror(tr('Error'), tr('Set 4-point CCTV ROI first.'))
            return
        self.settings = load_json(SETTINGS_PATH, self.settings)
        video = self.video_var.get().strip()
        gt_path = self.gt_var.get().strip()

        grid = self.settings.get('detector_sweep', {})
        models = grid.get('models', [])
        imgsz = grid.get('imgsz', [])
        conf = grid.get('conf', [])
        tiling = grid.get('tiling', [])
        msg = (
            'This runs the configured YOLO grid only on perspective-warped DEV frames (time < 15:00).\n'
            f'Models: {models}\nimgsz: {imgsz}\nconf: {conf}\ntiling: {tiling}\n\n'
            'It can take a long time. TEST 15:00-20:30 is not used. Continue?'
        )
        if not messagebox.askyesno(tr('Warped detector DEV tuning'), tr(msg)):
            return

        def job():
            result = run_warped_detector_sweep(
                video, rois, gt_path, self.settings, str(DETECTOR_TUNE_DIR), progress=self._progress
            )
            if bool(self.settings.get('detector_sweep', {}).get('apply_best_to_settings', True)):
                save_json(SETTINGS_PATH, self.settings)
            best = result.get('best')
            lines = []
            if best is not None and not best.empty:
                for _, r in best.iterrows():
                    lines.append(
                        f"{r['cctv']}: {r['model']} / {int(r['imgsz'])} / conf {float(r['conf']):.2f} / "
                        f"tile={bool(r['tiling'])} | DEV exact {float(r['DEV_exact_rate'])*100:.1f}% MAE {float(r['DEV_MAE']):.3f}"
                    )
            text = 'Warped detector tuning complete.\n\n' + '\n'.join(lines)
            self.after(0, lambda: messagebox.showinfo(tr('Detector tuning complete'), tr(text)))
        self._run_thread(job, 'Warped detector DEV tuning finished and best settings were applied.')

    def open_detector_tuning_folder(self):
        try:
            os.startfile(str(DETECTOR_TUNE_DIR))
        except Exception:
            messagebox.showinfo(tr('Detector tuning folder'), tr(str(DETECTOR_TUNE_DIR)))

    def review_learning_diagnostics(self):
        p = LEARN_DIR / 'slot_learning_diagnostics.csv'
        if not p.exists():
            messagebox.showinfo(tr('Learning diagnostics'), tr('Run slot learning first.'))
            return
        try:
            import pandas as pd
            df = pd.read_csv(p, encoding='utf-8-sig')
        except Exception as e:
            messagebox.showerror(tr('Error'), tr(str(e)))
            return
        if df.empty:
            messagebox.showinfo(tr('Learning diagnostics'), tr('No slot diagnostics were generated.'))
            return
        trusted = int(df['anchor_trusted'].fillna(0).astype(int).sum()) if 'anchor_trusted' in df else 0
        fallback = int(len(df)-trusted)
        max_shift = float(df['anchor_shift_px'].max()) if 'anchor_shift_px' in df else 0.0
        few = df[df.get('learned_center_samples', 0) < int(self.settings.get('matching', {}).get('min_anchor_samples', 30))] if 'learned_center_samples' in df else df.iloc[0:0]
        worst = df.sort_values(['anchor_trusted','learned_center_samples'], ascending=[True, True]).head(12)
        lines = []
        for _, r in worst.iterrows():
            lines.append(
                f"{r.get('cctv','')} {r.get('local_id','')} | samples={int(r.get('learned_center_samples',0))} "
                f"shift={float(r.get('anchor_shift_px',0)):.1f}px | {r.get('anchor_reason','')}"
            )
        messagebox.showinfo(
            tr('Learning diagnostics'),
            tr(f'Total slots={len(df)}\nTrusted learned anchors={trusted}\nManual fallback={fallback}\n'
            f'Low-sample slots={len(few)}\nMax applied anchor shift={max_shift:.1f}px\n\n' + '\n'.join(lines))
        )
        opened = False
        for cctv in configured_camera_names(self.settings, load_json(ROIS_PATH,{}) or {}):
            vp = LEARN_DIR / f'{cctv}_manual_voronoi_preview.jpg'
            if vp.exists():
                img = cv2.imread(str(vp))
                if img is not None:
                    cv2.namedWindow(f'Manual Voronoi preview - {cctv}', cv2.WINDOW_NORMAL)
                    cv2.imshow(f'Manual Voronoi preview - {cctv}', img)
                    opened = True
        if opened:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def start_learning(self):
        if not self._validate_inputs(need_gt=False):
            return
        rois = load_json(ROIS_PATH, {})
        if not rois:
            messagebox.showerror(tr('Error'),tr('Set ROI first.'))
            return
        if not load_slots(SLOTS_PATH):
            messagebox.showerror(tr('Error'),tr('Set slot points first.'))
            return
        self.settings = load_json(SETTINGS_PATH, self.settings)
        def job():
            result = learn_slots_and_candidates(
                self.video_var.get().strip(), rois, str(SLOTS_PATH), self.settings,
                str(LEARN_DIR), progress=self._progress
            )
            wc = len(result.get('warnings', []))
            self._progress(1.0, f'Learning complete. Slot-overlap warnings: {wc}.')
        self._run_thread(job, 'Manual-point slot learning complete. Review diagnostics/candidates/warnings before the final experiment.')

    def review_candidates(self):
        path=LEARN_DIR/'candidate_slots.json';data=load_json(path,{})
        if not data:
            messagebox.showinfo(tr('Candidates'),tr('No candidate data. Run slot learning first.'));return
        if not self._validate_inputs(need_gt=False):return
        rois=load_json(ROIS_PATH,{});frame=self._load_first_frame();slots=load_slots(SLOTS_PATH)
        for cctv,cands in data.items():
            if not cands:continue
            img=crop_roi(frame,rois[cctv]);window=f'Candidate review - {cctv}';last_mouse=[0,0]
            def nearest_candidate(x,y,max_dist=42):
                best=None;bd=1e9
                for c in cands:
                    px,py=c['point'];d=((px-x)**2+(py-y)**2)**0.5
                    if d<bd:best,bd=c,d
                return best if bd<=max_dist else None
            def draw():
                canvas=img.copy()
                for sl in [z for z in slots if z['cctv']==cctv]:
                    px,py=map(int,sl['point']);cv2.circle(canvas,(px,py),7,(255,210,0),2)
                    cv2.putText(canvas,sl['local_id'],(px+9,max(18,py-8)),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2,cv2.LINE_AA)
                for c in cands:
                    px,py=map(int,c['point']);st=str(c.get('status','PENDING'));kind=str(c.get('candidate_kind','NEW')).upper()
                    color=(0,0,255) if kind=='FALSE' or st.startswith('FALSE') else ((180,70,220) if kind=='DUPLICATE' or st.startswith('DUPLICATE') else ((0,255,0) if st=='ACCEPTED' else (0,165,255)))
                    cv2.circle(canvas,(px,py),11,color,3)
                    suffix=f" -> {c.get('target_local_id','')}" if kind=='DUPLICATE' else ''
                    text=f"{c['candidate_id']} {kind}/{st}{suffix}"
                    cv2.putText(canvas,text,(px+12,py+18),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),4,cv2.LINE_AA);cv2.putText(canvas,text,(px+12,py+18),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2,cv2.LINE_AA)
                cv2.rectangle(canvas,(4,4),(min(canvas.shape[1]-4,980),38),(15,15,15),-1)
                cv2.putText(canvas,'PENDING only: L=accept NEW | R=reject. Confirmed FALSE/DUPLICATE are locked. S=save/next',(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
                return canvas
            def mouse(event,x,y,flags,param):
                last_mouse[:]=[x,y];c=nearest_candidate(x,y)
                if not c:return
                if str(c.get('status','PENDING')) not in ('PENDING','ACCEPTED','REJECTED'):return
                if event==cv2.EVENT_LBUTTONDOWN:c['status']='ACCEPTED';c['candidate_kind']='NEW'
                elif event==cv2.EVENT_RBUTTONDOWN:c['status']='REJECTED';c['candidate_kind']='FALSE'
            cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.setMouseCallback(window,mouse)
            while True:
                cv2.imshow(window,draw());k=cv2.waitKey(30)&0xFF
                if k in (ord('s'),13,ord('q'),27):break
            cv2.destroyWindow(window)
        # Only explicitly ACCEPTED NEW candidates can become new slots. Confirmed DUPLICATE/FALSE remain advisory metadata.
        existing_candidate_ids={sl.get('candidate_id') for sl in slots}
        for cctv,cands in data.items():
            for c in cands:
                if c.get('status')!='ACCEPTED' or str(c.get('candidate_kind','NEW')).upper()!='NEW' or c['candidate_id'] in existing_candidate_ids:continue
                slots.append({'local_id':next_local_id(slots,cctv),'cctv':cctv,'point':c['point'],'initial_state':'OCCUPIED','global_id':next_global_id(slots),'source':'AUTO_CANDIDATE_USER_ACCEPTED','candidate_id':c['candidate_id'],'region':c.get('region')})
        save_json(path,data);save_slots(SLOTS_PATH,slots)
        messagebox.showinfo(tr('Saved'),tr('Candidate decisions saved. Confirmed duplicate candidates do NOT create another global parking space. Ground truth is never auto-edited from model detections.'))

    def _mapped_progress(self, start_ratio, end_ratio, stage_name):
        span = float(end_ratio) - float(start_ratio)
        started = time.monotonic()
        def cb(ratio, text):
            r=max(0.0,min(1.0,float(ratio)))
            elapsed=max(0.0,time.monotonic()-started)
            remain=None
            if r>=0.02 and r<0.999:
                remain=max(0.0,elapsed*(1.0-r)/r)
            elif r>=0.999:
                remain=0.0
            self._progress(float(start_ratio) + span*r, f'{stage_name} | {text}', step_elapsed=elapsed, step_eta=remain)
        return cb

    def _package_current_run(self, run_dir, evidence_dir, gt_path, slot_gt_path=''):
        safe_copy(SETTINGS_PATH, run_dir/'settings.json')
        safe_copy(ROIS_PATH, run_dir/'rois.json')
        safe_copy(SLOTS_PATH, run_dir/'slots.json')
        safe_copy(gt_path, run_dir/'ground_truth.csv')
        safe_copy(evidence_dir/'slot_evidence.csv', run_dir/'slot_evidence.csv')
        safe_copy(evidence_dir/'frame_detection_summary.csv', run_dir/'frame_detection_summary.csv')
        safe_copy(evidence_dir/'vehicle_tracks.csv', run_dir/'vehicle_tracks.csv')
        safe_copy(evidence_dir/'slot_crop_geometry.csv', run_dir/'slot_crop_geometry.csv')
        safe_copy(evidence_dir/'slot_detector_diagnostics.csv', run_dir/'slot_detector_diagnostics.csv')
        safe_copy(evidence_dir/'segmentation_evidence.csv', run_dir/'segmentation_evidence.csv')
        safe_copy(evidence_dir/'segmentation_summary.csv', run_dir/'segmentation_summary.csv')
        safe_copy(evidence_dir/'segmentation_condition_thresholds.csv', run_dir/'segmentation_condition_thresholds.csv')
        safe_copy(evidence_dir/'segmentation_condition_counts.csv', run_dir/'segmentation_condition_counts.csv')
        safe_copy(evidence_dir/'SEGMENTATION_UNAVAILABLE.txt', run_dir/'SEGMENTATION_UNAVAILABLE.txt')
        safe_copy(LEARN_DIR/'slot_overlap_warnings.csv', run_dir/'slot_overlap_warnings.csv')
        safe_copy(LEARN_DIR/'slot_learning_diagnostics.csv', run_dir/'slot_learning_diagnostics.csv')
        safe_copy(LEARN_DIR/'candidate_slots.csv', run_dir/'candidate_slots.csv')
        safe_copy(LEARN_DIR/'candidate_slots.json', run_dir/'candidate_slots.json')
        safe_copy(LEARN_DIR/'GT_REVIEW_REQUIRED.txt', run_dir/'GT_REVIEW_REQUIRED.txt')
        safe_copy(LEARN_DIR/'gt_review_candidate_times.csv', run_dir/'gt_review_candidate_times.csv')
        for cctv_name in configured_camera_names(self.settings, load_json(ROIS_PATH,{}) or {}):
            safe_copy(LEARN_DIR/f'{cctv_name}_manual_voronoi_preview.jpg', run_dir/f'{cctv_name}_manual_voronoi_preview.jpg')
            safe_copy(LEARN_DIR/f'{cctv_name}_slot_overlap_preview.jpg', run_dir/f'{cctv_name}_slot_overlap_preview.jpg')
        for tune_name in ['warped_detector_sweep.csv','warped_detector_best.csv','warped_detector_selected.json','WARPED_DETECTOR_REPORT.txt']:
            safe_copy(DETECTOR_TUNE_DIR/tune_name, run_dir/tune_name)
        if slot_gt_path and Path(slot_gt_path).exists():
            safe_copy(slot_gt_path, run_dir/'slot_gt_events.csv')
        upload_files=[]
        for name in [
            'REPORT.txt','metrics_dev_test.csv','state_search_leaderboard.csv','selected_state_params.json',
            'selected_count_timeseries.csv','selected_slot_timeseries.csv','selected_global_slot_timeseries.csv',
            'selected_state_transitions.csv','test_error_cases.csv','settings.json','rois.json','slots.json','ground_truth.csv',
            'slot_evidence.csv','frame_detection_summary.csv','vehicle_tracks.csv','slot_overlap_warnings.csv',
            'slot_learning_diagnostics.csv','candidate_slots.csv','candidate_slots.json','GT_REVIEW_REQUIRED.txt','gt_review_candidate_times.csv','slot_gt_events.csv','slot_crop_geometry.csv','slot_detector_diagnostics.csv','detection_mode_comparison.csv','mode_selection_audit.json','CANDIDATE_REVIEW_REQUIRED.txt',
            'warped_detector_sweep.csv','warped_detector_best.csv','warped_detector_selected.json','WARPED_DETECTOR_REPORT.txt','camera_detector_metrics.csv','camera_detector_timeseries.csv',
            'ALL_IN_ONE_LOG.txt','CACHE_INFO.txt','temporal_variant_comparison.csv','baseline_count_timeseries.csv','baseline_global_slot_timeseries.csv','baseline_slot_timeseries.csv','transition_guard_count_timeseries.csv','transition_guard_global_slot_timeseries.csv','transition_guard_slot_timeseries.csv',
            'slot_level_metrics.csv','slot_level_by_slot.csv','slot_level_comparison.csv','slot_level_test_errors.csv',
            'segmentation_evidence.csv','segmentation_summary.csv','segmentation_condition_thresholds.csv','segmentation_condition_counts.csv','SEGMENTATION_UNAVAILABLE.txt',
            'seg_assist_count_timeseries.csv','seg_assist_global_slot_timeseries.csv','seg_assist_slot_timeseries.csv','seg_assist_state_transitions.csv',
            'empty_ref_assist_global_slot_timeseries.csv','empty_ref_assist_slot_timeseries.csv','empty_ref_assist_state_transitions.csv','EMPTY_REF_EXPERIMENT.txt',
            'transition_guard_state_transitions.csv','baseline_state_transitions.csv','STATE_SEARCH_REPAIR.log','EVIDENCE_CACHE_SANITY.txt','NON_REGRESSION_CHECK.txt',
            'candidate_v164_slot_timeseries.csv','candidate_v164_global_slot_timeseries.csv','candidate_v164_count_timeseries.csv',
            'candidate_v164_metrics.csv','candidate_v164_transition_events.csv','candidate_v164_event_balanced_metrics.csv',
            'candidate_v164_summary.json','candidate_v164_config.json','CANDIDATE_v16_4_REPORT.txt',
            'episode_evaluation_mask.csv','episode_metrics.csv','repeat_stability.csv','REPEAT_STABILITY_REPORT.txt',
            'window_metrics.csv','window_comparison.csv','window_evaluation_summary.json','WINDOW_EVALUATION_REPORT.txt'
        ]:
            pp=run_dir/name
            if pp.exists():
                upload_files.append((pp,name))
        for cctv_name in configured_camera_names(self.settings, load_json(ROIS_PATH,{}) or {}):
            for suffix in ['manual_voronoi_preview.jpg','slot_overlap_preview.jpg']:
                pp=run_dir/f'{cctv_name}_{suffix}'
                if pp.exists(): upload_files.append((pp,pp.name))
        for root_name in ['screenshots','ground_truth_review','candidate_review','duplicate_review','robustness']:
            tree_root = run_dir/root_name
            if tree_root.exists():
                for sp in tree_root.rglob('*'):
                    if not sp.is_file():
                        continue
                    rel = str(sp.relative_to(run_dir)).replace('\\','/')
                    # v15.2 keeps the complete GT review locally but sends only a compact,
                    # high-value subset to ChatGPT: contact sheets, change/event frames,
                    # upload_review checkpoints, and the upload-specific index/readme.
                    if rel.startswith('ground_truth_review/'):
                        allowed=(
                            rel.startswith('ground_truth_review/contact_sheets/') or
                            rel.startswith('ground_truth_review/events/') or
                            rel.startswith('ground_truth_review/upload_review/') or
                            rel.endswith('ground_truth_review/gt_review_upload_index.csv') or
                            rel.endswith('ground_truth_review/README_GT_REVIEW.txt')
                        )
                        if not allowed:
                            continue
                    if rel.startswith('robustness/gt_review/clean_samples/'):
                        # Keep all labeling crops locally; upload only the compact template/readme.
                        continue
                    upload_files.append((sp, rel))
        zip_paths(run_dir/'UPLOAD_TO_CHATGPT.zip', upload_files)

    def start_all_in_one(self):
        if not self._validate_inputs(need_gt=True):
            return
        rois = load_json(ROIS_PATH, {})
        slots = load_slots(SLOTS_PATH)
        if not rois or not slots:
            messagebox.showerror(tr('Setup required'), tr('Complete 4-point ROI, slot points, and duplicate-slot linking first.'))
            return
        if not messagebox.askyesno(
            tr('Start ALL-IN-ONE'),
            tr('This will run the full unattended pipeline:\n\n'
            '1) Tune YOLO on warped DEV CCTV\n'
            '2) Learn manual-point/Voronoi slots\n'
            '3) Extract FULL CCTV at 1 FPS + AUX crop only on ambiguous slots\n'
            '4) Run lazy RAW + per-CCTV relative ADAPTIVE slot segmentation only on transition-relevant ambiguity\n'
            '5) Run bounded degradation robustness test (low-res / glare / causal temporal de-moire)\n'
            '6) Compare SAFE_BASELINE / TRANSITION_GUARD / SEG_ASSIST, then auto-run v16.4 Candidate\n'
            '7) Add episode/cut + repeated-source stability metrics, capture review figures, and build ZIP\n\n'
            'No more clicks are required after starting. Candidate slots are discovered but NOT auto-accepted. '
            'You can minimize this window while it runs.\n\nContinue?')
        ):
            return

        self._pipeline_started_at=time.monotonic()
        self._step_started_at=self._pipeline_started_at
        self._step_key=None
        self.timing_var.set(tr('Total Elapsed 00:00:00 | Step Elapsed 00:00:00 | Step ETA --:--:-- | Total Remaining --:--:--'))
        self.settings = load_json(SETTINGS_PATH, self.settings)
        self.settings.setdefault('robustness_experiment',{})['gt_file']=str(ROBUSTNESS_GT_PATH)
        self.cache_status_var.set(tr('CACHE | auditing current dataset/profile...'))
        video = self.video_var.get().strip()
        gt_path = self.gt_var.get().strip()
        slot_gt_path = self.slot_gt_var.get().strip()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = ensure_dir(OUTPUT_ROOT/f'run_{stamp}_ALL_IN_ONE')
        evidence_dir = ensure_dir(run_dir/'evidence')
        log_path = run_dir/'ALL_IN_ONE_LOG.txt'

        def log(text):
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with log_path.open('a', encoding='utf-8') as f:
                f.write(f'[{ts}] {text}\n')

        def job():
            log(f'Parking Research Agent {APP_VERSION} ALL-IN-ONE started. v16.2 SAFE + v16.4 candidate validation.')
            cache_cfg=self.settings.get('pipeline_cache',{}) or {}
            cache_enabled=bool(cache_cfg.get('enabled',True))
            slots_for_sig=load_slots(SLOTS_PATH)
            cache_lines=['Parking Slot Engine v16.2 cache audit','',f'shared_cache={CACHE_ROOT}']

            # Stage 1: detector tuning cache.
            det_sig=_stable_signature('detector',video,gt_path,rois,slots_for_sig,self.settings)
            det_sig_path=DETECTOR_TUNE_DIR/'v15_cache_signature.json'
            det_selected_path=DETECTOR_TUNE_DIR/'warped_detector_selected.json'
            det_cache_hit=(cache_enabled and bool(cache_cfg.get('reuse_detector_tuning',True)) and
                           _cache_sig_matches(det_sig_path,det_sig) and det_selected_path.is_file())
            reuse_existing=(not bool(cache_cfg.get('retune_detector_each_all_in_one',False)) and
                            all(str(c) in (self.settings.get('detector_by_cctv',{}) or {}) for c in rois.keys()))
            if reuse_existing:
                # v14 ships with the latest validated per-CCTV detector selection from the
                # previous full sweep. ALL-IN-ONE reuses it by default; the dedicated Tune
                # button remains available whenever the camera/ROI setup changes.
                self._progress(0.42,'1/7 Detector tuning | REUSED existing per-CCTV detector settings')
                log('Stage 1/7 CONFIG REUSE: existing detector_by_cctv settings kept; full sweep skipped.')
                cache_lines.append('detector_tuning=CONFIG_REUSE_NO_SWEEP')
            elif det_cache_hit:
                selected=load_json(det_selected_path,{}) or {}
                if selected:
                    self.settings['detector_by_cctv']=selected
                    save_json(SETTINGS_PATH,self.settings)
                self._progress(0.42,'1/7 Detector tuning | CACHE HIT - reused previous detector sweep')
                log('Stage 1/7 CACHE HIT: detector tuning reused.')
                cache_lines.append('detector_tuning=CACHE_HIT')
            else:
                log('Stage 1/7: warped detector DEV tuning started.')
                run_warped_detector_sweep(
                    video, rois, gt_path, self.settings, str(DETECTOR_TUNE_DIR),
                    progress=self._mapped_progress(0.00, 0.42, '1/7 Detector tuning')
                )
                if bool(self.settings.get('detector_sweep', {}).get('apply_best_to_settings', True)):
                    save_json(SETTINGS_PATH, self.settings)
                _write_cache_sig(det_sig_path,det_sig,'detector')
                log('Stage 1/7 complete. Selected detector settings were saved.')
                cache_lines.append('detector_tuning=CACHE_MISS_RECOMPUTED')

            # Reload after detector selection before computing the slot-learning signature.
            self.settings=load_json(SETTINGS_PATH,self.settings)
            self.settings.setdefault('robustness_experiment',{})['gt_file']=str(ROBUSTNESS_GT_PATH)
            slots_for_sig=load_slots(SLOTS_PATH)
            learn_sig=_stable_signature('learning',video,gt_path,rois,slots_for_sig,self.settings)
            learn_cache_dir=ensure_dir(CACHE_ROOT/f'learning_{learn_sig[:20]}')
            learn_sig_path=learn_cache_dir/'cache_signature.json'
            # v15.4 cross-version learning migration. Older v15.x builds did not write a
            # learning signature into work/learned, so also validate sibling app config/UI state.
            if not (learn_cache_dir/'slot_learning_diagnostics.csv').is_file():
                try:
                    for sibling in APP_DIR.parent.glob('parking_slot_engine_v*/work/learned'):
                        if sibling.resolve()==LEARN_DIR.resolve(): continue
                        if not ((sibling/'slot_learning_diagnostics.csv').is_file() and (sibling/'templates').is_dir()): continue
                        sp=sibling/'v15_cache_signature.json'; compatible=_cache_sig_matches(sp,learn_sig)
                        if not compatible:
                            sibling_app=sibling.parent.parent
                            sui=load_json(sibling_app/'ui_state.json',{}) or {}; sset=load_json(sibling_app/'settings.json',{}) or {}; sroi=load_json(sibling_app/'rois.json',{}) or {}
                            try: sslots=load_slots(sibling_app/'slots.json')
                            except Exception: sslots=[]
                            same_video=str(Path(str(sui.get('video_path',''))).resolve())==str(Path(video).resolve()) if sui.get('video_path') else False
                            compatible=(same_video and sroi==rois and _manual_slot_payload(sslots)==_manual_slot_payload(slots_for_sig)
                                        and sset.get('detector_by_cctv',{})==self.settings.get('detector_by_cctv',{})
                                        and sset.get('matching',{})==self.settings.get('matching',{})
                                        and float(sset.get('learn_sample_sec',-1))==float(self.settings.get('learn_sample_sec',-2)))
                        if compatible:
                            shutil.copytree(sibling,learn_cache_dir,dirs_exist_ok=True)
                            _write_cache_sig(learn_sig_path,learn_sig,'learning')
                            break
                except Exception:
                    pass
            learn_cache_hit=(cache_enabled and bool(cache_cfg.get('reuse_slot_learning',True)) and
                             _cache_sig_matches(learn_sig_path,learn_sig) and
                             (learn_cache_dir/'slot_learning_diagnostics.csv').is_file() and
                             (learn_cache_dir/'templates').is_dir())
            if learn_cache_hit:
                shutil.copytree(learn_cache_dir,LEARN_DIR,dirs_exist_ok=True)
                restored,restore_source,restore_stats=_restore_learning_metadata_from_cache(SLOTS_PATH,LEARN_DIR,learn_cache_dir)
                if restored:
                    cand_json=load_json(LEARN_DIR/'candidate_slots.json',{}) or {}
                    learned={'candidates':cand_json,'warnings':[]}
                    self._progress(0.58,'2/7 Slot learning | SHARED CACHE HIT - geometry + template metadata restored')
                    log(f'Stage 2/7 SHARED CACHE HIT: slot learning reused; metadata source={restore_source}.')
                    cache_lines.append(f'slot_learning=SHARED_CACHE_HIT_METADATA_RESTORED:{restore_source}')
                else:
                    # Old v16 cache could contain template JPEGs without the learned slots metadata
                    # needed to resolve them. One fresh learning pass is safer than silently using
                    # visual_diff_initial=1.0 for every frame.
                    log(f'Stage 2/7 CACHE REJECTED: learned metadata incomplete; forcing one fresh learn. stats={restore_stats}')
                    cache_lines.append('slot_learning=CACHE_REJECTED_MISSING_METADATA')
                    learn_cache_hit=False
            if not learn_cache_hit:
                log('Stage 2/7: manual-point/Voronoi slot learning started.')
                learned=learn_slots_and_candidates(
                    video, rois, str(SLOTS_PATH), self.settings, str(LEARN_DIR),
                    progress=self._mapped_progress(0.42, 0.58, '2/7 Slot learning')
                )
                # v16.2 snapshot: shared learning cache must carry learned slot metadata as well
                # as template JPEGs. This makes cross-version reuse self-contained.
                _write_learned_slot_snapshot(SLOTS_PATH,LEARN_DIR)
                _write_cache_sig(LEARN_DIR/'v15_cache_signature.json',learn_sig,'learning')
                shutil.copytree(LEARN_DIR,learn_cache_dir,dirs_exist_ok=True)
                _write_cache_sig(learn_sig_path,learn_sig,'learning')
                log('Stage 2/7 slot learning recomputed and moved to shared cache with metadata snapshot.')
                cache_lines.append('slot_learning=CACHE_MISS_RECOMPUTED_SHARED_WITH_METADATA')

            candidate_rows=[]
            for cand_cctv,cand_list in (learned.get('candidates',{}) or {}).items():
                for cand in cand_list: candidate_rows.append((cand_cctv,cand))
            review_path=run_dir/'CANDIDATE_REVIEW_REQUIRED.txt'
            if candidate_rows:
                lines=[
                    'Parking Slot Engine v16.2 - Candidate Slot Review','',
                    'These are advisory only. They were NOT added to the real parking-slot map and did NOT change the final slot count.',
                    'Review these after the unattended run before deciding whether a manual slot was missed.',''
                ]
                for cand_cctv,cand in candidate_rows:
                    lines.append(f"{cand.get('candidate_id','?')} | {cand_cctv} | kind={cand.get('candidate_kind','NEW')} | status={cand.get('status','PENDING')} | point={cand.get('point')} | target={cand.get('target_local_id','')}/{cand.get('target_global_id','')} | presence={float(cand.get('presence_ratio',0.0))*100:.1f}% | std={float(cand.get('center_std_px',0.0)):.2f}px")
                review_path.write_text('\n'.join(lines),encoding='utf-8')
            pending_count=sum(1 for _,c in candidate_rows if str(c.get('status','PENDING'))=='PENDING')
            log(f"Stage 2/7 complete. pending_candidates={pending_count}; confirmed duplicate/false candidates remain advisory metadata.")

            # Stage 3: expensive evidence cache. Temporal-rule changes do not invalidate it.
            self.settings=load_json(SETTINGS_PATH,self.settings)
            self.settings.setdefault('robustness_experiment',{})['gt_file']=str(ROBUSTNESS_GT_PATH)
            slots_for_sig=load_slots(SLOTS_PATH)
            ev_sig=_stable_signature('evidence',video,gt_path,rois,slots_for_sig,self.settings)
            ev_cache_dir=ensure_dir(CACHE_ROOT/f'evidence_{ev_sig[:20]}')
            ev_sig_path=ev_cache_dir/'cache_signature.json'
            ev_names=['slot_evidence.csv','frame_detection_summary.csv','vehicle_tracks.csv','slot_crop_geometry.csv','slot_detector_diagnostics.csv']
            ev_cache_hit=(cache_enabled and bool(cache_cfg.get('reuse_evidence',True)) and
                          _cache_sig_matches(ev_sig_path,ev_sig) and all((ev_cache_dir/x).is_file() for x in ev_names[:3]))
            gt=load_ground_truth(gt_path)
            evidence_rebuilt_due_sanity=False
            if ev_cache_hit:
                copied=_copy_cache_files(ev_cache_dir,evidence_dir,ev_names)
                evidence=pd.read_csv(evidence_dir/'slot_evidence.csv',encoding='utf-8-sig')
                sane,sanity_reason,sanity_stats=_evidence_sanity(evidence,load_slots(SLOTS_PATH),LEARN_DIR)
                if sane:
                    _write_evidence_sanity_report(run_dir/'EVIDENCE_CACHE_SANITY.txt',True,sanity_reason,sanity_stats,'CACHE_REUSED')
                    self._progress(0.84,f'3/7 FULL + recovery-only AUX evidence | CACHE HIT + SANITY PASS ({len(copied)} files)')
                    log(f'Stage 3/7 CACHE HIT + SANITY PASS: reused selective evidence ({len(evidence)} rows).')
                    cache_lines.append('evidence=CACHE_HIT_SANITY_PASS')
                else:
                    evidence_rebuilt_due_sanity=True
                    ev_cache_hit=False
                    _write_evidence_sanity_report(run_dir/'EVIDENCE_CACHE_SANITY.txt',False,sanity_reason,sanity_stats,'CACHE_REJECTED_RECOMPUTE')
                    log(f'Stage 3/7 CACHE REJECTED by sanity check: {sanity_reason}; stats={sanity_stats}')
                    cache_lines.append(f'evidence=CACHE_REJECTED_SANITY:{sanity_reason}')
            if not ev_cache_hit:
                log('Stage 3/7: FULL CCTV + recovery-only AUX recheck 1 FPS evidence extraction started.')
                evidence=extract_evidence(
                    video, rois, str(SLOTS_PATH), self.settings, str(LEARN_DIR), str(evidence_dir),
                    progress=self._mapped_progress(0.58, 0.84, '3/7 FULL + recovery-only AUX evidence')
                )
                sane,sanity_reason,sanity_stats=_evidence_sanity(evidence,load_slots(SLOTS_PATH),LEARN_DIR)
                _write_evidence_sanity_report(run_dir/'EVIDENCE_CACHE_SANITY.txt',sane,sanity_reason,sanity_stats,'RECOMPUTED_AND_VALIDATED' if sane else 'RECOMPUTED_BUT_INVALID_ABORT')
                if not sane:
                    raise RuntimeError(f'Fresh evidence failed sanity check: {sanity_reason}. See EVIDENCE_CACHE_SANITY.txt')
                _copy_cache_files(evidence_dir,ev_cache_dir,ev_names)
                _write_cache_sig(ev_sig_path,ev_sig,'evidence')
                log(f'Stage 3/7 evidence recomputed, sanity-checked, and cached ({len(evidence)} rows).')
                cache_lines.append('evidence=CACHE_MISS_RECOMPUTED_SANITY_PASS')
            write_camera_detector_metrics(evidence_dir/'frame_detection_summary.csv',gt,self.settings,run_dir)
            log('Stage 3/7 complete. Raw FULL detector metrics by CCTV were also written.')

            # Stage 4: separate selective segmentation cache. Changing temporal logic does not rerun segmentation.
            seg_sig=_stable_signature('segmentation',video,gt_path,rois,load_slots(SLOTS_PATH),self.settings)
            seg_cache_dir=ensure_dir(CACHE_ROOT/f'segmentation_{seg_sig[:20]}')
            seg_sig_path=seg_cache_dir/'cache_signature.json'
            seg_names=['segmentation_evidence.csv','segmentation_summary.csv','SEGMENTATION_UNAVAILABLE.txt']
            seg_cache_hit=(cache_enabled and bool(cache_cfg.get('reuse_segmentation',True)) and
                           (not evidence_rebuilt_due_sanity) and
                           _cache_sig_matches(seg_sig_path,seg_sig) and (seg_cache_dir/'segmentation_evidence.csv').is_file() and not (seg_cache_dir/'SEGMENTATION_UNAVAILABLE.txt').is_file())
            if seg_cache_hit:
                _copy_cache_files(seg_cache_dir,evidence_dir,seg_names)
                seg_cached=pd.read_csv(evidence_dir/'segmentation_evidence.csv',encoding='utf-8-sig')
                seg_sane,seg_reason,seg_stats=_evidence_sanity(seg_cached,load_slots(SLOTS_PATH),LEARN_DIR)
                if seg_sane:
                    evidence=seg_cached
                    cache_lines.append('segmentation=CACHE_HIT_SANITY_PASS')
                    self._progress(0.91,'4/7 RAW+ADAPTIVE segmentation | CACHE HIT + SANITY PASS')
                    log(f'Stage 4/7 CACHE HIT + SANITY PASS: segmentation assist reused ({len(evidence)} rows).')
                else:
                    seg_cache_hit=False
                    log(f'Stage 4/7 segmentation CACHE REJECTED: inherited evidence sanity failed: {seg_reason}; stats={seg_stats}')
                    cache_lines.append(f'segmentation=CACHE_REJECTED_SANITY:{seg_reason}')
            if not seg_cache_hit:
                log('Stage 4/7: lazy RAW + condition-specific ADAPTIVE slot-crop segmentation assist started.')
                evidence=extract_segmentation_assist(
                    video,rois,str(SLOTS_PATH),self.settings,evidence,str(evidence_dir),
                    progress=self._mapped_progress(0.84,0.91,'4/7 RAW+ADAPTIVE segmentation')
                )
                if (evidence_dir/'SEGMENTATION_UNAVAILABLE.txt').is_file():
                    cache_lines.append('segmentation=MODEL_UNAVAILABLE_NOT_CACHED')
                    log('Stage 4/7 segmentation model unavailable; result was not cached so a future run can retry.')
                else:
                    _copy_cache_files(evidence_dir,seg_cache_dir,seg_names)
                    _write_cache_sig(seg_sig_path,seg_sig,'segmentation')
                    cache_lines.append('segmentation=CACHE_MISS_RECOMPUTED')
                    log(f'Stage 4/7 segmentation complete and cached ({len(evidence)} rows).')
            # Stage 5: bounded degradation-robustness experiment. This is separate from
            # the production state engine and explicitly labels synthetic stresses as stress tests.
            robustness_dir=ensure_dir(run_dir/'robustness')
            rob_sig=_stable_signature('robustness',video,gt_path,rois,load_slots(SLOTS_PATH),self.settings)
            rob_cache_dir=ensure_dir(CACHE_ROOT/f'robustness_{rob_sig[:20]}')
            rob_sig_path=rob_cache_dir/'cache_signature.json'
            rob_required=rob_cache_dir/'degradation_robustness_summary.csv'
            rob_cache_hit=(cache_enabled and bool(cache_cfg.get('reuse_robustness',True)) and
                           (not evidence_rebuilt_due_sanity) and
                           _cache_sig_matches(rob_sig_path,rob_sig) and rob_required.is_file())
            if rob_cache_hit:
                shutil.copytree(rob_cache_dir,robustness_dir,dirs_exist_ok=True)
                cache_lines.append('robustness=CACHE_HIT')
                self._progress(0.95,'5/7 Degradation robustness | CACHE HIT')
                log('Stage 5/7 CACHE HIT: degradation robustness experiment reused.')
            else:
                log('Stage 5/7: bounded adaptive segmentation + causal temporal de-moire robustness experiment started.')
                try:
                    run_degradation_robustness_experiment(
                        video,rois,str(SLOTS_PATH),self.settings,evidence,str(robustness_dir),
                        progress=self._mapped_progress(0.91,0.95,'5/7 degradation robustness')
                    )
                    # Cache only successful experiment output.
                    for old in rob_cache_dir.iterdir():
                        if old.name!='cache_signature.json':
                            if old.is_dir(): shutil.rmtree(old)
                            else: old.unlink()
                    shutil.copytree(robustness_dir,rob_cache_dir,dirs_exist_ok=True)
                    _write_cache_sig(rob_sig_path,rob_sig,'robustness')
                    cache_lines.append('robustness=CACHE_MISS_RECOMPUTED')
                    log('Stage 5/7 robustness experiment complete and cached.')
                except Exception as exc:
                    (robustness_dir/'ROBUSTNESS_UNAVAILABLE.txt').write_text(
                        f'Robustness experiment failed but the main parking engine can continue.\n{type(exc).__name__}: {exc}\n',encoding='utf-8')
                    cache_lines.append('robustness=FAILED_CONTINUED')
                    log(f'Stage 5/7 robustness experiment unavailable: {type(exc).__name__}: {exc}')

            (run_dir/'CACHE_INFO.txt').write_text('\n'.join(cache_lines)+'\n',encoding='utf-8')
            short_cache=[]
            for line in cache_lines:
                if '=' in line and not line.startswith('shared_cache='):
                    k,v=line.split('=',1); short_cache.append(f'{k}:{v.split(":",1)[0]}')
            self.after(0,lambda txt='CACHE | '+' | '.join(short_cache): self.cache_status_var.set(tr(txt)))

            log('Stage 6/7: SAFE_BASELINE vs TRANSITION_GUARD vs SEG_ASSIST DEV comparison and frozen TEST evaluation started.')
            state_result=search_state_parameters(
                evidence, gt, self.settings, str(run_dir),
                progress=self._mapped_progress(0.95, 0.985, '6/7 baseline / guard / SEG comparison'),
                slot_gt_events_path=slot_gt_path if slot_gt_path and Path(slot_gt_path).exists() else None
            )
            nr=_write_non_regression_check(run_dir,gt,state_result,self.settings)
            if nr.get('applicable') and not nr.get('passed'):
                log(f"Stage 6/7 NON-REGRESSION WARNING: SAFE_BASELINE TEST differs from the known reference: {nr}")
                cache_lines.append('non_regression=WARNING')
            elif nr.get('applicable'):
                log('Stage 6/7 non-regression PASS: SAFE_BASELINE restored to the known reference envelope.')
                cache_lines.append('non_regression=PASS')
            # Refresh cache audit after the non-regression result was appended.
            (run_dir/'CACHE_INFO.txt').write_text('\n'.join(cache_lines)+'\n',encoding='utf-8')
            log('Stage 6/7 temporal evaluation complete.')
            try:
                cand_summary=run_v164_candidate(Path(run_dir),RefinerConfig())
                log(f"Stage 6/7 v16.4 candidate complete: TEST exact={cand_summary.get('candidate_test_exact_rate')} MAE={cand_summary.get('candidate_test_mae')} decision={cand_summary.get('decision')}")
            except Exception as exc:
                (run_dir/'CANDIDATE_v16_4_UNAVAILABLE.txt').write_text(f'{type(exc).__name__}: {exc}\n',encoding='utf-8')
                log(f'Stage 6/7 v16.4 candidate unavailable: {type(exc).__name__}: {exc}')
            try:
                ep_path=write_episode_metrics(run_dir,self.settings)
                rep_path=write_repeat_stability(run_dir,self.settings)
                write_window_evaluation(run_dir,self.settings)
                log(f'Stage 6/7 validation extras: episode={ep_path} repeat={rep_path}')
            except Exception as exc:
                (run_dir/'VALIDATION_EXTRAS_UNAVAILABLE.txt').write_text(f'{type(exc).__name__}: {exc}\n',encoding='utf-8')
                log(f'Stage 6/7 validation extras unavailable: {type(exc).__name__}: {exc}')
            log('Stage 7/7: GT review package + automatic paper/debug figures started.')
            generate_ground_truth_review(video,rois,gt_path,str(SLOTS_PATH),self.settings,run_dir,LEARN_DIR)
            generate_research_screenshots(video, rois, str(SLOTS_PATH), self.settings, run_dir, LEARN_DIR, evidence_dir)
            log('Stage 7/7 review/screenshots complete.')
            self._package_current_run(run_dir, evidence_dir, gt_path, slot_gt_path)
            log(f'Packaging complete: {run_dir / "UPLOAD_TO_CHATGPT.zip"}')
            try:
                append_run(APP_VERSION, str(run_dir), {'mode':'ALL_IN_ONE','package':str(run_dir/'UPLOAD_TO_CHATGPT.zip')})
            except Exception as exc:
                log(f'Research history local journal warning: {type(exc).__name__}: {exc}')
            self._progress(1.0, f'ALL-IN-ONE complete | {run_dir / "UPLOAD_TO_CHATGPT.zip"}')
            self.after(0, lambda: messagebox.showinfo(
                tr('ALL-IN-ONE complete'),
                tr(f'Everything finished without intermediate clicks.\n\nResult folder:\n{run_dir}\n\nUpload UPLOAD_TO_CHATGPT.zip to ChatGPT.')
            ))

        self.all_in_one_btn.config(state='disabled')
        def wrapped_job():
            try:
                job()
            finally:
                self.after(0, lambda: self.all_in_one_btn.config(state='normal'))
        self._run_thread(wrapped_job, None)

    def start_experiment(self):
        if not self._validate_inputs(need_gt=True):
            return
        rois=load_json(ROIS_PATH,{})
        if not rois or not load_slots(SLOTS_PATH):
            messagebox.showerror(tr('Error'),tr('ROI and slots are required.'))
            return
        slots_now = load_slots(SLOTS_PATH)
        if not (LEARN_DIR/'templates').exists():
            messagebox.showerror(tr('Error'),tr('Run v16 slot learning first, or use ALL-IN-ONE.'))
            return
        stale = [x.get('local_id','?') for x in slots_now if x.get('geometry_version') != 'v12_manual_voronoi']
        if stale:
            messagebox.showerror(tr('Error'), tr('These slots need slot learning before v16 can run:\n' + ', '.join(stale[:12]) + (' ...' if len(stale)>12 else '') + '\n\nRun Learn manual-point slots again, or use ALL-IN-ONE.'))
            return
        self.settings=load_json(SETTINGS_PATH,self.settings)
        video=self.video_var.get().strip(); gt_path=self.gt_var.get().strip()
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir=ensure_dir(OUTPUT_ROOT/f'run_{stamp}')
        evidence_dir=ensure_dir(run_dir/'evidence')

        def job():
            gt=load_ground_truth(gt_path)
            evidence=extract_evidence(video,rois,str(SLOTS_PATH),self.settings,str(LEARN_DIR),str(evidence_dir),progress=self._mapped_progress(0.0,0.72,'FULL + recovery AUX'))
            evidence=extract_segmentation_assist(video,rois,str(SLOTS_PATH),self.settings,evidence,str(evidence_dir),progress=self._mapped_progress(0.72,0.84,'Selective segmentation'))
            self._progress(0.84,'Comparing SAFE_BASELINE / TRANSITION_GUARD / SEG_ASSIST on DEV...')
            slot_gt_path=self.slot_gt_var.get().strip()
            search_state_parameters(evidence,gt,self.settings,str(run_dir),progress=self._mapped_progress(0.84,0.94,'Temporal comparison'),slot_gt_events_path=slot_gt_path if slot_gt_path and Path(slot_gt_path).exists() else None)
            try:
                run_v164_candidate(Path(run_dir),RefinerConfig())
                write_episode_metrics(run_dir,self.settings)
                write_repeat_stability(run_dir,self.settings)
                write_window_evaluation(run_dir,self.settings)
            except Exception:
                pass
            write_camera_detector_metrics(evidence_dir/'frame_detection_summary.csv',gt,self.settings,run_dir)
            generate_ground_truth_review(video,rois,gt_path,str(SLOTS_PATH),self.settings,run_dir,LEARN_DIR)
            generate_research_screenshots(video, rois, str(SLOTS_PATH), self.settings, run_dir, LEARN_DIR, evidence_dir)
            self._package_current_run(run_dir,evidence_dir,gt_path,slot_gt_path)
            try:
                append_run(APP_VERSION, str(run_dir), {'mode':'FINAL_EXPERIMENT','package':str(run_dir/'UPLOAD_TO_CHATGPT.zip')})
            except Exception:
                pass
            self._progress(1.0,f"Complete. Upload {run_dir/'UPLOAD_TO_CHATGPT.zip'}")
            self.after(0,lambda: messagebox.showinfo(tr('Experiment complete'),tr(f"Result folder:\n{run_dir}\n\nUpload UPLOAD_TO_CHATGPT.zip to ChatGPT.")))
        self._run_thread(job,'Experiment finished.')


if __name__ == '__main__':
    App().mainloop()
