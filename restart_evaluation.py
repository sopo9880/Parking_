"""Restart experiments: fresh inference history, frozen fitting, causal recovery candidate."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_RECOVERY={'duration_sec':30,'window_sec':10,'min_observation_sec':10,
                  'min_samples':8,'hit_ratio':0.75,'min_conf':0.10,
                  'single_camera_conf':0.25,'max_motion_px':24,'max_speed_px_s':14}


def recovery_gate(history,elapsed,cfg):
    if elapsed<float(cfg.get('min_observation_sec',10)) or len(history)<int(cfg.get('min_samples',8)):
        return False
    hits=[r for r in history if int(r.get('full_detected',0)) and float(r.get('full_det_conf',0))>=float(cfg.get('min_conf',.1))]
    if len(hits)/len(history)<float(cfg.get('hit_ratio',.75)): return False
    centers=np.array([[r.get('full_center_x',np.nan),r.get('full_center_y',np.nan)] for r in hits],dtype=float)
    if len(centers)<2 or not np.isfinite(centers).all(): return False
    # Range rejects sustained maneuvers rather than relying on a stable Track ID.
    if np.linalg.norm(np.ptp(centers,axis=0))>float(cfg.get('max_motion_px',24)): return False
    speeds=[float(r.get('full_track_speed_px_s',0)) for r in hits]
    return bool(np.isfinite(speeds).all() and max(speeds)<=float(cfg.get('max_speed_px_s',14)))


def recovery_metrics(trace,start,stable_samples=3):
    d=trace.sort_values('time_sec');err=d.error.to_numpy();times=d.time_sec.to_numpy()
    exact=err==0; first=np.flatnonzero(exact)
    stable=None
    for i in range(len(d)-stable_samples+1):
        if exact[i:i+stable_samples].all():
            stable=i;break
    result={'N':len(d),'exact_rate':float(exact.mean()) if len(d) else None,
            'mae':float(np.abs(err).mean()) if len(d) else None,
            'max_abs_error':float(np.abs(err).max()) if len(d) else None,
            'over_rate':float((err>0).mean()) if len(d) else None,
            'under_rate':float((err<0).mean()) if len(d) else None,
            'time_to_first_exact_sec':float(times[first[0]]-start) if len(first) else None,
            'time_to_stable_exact_sec':float(times[stable]-start) if stable is not None else None,
            'stable_confirmed_at_sec':float(times[stable+stable_samples-1]-start) if stable is not None else None,
            'stable_samples':stable_samples,'never_stabilized':stable is None,
            'post_stable_exact_rate':float(exact[stable:].mean()) if stable is not None else None}
    for seconds in (30,60,120):
        mask=times<start+seconds
        result[f'first_{seconds}s_N']=int(mask.sum())
        result[f'first_{seconds}s_mae']=float(np.abs(err[mask]).mean()) if mask.any() else None
        result[f'first_{seconds}s_exact_rate']=float(exact[mask].mean()) if mask.any() else None
    return result


def _run_restart_experiments(video,rois,gt,settings,fit_dir,progress=None):
    from slot_engine import (extract_evidence,extract_segmentation_assist,load_slots,save_slots,
        _run_state_engine_v13_safe,_v16_baseline_params,evaluate_state_output)
    from split_protocol import protocol_context
    from transition_refiner_v164 import apply_transition_refiner,build_global_trace,RefinerConfig
    fit=Path(fit_dir);root=Path(settings['_restart_output_dir']);root.mkdir(parents=True,exist_ok=True)
    cfg=settings.get('validation',{}).get('restart_experiments',{})
    if int(cfg.get('stable_samples',3))<2: raise ValueError('Stable exact requires at least two GT checkpoints')
    if gt.time_sec.duplicated().any(): raise ValueError('Duplicate GT timestamps are not allowed')
    horizon=float(settings.get('eval_end_sec',1230))
    starts=sorted(set(float(t) for t in cfg.get('starts_sec',[0,330,660,900])))
    starts=sorted(set(starts+[float(settings.get('active_split_protocol',{}).get('test',[0])[0])]))
    if not starts or any(not np.isfinite(t) or not 0<=t<horizon for t in starts):
        raise ValueError('Restart times must be within the evaluation horizon')
    freeze=(fit/'selected_state_params.json').read_bytes()
    freeze_record=json.loads((fit/'selection_freeze.json').read_text(encoding='utf-8'))
    if hashlib.sha256(freeze).hexdigest()!=freeze_record['sha256']:
        raise ValueError('Fitted parameter freeze hash mismatch')
    if settings.get('detector_by_cctv')!=freeze_record.get('detectors'):
        raise ValueError('Fitted detector settings differ from selection freeze')
    quality=fit/'evidence/segmentation_condition_thresholds.csv'
    thresholds={}
    if quality.exists() and quality.stat().st_size>5:
        for row in pd.read_csv(quality).to_dict('records'):
            thresholds[row.pop('cctv')]=row
    rows=[]
    warm=float(settings.get('evaluation_warmup_sec',10))
    with protocol_context(None):
        for index,start in enumerate(starts):
            end=min(horizon,start+float(cfg.get('duration_sec',330)))
            window_gt=gt[(gt.time_sec>=start)&((gt.time_sec<=end) if end==horizon else (gt.time_sec<end))].copy()
            if window_gt.empty: raise ValueError(f'No ground truth for restart at {start}s')
            out=root/f'restart_{start:g}';out.mkdir(exist_ok=True)
            cold=copy.deepcopy(settings);cold['inference_start_sec']=start;cold['eval_end_sec']=end
            cold['frozen_segmentation_quality_thresholds']=thresholds
            cold['unlabeled_restart']=True
            # Neither file labels nor earlier runtime occupancy initialize a restart.
            slots=load_slots(fit/'slots.json')
            for slot in slots: slot['initial_state']='UNKNOWN'
            save_slots(out/'slots.json',slots)
            if progress: progress(index/len(starts),f'Restart @{start:g}s | fresh detector history')
            evidence=extract_evidence(video,rois,str(out/'slots.json'),cold,str(fit/'learned'),str(out/'evidence'),progress)
            evidence=extract_segmentation_assist(video,rois,str(out/'slots.json'),cold,evidence,str(out/'evidence'),progress)
            if evidence.empty or evidence.time_sec.min()<start: raise RuntimeError('Invalid fresh restart evidence')
            baseline_params={**_v16_baseline_params(settings),'unlabeled_restart':True}
            traces={}
            for variant,params in [('RESET_SAFE',baseline_params),('RESET_RECOVERY',{**baseline_params,'startup_recovery':DEFAULT_RECOVERY})]:
                local,global_,transitions=_run_state_engine_v13_safe(evidence,params)
                local.to_csv(out/f'{variant}_slot_timeseries.csv',index=False,encoding='utf-8-sig')
                transitions.to_csv(out/f'{variant}_transitions.csv',index=False,encoding='utf-8-sig')
                traces[variant]=global_
                if variant=='RESET_SAFE':
                    refined,_=apply_transition_refiner(local,RefinerConfig())
                    traces['RESET_CANDIDATE']=build_global_trace(refined)
            for variant,filename in [('CONTINUOUS_SAFE','baseline_global_slot_timeseries.csv'),('CONTINUOUS_CANDIDATE','candidate_v164_global_slot_timeseries.csv')]:
                traces[variant]=pd.read_csv(fit/filename)
            counts={}
            for variant,global_ in traces.items():
                if not set(window_gt.time_sec).issubset(set(global_.time_sec)):
                    raise ValueError('GT checkpoints missing from inference timestamps; align restart times and evidence sampling')
                # Inclusive startup: cold-start metrics cannot hide the first 10 seconds.
                trace=evaluate_state_output(global_,window_gt,0,0)['timeseries']
                trace.to_csv(out/f'{variant}_count_timeseries.csv',index=False,encoding='utf-8-sig')
                counts[variant]=trace
                for scope,part in [('INCLUSIVE_STARTUP',trace),('AFTER_RESTART_WARMUP',trace[trace.time_sec>=start+warm])]:
                    rows.append({'restart_sec':start,'end_sec':end,'variant':variant,'scope':scope,
                                 **recovery_metrics(part,start,int(cfg.get('stable_samples',3)))})
            paired=counts['CONTINUOUS_SAFE'][['time_sec','occupied_pred']].merge(
                counts['RESET_SAFE'][['time_sec','occupied_pred']],on='time_sec',suffixes=('_continuous','_reset'))
            paired['restart_count_difference']=paired.occupied_pred_reset-paired.occupied_pred_continuous
            paired.to_csv(out/'restart_state_count_difference.csv',index=False,encoding='utf-8-sig')
            state_pairs=traces['CONTINUOUS_SAFE'][['time_sec','global_id','state']].merge(
                traces['RESET_SAFE'][['time_sec','global_id','state']],on=['time_sec','global_id'],suffixes=('_continuous','_reset'))
            state_pairs['state_differs']=state_pairs.state_continuous!=state_pairs.state_reset
            state_pairs.to_csv(out/'restart_slot_state_difference.csv',index=False,encoding='utf-8-sig')
            (out/'restart_audit.json').write_text(json.dumps({'start_sec':start,'end_sec':end,
                'runtime_history_reused':False,'initial_state':'UNKNOWN','quality_fit':'frozen DEV thresholds',
                'selected_parameters_sha256':hashlib.sha256(freeze).hexdigest(),
                'recovery_candidate_config':DEFAULT_RECOVERY,'gt_used_for_bootstrap':False,
                'fresh_inference_start_sec':float(evidence.time_sec.min()),
                'segmentation_available':not (out/'evidence/SEGMENTATION_UNAVAILABLE.txt').exists()},indent=2),encoding='utf-8')
    assert freeze==(fit/'selected_state_params.json').read_bytes(),'Restart evaluation changed fitted selection'
    pd.DataFrame(rows).to_csv(root/'restart_metrics.csv',index=False,encoding='utf-8-sig')
    (root/'RESTART_REPORT.txt').write_text('Continuous versus fresh restart with frozen fitting.\n'
        'Startup-inclusive and matched post-restart warm-up metrics are separate. Stable exact means '
        'three consecutive available GT checkpoints, not permanent slot correctness. Never recovered is missing, not zero.\n'
        'RESET_RECOVERY is experimental and does not replace SAFE_BASELINE. Count recovery is not slot-level validation.\n'
        'Historical same-video windows are not independent hold-out. Random-restart coverage is a configured deterministic schedule.\n',encoding='utf-8')
    return root


def run_restart_experiments(video,rois,gt,settings,fit_dir,progress=None):
    from datetime import datetime
    root=Path(fit_dir)/'restart_experiments'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    settings=copy.deepcopy(settings)
    settings['_restart_output_dir']=str(root)
    try:
        result=_run_restart_experiments(video,rois,gt,settings,fit_dir,progress)
        (root/'restart_summary.json').write_text(json.dumps({'status':'completed','metrics':'restart_metrics.csv','scope':'offline historically reviewed same-video; recovery candidate only'}),encoding='utf-8')
        if (root/'failure.json').exists(): (root/'failure.json').unlink()
        return result
    except Exception as exc:
        root.mkdir(parents=True,exist_ok=True)
        (root/'restart_summary.json').write_text(json.dumps({'status':'failed','error':str(exc),'partial_outputs_retained':True}),encoding='utf-8')
        raise
