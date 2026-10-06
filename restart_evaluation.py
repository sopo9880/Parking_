"""Restart experiments: fresh inference history, frozen fitting, causal recovery candidate."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_RECOVERY={'preserve_safe_entry':True,'duration_sec':30,'window_sec':10,'min_observation_sec':10,
                  'min_samples':8,'hit_ratio':0.75,'min_conf':0.10,
                  'single_camera_conf':0.25,'max_motion_px':24,'max_speed_px_s':14}
LEGACY_RECOVERY={k:v for k,v in DEFAULT_RECOVERY.items() if k!='preserve_safe_entry'}
LEGACY_RECOVERY['preserve_safe_entry']=False


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
    # Count agreement on three checkpoints can relapse immediately afterwards.
    exact_runs=[];run=0
    for value in exact:
        run=run+1 if value else 0;exact_runs.append(run)
    relapse_start=stable+stable_samples if stable is not None else len(d)
    relapse=np.flatnonzero(~exact[relapse_start:])+relapse_start
    result.update({'post_stable_mae':float(np.abs(err[stable:]).mean()) if stable is not None else None,
        'time_to_first_relapse_sec':float(times[relapse[0]]-start) if len(relapse) else None,
        'relapse_count':int(sum(bool(exact[i-1] and not exact[i]) for i in range(relapse_start,len(d)))),
        'longest_exact_run_samples':max(exact_runs,default=0),
        'tail_exact_samples':exact_runs[-1] if exact_runs else 0})
    for seconds in (30,60,120):
        mask=times<start+seconds
        result[f'first_{seconds}s_N']=int(mask.sum())
        result[f'first_{seconds}s_mae']=float(np.abs(err[mask]).mean()) if mask.any() else None
        result[f'first_{seconds}s_exact_rate']=float(exact[mask].mean()) if mask.any() else None
    return result


def _run_restart_experiments(video,rois,gt,settings,fit_dir,progress=None):
    from slot_engine import (extract_evidence,extract_segmentation_assist,load_slots,save_slots,
        _run_state_engine_v13_safe,_run_state_engine_v15_transition_guard,
        _run_state_engine_v15_1_seg_assist,_v16_baseline_params,evaluate_state_output)
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
    from experimental_selection import load_variant_params,freeze_dev_selection
    if (fit/'DEV_selection_variant_comparison.csv').exists() and not (fit/'dev_selected_experimental_params.json').exists():
        freeze_dev_selection(fit)
    guard_params=load_variant_params(fit,'TRANSITION_GUARD') if (fit/'DEV_selection_variant_comparison.csv').exists() else None
    experimental=None
    if (fit/'dev_selected_experimental_params.json').exists():
        raw=(fit/'dev_selected_experimental_params.json').read_bytes()
        experimental_freeze=json.loads((fit/'dev_selection_freeze.json').read_text())
        if hashlib.sha256(raw).hexdigest()!=experimental_freeze['sha256']:
            raise ValueError('Experimental DEV parameter freeze mismatch')
        experimental=json.loads(raw)
    from manual_initialization import read_snapshots,load_snapshot,apply_snapshot,write_snapshot,snapshot_audit,occupancy_bounds
    manual=cfg.get('manual_init',{})
    snapshots={}
    if manual.get('enabled',False):
        gids={str(slot['global_id']) for slot in load_slots(fit/'slots.json')}
        raw_snapshot=Path(manual.get('path','')).read_bytes()
        manual_source_sha256=hashlib.sha256(raw_snapshot).hexdigest()
        snapshot_rows,_=read_snapshots(manual['path'],raw=raw_snapshot)
        snapshots={start:load_snapshot(manual['path'],start,gids,rows=snapshot_rows) for start in starts}
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
            runners=[('RESET_SAFE',_run_state_engine_v13_safe,baseline_params),
                ('RESET_RECOVERY',_run_state_engine_v13_safe,{**baseline_params,'startup_recovery':DEFAULT_RECOVERY}),
                ('RESET_RECOVERY_V1655',_run_state_engine_v13_safe,{**baseline_params,'startup_recovery':LEGACY_RECOVERY})]
            if guard_params:
                runners.append(('RESET_TRANSITION_GUARD',_run_state_engine_v15_transition_guard,{**guard_params,'unlabeled_restart':True}))
            if experimental:
                runner={'SAFE_BASELINE':_run_state_engine_v13_safe,'TRANSITION_GUARD':_run_state_engine_v15_transition_guard,
                        'SEG_ASSIST':_run_state_engine_v15_1_seg_assist}[experimental['temporal_variant']]
                runners.append(('RESET_DEV_SELECTED',runner,{**experimental,'unlabeled_restart':True}))
            manual_states=snapshots.get(start)
            manual_audit={'status':'disabled' if not manual.get('enabled',False) else 'skipped_missing_exact_time_snapshot','start_sec':start}
            if manual_states is not None:
                write_snapshot(out/'manual_init_snapshot.csv',start,manual_states)
                manual_audit={'status':'initialized',**snapshot_audit(manual['path'],start,manual_states,manual.get('source','operator_snapshot'),source_sha256=manual_source_sha256)}
                runners += [('RESET_UNKNOWN',_run_state_engine_v13_safe,{**baseline_params,'preserve_unknown':True}),
                            ('RESET_MANUAL_INIT',_run_state_engine_v13_safe,{**baseline_params,'preserve_unknown':True,'manual_initialization':True})]
            (out/'manual_init_audit.json').write_text(json.dumps(manual_audit,indent=2),encoding='utf-8')
            for variant,runner,params in runners:
                input_evidence=apply_snapshot(evidence,manual_states) if variant=='RESET_MANUAL_INIT' else evidence
                local,global_,transitions=runner(input_evidence,params)
                local.to_csv(out/f'{variant}_slot_timeseries.csv',index=False,encoding='utf-8-sig')
                transitions.to_csv(out/f'{variant}_transitions.csv',index=False,encoding='utf-8-sig')
                traces[variant]=global_
                if variant in ('RESET_UNKNOWN','RESET_MANUAL_INIT'):
                    global_.to_csv(out/f'{variant}_global_slot_timeseries.csv',index=False,encoding='utf-8-sig')
                if variant=='RESET_SAFE':
                    refined,_=apply_transition_refiner(local,RefinerConfig())
                    traces['RESET_CANDIDATE']=build_global_trace(refined)
            for variant,filename in [('CONTINUOUS_SAFE','baseline_global_slot_timeseries.csv'),('CONTINUOUS_CANDIDATE','candidate_v164_global_slot_timeseries.csv')]:
                traces[variant]=pd.read_csv(fit/filename)
            if experimental:
                traces['CONTINUOUS_DEV_SELECTED']=pd.read_csv(fit/'dev_selected_experimental_global_slot_timeseries.csv')
            counts={}
            for variant,global_ in traces.items():
                if not set(window_gt.time_sec).issubset(set(global_.time_sec)):
                    raise ValueError('GT checkpoints missing from inference timestamps; align restart times and evidence sampling')
                # Inclusive startup: cold-start metrics cannot hide the first 10 seconds.
                trace=evaluate_state_output(global_,window_gt,0,0)['timeseries']
                if variant in ('RESET_UNKNOWN','RESET_MANUAL_INIT'):
                    bounds=occupancy_bounds(global_)
                    trace=trace.merge(bounds,on='time_sec',validate='one_to_one')
                    trace['gt_within_bounds']=(trace.ground_truth_occupied_space_count>=trace.confirmed_occupied)&(trace.ground_truth_occupied_space_count<=trace.possible_occupied)
                    trace.to_csv(out/f'{variant}_occupancy_bounds.csv',index=False,encoding='utf-8-sig')
                trace.to_csv(out/f'{variant}_count_timeseries.csv',index=False,encoding='utf-8-sig')
                counts[variant]=trace
                for scope,part in [('INCLUSIVE_STARTUP',trace),('AFTER_RESTART_WARMUP',trace[trace.time_sec>=start+warm])]:
                    metrics=recovery_metrics(part,start,int(cfg.get('stable_samples',3)))
                    if variant in ('RESET_UNKNOWN','RESET_MANUAL_INIT'):
                        metrics.update(count_interpretation='confirmed_lower_bound',fully_resolved_N=int(part.unknown_slots.eq(0).sum()),gt_within_bounds_rate=float(part.gt_within_bounds.mean()))
                        if part.unknown_slots.gt(0).any():
                            metrics['confirmed_count_exact_rate']=metrics['exact_rate'];metrics['confirmed_count_mae']=metrics['mae']
                            for key in list(metrics):
                                if key in ('exact_rate','mae','max_abs_error','over_rate','under_rate','post_stable_mae','post_stable_exact_rate','time_to_first_exact_sec','time_to_stable_exact_sec','stable_confirmed_at_sec','time_to_first_relapse_sec','relapse_count','longest_exact_run_samples','tail_exact_samples','never_stabilized') or key.startswith('first_'):
                                    metrics[key]=None
                    rows.append({'restart_sec':start,'end_sec':end,'variant':variant,'scope':scope,**metrics})
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
                'recovery_candidate_config':DEFAULT_RECOVERY,'legacy_recovery_config':LEGACY_RECOVERY,
                'dev_selected_experimental':experimental_freeze if experimental else None,
                'transition_guard_available':guard_params is not None,'gt_used_for_bootstrap':False,
                'fresh_inference_start_sec':float(evidence.time_sec.min()),
                'segmentation_available':not (out/'evidence/SEGMENTATION_UNAVAILABLE.txt').exists()},indent=2),encoding='utf-8')
    assert freeze==(fit/'selected_state_params.json').read_bytes(),'Restart evaluation changed fitted selection'
    if experimental:
        assert hashlib.sha256((fit/'dev_selected_experimental_params.json').read_bytes()).hexdigest()==experimental_freeze['sha256']
    pd.DataFrame(rows).to_csv(root/'restart_metrics.csv',index=False,encoding='utf-8-sig')
    (root/'RESTART_REPORT.txt').write_text('Continuous versus fresh restart with frozen fitting.\n'
        'Startup-inclusive and matched post-restart warm-up metrics are separate. Stable exact means '
        'three consecutive available GT checkpoints, not permanent slot correctness. Never recovered is missing, not zero.\n'
        'Three-checkpoint agreement is provisional: relapse time/count and post-agreement MAE are reported.\n'
        'RESET_RECOVERY is additive v16.5.6; RESET_RECOVERY_V1655 retains the old blocking candidate.\n'
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
