import numpy as np

from detector import Detection, physical_vehicle_nms
from slot_engine import assign_detections_to_slots, _learn_slot_geometry, run_state_engine, _slot_crop_geometry, _effective_observation, _run_state_engine_v15_transition_guard, _run_state_engine_v15_1_seg_assist, _run_state_engine_v16_empty_ref_assist, _v16_augment_empty_reference, _v16_transition_seg_requests, _repair_state_compare_columns, _append_state_variant_row
from segmentation import enhance_parking_crop, image_quality_metrics, recover_low_res_crop, recover_sun_glare_crop, recover_monitor_stripes_crop, conditioned_parking_crop, temporal_median_crop, build_relative_quality_thresholds, classify_quality_condition
from degradation_robustness import degrade_low_res, degrade_sun_glare, degrade_monitor_stripes, _pick_hard_slots, _pick_balanced_samples, _augment_samples_for_human_balance, _write_ground_truth_metrics, _canonical_gt_label, _gt_label_numeric_series
from app import _evidence_sanity, _merge_learned_slot_metadata, _learned_metadata_health


def main():
    # Cross-class duplicate removal still must produce one physical vehicle.
    dets = [
        Detection(0, 0, 100, 100, 0.90, 2),
        Detection(5, 5, 98, 98, 0.70, 7),
        Detection(120, 0, 220, 100, 0.85, 2),
    ]
    clean = physical_vehicle_nms(dets, 0.42, 0.68)
    assert len(clean) == 2, f'physical NMS failed: {len(clean)}'

    cfg = {
        'unmatched_cost': 1.15,
        'invalid_cost': 5.0,
        'voronoi_boundary_slack_px': 0.0,
        'manual_gate_neighbor_factor': 0.82,
        'manual_gate_min_px': 28.0,
        'manual_gate_max_px': 120.0,
        'manual_distance_weight': 0.82,
        'anchor_distance_weight': 0.18,
        'min_anchor_samples': 30,
        'max_anchor_std_px': 30.0,
        'anchor_max_shift_neighbor_factor': 0.30,
        'anchor_max_shift_min_px': 18.0,
        'anchor_max_shift_max_px': 40.0,
        'anchor_reject_if_exceeds_max_shift': True,
        'anchor_sample_gate_neighbor_factor': 0.90,
        'anchor_sample_gate_min_px': 35.0,
        'anchor_sample_gate_max_px': 120.0,
    }

    # Even a deliberately wrong learned anchor may not steal a detection from the
    # manual-point Voronoi owner.
    slots = [
        {
            'local_id':'S1','cctv':'cctv1','point':[50,50],
            'anchor_center':[165,50],'anchor_trusted':True,
            'expected_box_wh':[80,80], 'region':[35,35,65,65],
        },
        {
            'local_id':'S2','cctv':'cctv1','point':[170,50],
            'anchor_center':[170,50],'anchor_trusted':True,
            'expected_box_wh':[80,80], 'region':[155,35,185,65],
        },
    ]
    near_s1 = [Detection(15, 10, 95, 90, 0.92, 2)]
    assigned, unmatched, meta = assign_detections_to_slots(near_s1, slots, cfg)
    assert set(assigned) == {'S1'}, f'Voronoi identity protection failed: {assigned.keys()}'
    assert not unmatched, 'valid S1 detection was left unmatched'

    # One physical detection still may occupy at most one slot.
    boundary = [Detection(70, 0, 150, 100, 0.90, 2)]
    assigned, _, _ = assign_detections_to_slots(boundary, slots, {**cfg, 'voronoi_boundary_slack_px': 20.0})
    assert len(assigned) <= 1, 'one physical vehicle was assigned to multiple slots'

    # Low-sample anchor learning must fall back to the manual point.
    learn_slot = {'local_id':'S1','cctv':'cctv1','point':[50,50]}
    neighbor = {'local_id':'S2','cctv':'cctv1','point':[150,50]}
    centers = [(82.0, 50.0)] * 8
    boxes = [[42,10,122,90] for _ in centers]
    _learn_slot_geometry(learn_slot, centers, boxes, [learn_slot, neighbor], (200,240,3), cfg)
    assert learn_slot['anchor_trusted'] is False, 'low-sample anchor should not be trusted'
    assert learn_slot['anchor_center'] == [50.0, 50.0], 'manual point must be preserved on fallback'

    # With enough stable samples, learned anchor may move, but only within the configured cap.
    learn_slot2 = {'local_id':'S1','cctv':'cctv1','point':[50,50]}
    stable = [(75.0, 50.0)] * 40
    boxes2 = [[35,10,115,90] for _ in stable]
    _learn_slot_geometry(learn_slot2, stable, boxes2, [learn_slot2, neighbor], (200,240,3), cfg)
    assert learn_slot2['anchor_trusted'] is True, 'stable anchor should be trusted'
    assert learn_slot2['anchor_shift_px'] <= learn_slot2['anchor_max_shift_px'] + 1e-6, 'anchor drift cap failed'

    # v15: a stable-looking cluster that lies beyond the manual-point shift cap is rejected, not clamped.
    far_slot = {'local_id':'S1','cctv':'cctv1','point':[50,50]}
    far_centers = [(95.0,50.0)] * 40
    far_boxes = [[55,10,135,90] for _ in far_centers]
    _learn_slot_geometry(far_slot, far_centers, far_boxes, [far_slot,neighbor], (200,240,3), cfg)
    assert far_slot['anchor_trusted'] is False and far_slot['anchor_center']==[50.0,50.0], 'far learned anchor was not rejected'


    # v10 temporal test: one vehicle may maneuver across S1/S2, but the transient
    # source slot must not become a second stable occupancy. After it settles on S2,
    # S2 should become OCCUPIED; after a full rolling window of no detections and
    # empty appearance, it should return to EMPTY.
    import pandas as pd
    rows = []
    for t in range(31):
        for lid in ['S1','S2']:
            detected = 0; tid = -1; speed = 0.0; motion = 0.0; vd = 0.0
            if t <= 3 and lid == 'S1':
                detected=1; tid=1; speed=25.0; motion=50.0; vd=0.20
            elif 4 <= t <= 7 and lid == 'S2':
                detected=1; tid=1; speed=25.0; motion=50.0; vd=0.20
            elif 8 <= t <= 18 and lid == 'S2':
                detected=1; tid=1; speed=1.0; motion=2.0; vd=0.20
            elif 19 <= t <= 22 and lid == 'S2':
                vd=0.20
            rows.append({
                'time_sec':float(t),'timestamp':f'0:{t:02d}','cctv':'cctv1','local_id':lid,
                'global_id':'G1' if lid=='S1' else 'G2','initial_state':'EMPTY',
                'detected':detected,'det_conf':0.8 if detected else 0.0,'track_id':tid,
                'track_speed_px_s':speed,'track_motion_span_px':motion,'visual_diff_initial':vd,
            })
    ev = pd.DataFrame(rows)
    params = {
        'temporal_window_sec':10.0,'recent_window_sec':3.0,'entry_hit_ratio':0.35,
        'exit_hit_ratio':0.10,'stable_track_ratio':0.55,'occupied_visual_diff_threshold':0.12,
        'stationary_speed_px_s':14.0,'stationary_motion_span_px':32.0,
        'max_track_switches_for_entry':1,'global_merge':'ANY','global_confidence_threshold':0.0,
    }
    local, _, _ = run_state_engine(ev, params)
    s1 = local[local['local_id']=='S1'].set_index('time_sec')
    s2 = local[local['local_id']=='S2'].set_index('time_sec')
    assert s1.loc[10.0,'state'] == 'EMPTY', 'maneuver ghost occupancy was not suppressed'
    assert s2.loc[15.0,'state'] == 'OCCUPIED', 'slot-zone settle gate did not confirm the parked vehicle' 
    assert s2.loc[28.0,'state'] == 'EMPTY', 'rolling exit confirmation failed'


    # v11 slot-crop geometry must remain centered around the immutable manual point
    # and FULL/CROP/HYBRID evidence modes must be distinguishable without rerunning YOLO.
    crop_slot = {'local_id':'S1','cctv':'cctv1','point':[80,100]}
    crop_neighbor = {'local_id':'S2','cctv':'cctv1','point':[160,100]}
    geom = _slot_crop_geometry(crop_slot, [crop_slot,crop_neighbor], (240,320,3), {
        'width_neighbor_factor':2.2,'height_neighbor_factor':2.6,'vertical_center_offset_neighbor_factor':0.3,
        'core_radius_neighbor_factor':0.62,'core_radius_min_px':22,'core_radius_max_px':95,
        'neighbor_min_px':36,'neighbor_max_px':180,'min_width_px':96,'max_width_px':360,
        'min_height_px':112,'max_height_px':420,
    })
    assert geom['x1'] <= 80 <= geom['x2'] and geom['y1'] <= 100 <= geom['y2'], 'slot crop lost manual point'
    sample = pd.Series({'full_detected':1,'full_det_conf':0.12,'full_track_id':7,'full_center_x':80,'full_center_y':100,
                        'crop_detected':1,'crop_det_conf':0.82,'crop_center_x':82,'crop_center_y':101})
    assert _effective_observation(sample, {'evidence_mode':'FULL','full_min_conf':0.0})['source'] == 'FULL'
    assert _effective_observation(sample, {'evidence_mode':'CROP_AUX','crop_min_conf':0.10})['source'] == 'AUX'
    assert _effective_observation(sample, {'evidence_mode':'HYBRID_ADD','crop_add_min_conf':0.20,'crop_add_high_conf':0.45,'crop_add_min_scales':2})['source'] == 'BOTH'


    # v12 additive-hybrid invariant: AUX disagreement may never delete a FULL detection.
    full_only = pd.Series({'full_detected':1,'full_det_conf':0.11,'full_track_id':9,'full_center_x':80,'full_center_y':100,
                           'aux_crop_detected':0,'aux_crop_det_conf':0.0,'aux_support_scales':0})
    obs = _effective_observation(full_only, {'evidence_mode':'HYBRID_ADD','full_min_conf':0.0,
                                             'crop_add_min_conf':0.20,'crop_add_high_conf':0.45,'crop_add_min_scales':2})
    assert obs['detected'] == 1 and obs['source'] == 'FULL', 'AUX was allowed to suppress a valid FULL detection'

    aux_multi = pd.Series({'full_detected':0,'full_det_conf':0.0,
                           'aux_crop_detected':1,'aux_crop_det_conf':0.31,'aux_support_scales':2,
                           'crop_center_x':82,'crop_center_y':101})
    obs2 = _effective_observation(aux_multi, {'evidence_mode':'HYBRID_ADD','full_min_conf':0.0,
                                              'crop_add_min_conf':0.20,'crop_add_high_conf':0.45,'crop_add_min_scales':2})
    assert obs2['detected'] == 1 and obs2['source'] == 'AUX_ADD', 'multi-scale AUX recovery failed'

    # v13: repeated very-low-confidence detections without appearance support
    # must not create a ghost occupied slot.
    rows=[]
    for t in range(15):
        rows.append({
            'time_sec':float(t),'timestamp':f'0:{t:02d}','cctv':'cctv1','local_id':'S1','global_id':'G1','initial_state':'EMPTY',
            'full_detected':1 if t>=3 else 0,'full_det_conf':0.08 if t>=3 else 0.0,
            'full_track_id':11 if t>=3 else -1,'full_track_speed_px_s':0.3,'full_track_motion_span_px':1.0,
            'full_center_x':50.0,'full_center_y':50.0,'visual_diff_initial':0.01,
            'aux_crop_detected':0,'aux_crop_det_conf':0.0,'aux_support_scales':0,
        })
    ghost=pd.DataFrame(rows)
    p13={
        'evidence_mode':'FULL','temporal_window_sec':10.0,'recent_window_sec':3.0,
        'entry_hit_ratio':0.35,'exit_hit_ratio':0.1,'stable_track_ratio':0.55,
        'occupied_visual_diff_threshold':0.08,'stationary_speed_px_s':14.0,'stationary_motion_span_px':32.0,
        'crop_stationary_motion_span_px':28.0,'max_track_switches_for_entry':1,'entry_mean_conf_min':0.12,
        'entry_high_conf_override':0.30,'entry_track_settle_sec':3.0,'exit_recent_clear_sec':2.0,
        'recovery_window_sec':8.0,'global_merge':'ANY','global_confidence_threshold':0.0,'full_min_conf':0.0,
    }
    g_local,_,_=run_state_engine(ghost,p13)
    assert (g_local['state']=='OCCUPIED').sum()==0, 'v13 low-confidence ghost was promoted to OCCUPIED'

    # v15 transition-only guard: baseline wants occupancy as soon as a vehicle settles,
    # but large movement in the preceding 10s must delay the transition. Stable state itself is preserved.
    rows=[]
    for t in range(25):
        hit = 1 if 2 <= t <= 20 else 0
        cx = 50.0
        if 2 <= t <= 7: cx = 50.0 + (t-2)*12.0  # parking maneuver
        elif 8 <= t <= 20: cx = 110.0
        rows.append({'time_sec':float(t),'timestamp':f'0:{t:02d}','cctv':'cctv1','local_id':'S1','global_id':'G1','initial_state':'EMPTY',
                     'full_detected':hit,'full_det_conf':0.85 if hit else 0.0,'full_track_id':1 if hit else -1,
                     'full_track_speed_px_s':20.0 if 2<=t<=7 else 0.5,'full_track_motion_span_px':60.0 if 2<=t<=7 else 2.0,
                     'full_center_x':cx if hit else float('nan'),'full_center_y':50.0 if hit else float('nan'),
                     'visual_diff_initial':0.25 if hit else 0.0,'aux_requested':1 if t>=21 else 0,'aux_crop_detected':0,'aux_crop_det_conf':0.0,'aux_support_scales':0})
    tev=pd.DataFrame(rows)
    gp={**p13,'entry_mean_conf_min':0.08,'entry_no_appearance_conf':0.25,'transition_guard_window_sec':10.0,'transition_guard_recent_sec':3.0,
        'entry_guard_motion_span_px':45.0,'entry_guard_recent_motion_span_px':22.0,'entry_guard_settle_sec':7.0,'entry_guard_min_recent_hit_ratio':0.5,
        'entry_guard_min_mean_conf':0.08,'zone_motion_step_px':14.0,'exit_guard_no_full_sec':4.0,'exit_guard_aux_window_sec':5.0,
        'exit_guard_aux_min_checks':2,'exit_guard_aux_conf':0.25,'exit_guard_aux_min_scales':2}
    gl,_,_=_run_state_engine_v15_transition_guard(tev,gp)
    g1=gl.set_index('time_sec')
    assert g1.loc[10.0,'state']=='EMPTY', 'v15 did not hold maneuvering entry'
    assert (g1.loc[18.0:20.0,'state']=='OCCUPIED').any(), 'v15 never confirmed settled occupancy'


    # v15.2 SEG_ASSIST: positive segmentation may protect an established OCCUPIED
    # state from an unsafe clear, but segmentation alone may never create occupancy.
    seg_rows=[]
    for t in range(15):
        full = 1 if t <= 5 else 0
        seg_rows.append({'time_sec':float(t),'timestamp':f'0:{t:02d}','cctv':'cctv1','local_id':'S1','global_id':'G1','initial_state':'OCCUPIED',
                         'full_detected':full,'full_det_conf':0.9 if full else 0.0,'full_track_id':1 if full else -1,
                         'full_track_speed_px_s':0.0,'full_track_motion_span_px':0.0,'full_center_x':50.0 if full else float('nan'),'full_center_y':50.0 if full else float('nan'),
                         'visual_diff_initial':0.0,'aux_requested':1 if t>=6 else 0,'aux_crop_detected':0,'aux_crop_det_conf':0.0,'aux_support_scales':0,
                         'seg_requested':1 if t>=6 else 0,'seg_available':1,'seg_detected':1 if 9<=t<=14 else 0,'seg_conf':0.7 if 9<=t<=14 else 0.0,
                         'seg_core_overlap_ratio':0.6 if 9<=t<=14 else 0.0,'seg_point_covered':1 if 9<=t<=14 else 0})
    sev=pd.DataFrame(seg_rows)
    sp={**gp,'exit_guard_no_full_sec':2.0,'exit_guard_aux_window_sec':2.0,'exit_guard_aux_min_checks':1,
        'seg_protect_occupied_conf':0.10,'seg_protect_occupied_core_overlap_ratio':0.12}
    sl,_,_=_run_state_engine_v15_1_seg_assist(sev,sp)
    ss=sl.set_index('time_sec')
    assert ss.loc[12.0,'state']=='OCCUPIED', 'v15.2 positive segmentation failed to protect established occupancy'

    empty_rows=[]
    for t in range(12):
        empty_rows.append({'time_sec':float(t),'timestamp':f'0:{t:02d}','cctv':'cctv1','local_id':'S2','global_id':'G2','initial_state':'EMPTY',
                           'full_detected':0,'full_det_conf':0.0,'full_track_id':-1,'full_track_speed_px_s':0.0,'full_track_motion_span_px':0.0,
                           'full_center_x':float('nan'),'full_center_y':float('nan'),'visual_diff_initial':0.0,'aux_requested':0,'aux_crop_detected':0,'aux_crop_det_conf':0.0,'aux_support_scales':0,
                           'seg_requested':1,'seg_available':1,'seg_detected':1,'seg_conf':0.9,'seg_core_overlap_ratio':0.9,'seg_point_covered':1})
    el,_,_=_run_state_engine_v15_1_seg_assist(pd.DataFrame(empty_rows),sp)
    assert (el['state']=='OCCUPIED').sum()==0, 'v15.2 segmentation illegally created occupancy by itself'

    # v15.2 image-conditioning / stress transforms preserve shape and dtype and
    # produce finite quality metrics. They are preprocessing/stress tools, not GT.
    test_img=np.zeros((96,128,3),dtype=np.uint8)
    test_img[:, :64]=35; test_img[:, 64:]=245
    for fn in (enhance_parking_crop, degrade_low_res, degrade_sun_glare, degrade_monitor_stripes):
        out=fn(test_img.copy())
        assert out.shape==test_img.shape and out.dtype==np.uint8, f'v15.2 transform contract failed for {fn.__name__}'
    qm=image_quality_metrics(test_img)
    assert 0.0<=qm['highlight_ratio']<=1.0 and qm['sharpness_laplacian_var']>=0.0, 'v15.2 image quality metrics invalid'

    # v15.3 condition-specific preprocessing must preserve geometry/dtype and causal
    # temporal median must reduce alternating stripe noise without using future frames.
    for fn in (recover_low_res_crop, recover_sun_glare_crop):
        out=fn(test_img.copy())
        assert out.shape==test_img.shape and out.dtype==np.uint8, f'v15.3 transform contract failed for {fn.__name__}'
    stripe_frames=[]
    for phase in (0.0,0.9,1.8,2.7,3.6):
        stripe_frames.append(degrade_monitor_stripes(test_img.copy(),strength=0.30,period=5,phase=phase))
    med=temporal_median_crop(stripe_frames)
    assert med is not None and med.shape==test_img.shape and med.dtype==np.uint8, 'v15.3 temporal median contract failed'
    de=conditioned_parking_crop(stripe_frames[-1],'MONITOR_STRIPES',temporal_frames=stripe_frames)
    assert de.shape==test_img.shape and de.dtype==np.uint8, 'v15.3 temporal de-moire contract failed'
    # Median should not increase absolute error vs the clean synthetic image by a large margin.
    raw_err=float(np.abs(stripe_frames[-1].astype(np.int16)-test_img.astype(np.int16)).mean())
    med_err=float(np.abs(med.astype(np.int16)-test_img.astype(np.int16)).mean())
    assert med_err <= raw_err*1.20 + 1e-6, 'v15.3 temporal median did not suppress varying stripe stress'

    # v15.4 per-CCTV relative quality thresholds must classify distribution tails,
    # even when absolute v15.3 thresholds would classify every sample as CLEAN.
    qrows=[]
    for i in range(100):
        qrows.append({'cctv':'cctv1','highlight_ratio':0.01+0.0005*i,'stripe_energy_ratio':0.05+0.001*i,'sharpness_laplacian_var':220.0-i})
    thr=build_relative_quality_thresholds(qrows,{'highlight_percentile':0.90,'stripe_percentile':0.90,'sharpness_percentile':0.10,'condition_min_samples':20})
    c,_=classify_quality_condition({'highlight_ratio':0.059,'stripe_energy_ratio':0.05,'sharpness_laplacian_var':220},thr['cctv1'])
    assert c=='SUN_GLARE', f'relative glare tail not detected: {c}'

    # v15.4 forced robustness slots must never be truncated away by automatic hard-slot picks.
    ev=pd.DataFrame([{'cctv':'cctv1','local_id':'C1_S001','full_detected':0,'full_det_conf':0.0},
                     {'cctv':'cctv2','local_id':'C2_S012','full_detected':1,'full_det_conf':0.9},
                     {'cctv':'cctv3','local_id':'C3_S009','full_detected':1,'full_det_conf':0.9}])
    sl=[{'local_id':'C1_S001','cctv':'cctv1'},{'local_id':'C2_S012','cctv':'cctv2'},{'local_id':'C3_S009','cctv':'cctv3'}]
    chosen=_pick_hard_slots(ev,sl,{'hard_slots_per_cctv':2,'max_slots':2,'force_slots':['C2_S012','C3_S009']})
    assert 'C2_S012' in chosen and 'C3_S009' in chosen, f'forced robustness slots lost: {chosen}'


    # v16 balanced robustness sampling should intentionally request both likely classes.
    rows=[]
    for t in range(20):
        for lid,cctv in [('C1_S001','cctv1'),('C1_S002','cctv1'),('C2_S012','cctv2'),('C3_S009','cctv3')]:
            occ=lid in ('C2_S012',) or (lid=='C1_S001' and t%2==0)
            rows.append({'time_sec':float(t),'timestamp':f'0:{t:02d}','cctv':cctv,'local_id':lid,'initial_state':'OCCUPIED' if lid=='C2_S012' else 'EMPTY',
                         'full_detected':int(occ),'full_det_conf':0.85 if occ else 0.0,'crop_detected':int(occ),'crop_det_conf':0.6 if occ else 0.0,
                         'recent_full_hit_ratio':1.0 if occ else 0.0,'visual_diff_initial':0.2 if occ else 0.01})
    bev=pd.DataFrame(rows); bslots=[{'local_id':'C1_S001','cctv':'cctv1'},{'local_id':'C1_S002','cctv':'cctv1'},{'local_id':'C2_S012','cctv':'cctv2'},{'local_id':'C3_S009','cctv':'cctv3'}]
    samp=_pick_balanced_samples(bev,bslots,{'balanced_gt_samples':24,'forced_samples_per_slot':2,'force_slots':['C2_S012','C3_S009']})
    assert len(samp)==24, f'v16 balanced sampler wrong size: {len(samp)}'
    assert (samp['sampling_stratum'].astype(str).str.contains('OCCUPIED')).any(), 'v16 sampler lost likely occupied stratum'
    assert (samp['sampling_stratum'].astype(str).str.contains('EMPTY')).any(), 'v16 sampler lost likely empty stratum'

    # v16 EMPTY-reference calibration: an initially-empty slot that departs from its warm-up appearance
    # should become positive only after the causal warm-up threshold.
    erows=[]
    for t in range(12):
        erows.append({'time_sec':float(t),'timestamp':f'0:{t:02d}','cctv':'cctv1','local_id':'S3','global_id':'G3','initial_state':'EMPTY',
                      'visual_diff_initial':0.01 if t<=3 else 0.12,'full_detected':0,'full_det_conf':0.0,'crop_detected':1 if t>=4 else 0,'crop_det_conf':0.7 if t>=4 else 0.0,
                      'full_track_id':-1,'full_track_speed_px_s':0.0,'full_track_motion_span_px':0.0,'full_center_x':float('nan'),'full_center_y':float('nan'),
                      'aux_requested':1 if t>=4 else 0,'aux_crop_detected':1 if t>=4 else 0,'aux_crop_det_conf':0.7 if t>=4 else 0.0,'aux_support_scales':2,
                      'seg_requested':0,'seg_available':0,'seg_detected':0,'seg_conf':0.0,'seg_core_overlap_ratio':0.0,'seg_point_covered':0,'adaptive_det_detected':0,'adaptive_det_conf':0.0})
    edf=pd.DataFrame(erows)
    aug=_v16_augment_empty_reference(edf,{'empty_reference':{'enabled':True,'calibration_sec':3.0,'min_margin':0.02,'mad_multiplier':3.0}})
    assert int(aug.loc[aug['time_sec']==1.0,'empty_ref_positive'].iloc[0])==0, 'v16 empty reference fired during calibration appearance'
    assert int(aug.loc[aug['time_sec']==8.0,'empty_ref_positive'].iloc[0])==1, 'v16 empty reference failed to detect changed appearance'

    # Transition-time SEG scheduling should be much narrower than per-frame eager SEG and should
    # include sparse empty-reference/crop disagreement checks.
    req=_v16_transition_seg_requests(aug,{'evidence_sample_sec':1.0,'segmentation_assist':{'empty_ref_disagreement_interval_sec':2.0},'state_search':{},'temporal':{}})
    assert any(v=='EMPTY_REF_CROP_DISAGREEMENT' for v in req.values()), 'v16 transition-time SEG missed empty-reference disagreement'

    # EMPTY_REF_ASSIST may create occupancy only from appearance change + an independent vehicle cue,
    # never from appearance alone.
    ep={**sp,'sample_sec':1.0,'empty_reference_cfg':{'enabled':True,'calibration_sec':3.0,'min_margin':0.02,'mad_multiplier':3.0,
        'entry_streak_sec':2.0,'protect_occupied':True,'require_vehicle_sensor':True,'vehicle_support_min_conf':0.05}}
    erl,_,_=_run_state_engine_v16_empty_ref_assist(aug,ep)
    assert (erl[erl['time_sec']>=7.0]['state']=='OCCUPIED').any(), 'v16 empty-reference + vehicle fusion never created experimental occupancy'

    # Low-resolution conditioned preprocessing should accept causal temporal frames while preserving geometry.
    low_hist=[degrade_low_res(test_img.copy(),scale=0.35) for _ in range(5)]
    low_multi=conditioned_parking_crop(low_hist[-1],'LOW_RES',temporal_frames=low_hist)
    assert low_multi.shape==test_img.shape and low_multi.dtype==np.uint8, 'v16 low-res multi-frame recovery contract failed'


    # v16.1 UNKNOWN persistence: U must stay a string and be excluded from binary metrics.
    labels=pd.Series(['O','E','U',1,0,''])
    nums=_gt_label_numeric_series(labels)
    assert nums.iloc[0]==1 and nums.iloc[1]==0 and np.isnan(nums.iloc[2]), 'v16.1 O/E/U numeric mapping failed'
    assert _canonical_gt_label('')=='U' and _canonical_gt_label(1)=='O' and _canonical_gt_label(0)=='E', 'v16.1 legacy GT compatibility failed'

    # v16.1 state-search CSV repair: preserve both meanings of old duplicate evidence_mode headers.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); fp=td/'detection_mode_comparison.csv'; logp=td/'STATE_SEARCH_REPAIR.log'
        # Simulate pandas reading the old malformed duplicate header as evidence_mode + evidence_mode.1.
        legacy=pd.DataFrame([{'evidence_mode':'SAFE_BASELINE','evidence_mode.1':'FULL','dev_exact_rate':0.5}])
        legacy.to_csv(fp,index=False,encoding='utf-8-sig')
        appended=_append_state_variant_row(fp,{'temporal_variant':'EMPTY_REF_ASSIST','evidence_mode':'FULL','dev_exact_rate':0.6},repair_log=logp)
        assert not appended.columns.duplicated().any(), 'v16.1 state comparison still has duplicate columns'
        assert 'evidence_mode' in appended.columns and 'source_evidence_mode' in appended.columns, 'v16.1 evidence-mode semantics were not separated'
        assert set(appended['evidence_mode'].astype(str))=={'SAFE_BASELINE','EMPTY_REF_ASSIST'}, 'v16.1 variant append/recovery failed'
        assert logp.is_file(), 'v16.1 repair audit log was not written'

    # v16.2 learned-cache metadata restore: manual identity must survive while learned
    # region/template metadata is restored from a cache snapshot.
    current=[{'local_id':'S1','cctv':'cctv1','point':[10,20],'global_id':'G1','initial_state':'EMPTY','source':'MANUAL'}]
    learned=[{'local_id':'S1','cctv':'cctv1','point':[999,999],'global_id':'BAD','initial_state':'OCCUPIED',
              'region':[1,2,30,40],'template_initial':'templates/S1_initial.jpg','geometry_version':'v12_manual_voronoi','anchor_center':[12,21]}]
    merged=_merge_learned_slot_metadata(current,learned)
    assert merged[0]['point']==[10,20] and merged[0]['global_id']=='G1' and merged[0]['initial_state']=='EMPTY', 'v16.2 cache restore overwrote manual slot identity'
    assert merged[0]['region']==[1,2,30,40] and merged[0]['template_initial'].endswith('S1_initial.jpg'), 'v16.2 learned metadata was not restored'

    # v16.2 evidence sanity must reject the exact v16 poison pattern (visual diff == 1.0
    # everywhere) while accepting a normal distribution with valid templates.
    import cv2
    with tempfile.TemporaryDirectory() as td2:
        td2=Path(td2); (td2/'templates').mkdir()
        cv2.imwrite(str(td2/'templates'/'S1_initial.jpg'),np.full((20,20,3),120,np.uint8))
        slots_ok=[{'local_id':'S1','cctv':'cctv1','point':[10,20],'global_id':'G1','initial_state':'EMPTY','source':'MANUAL',
                   'region':[1,2,30,40],'template_initial':'templates/S1_initial.jpg','geometry_version':'v12_manual_voronoi'}]
        good=pd.DataFrame({'visual_diff_initial':[0.01+0.001*(i%17) for i in range(200)]})
        bad=pd.DataFrame({'visual_diff_initial':[1.0]*200})
        ok,reason,_=_evidence_sanity(good,slots_ok,td2)
        assert ok, f'v16.2 sanity rejected normal evidence: {reason}'
        ok,reason,_=_evidence_sanity(bad,slots_ok,td2)
        assert not ok and '1.0' in reason, 'v16.2 sanity failed to reject all-1.0 appearance evidence'

    # v16.2 human-balance repair keeps persisted human GT and adds only new minority
    # candidates instead of asking the user to relabel the original set.
    hrows=[]
    for t in range(30):
        occ=t<10
        hrows.append({'time_sec':float(t),'timestamp':f'0:{t:02d}','cctv':'cctv1','local_id':'S1','initial_state':'EMPTY',
                      'full_detected':int(occ),'full_det_conf':0.9 if occ else 0.0,'crop_detected':int(occ),'crop_det_conf':0.7 if occ else 0.0,
                      'recent_full_hit_ratio':1.0 if occ else 0.0,'visual_diff_initial':0.2 if occ else 0.01})
    hev=pd.DataFrame(hrows)
    # Persist 8 occupied + 2 empty labels so EMPTY is the human minority.
    labs=[]
    for t in list(range(8))+[10,11]:
        labs.append({'sample_id':f'cctv1_S1_{t:04d}s','occupied_gt':'O' if t<8 else 'E','notes':''})
    old_gt=pd.DataFrame(labs)
    base=_pick_balanced_samples(hev,[{'local_id':'S1','cctv':'cctv1'}],{'balanced_gt_samples':24,'force_slots':[]})
    aug=_augment_samples_for_human_balance(hev,base,{'human_balance_target_minority_fraction':0.40,'human_balance_max_new_per_run':12,'gt_empty_recent_hit_max':0.10},old_gt)
    assert (aug['sampling_stratum'].astype(str)=='PERSISTED_HUMAN_GT').sum()>=10, 'v16.2 did not preserve existing human GT samples'
    assert (aug['sampling_stratum'].astype(str)=='HUMAN_BALANCE_REPAIR_EMPTY').any(), 'v16.2 did not add missing EMPTY-class candidates'

    print('Parking Slot Engine v16.2 evidence cache repair self-test: PASS')


if __name__ == '__main__':
    main()
