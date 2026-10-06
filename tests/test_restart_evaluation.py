import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from restart_evaluation import DEFAULT_RECOVERY,LEGACY_RECOVERY,recovery_gate,recovery_metrics,run_restart_experiments
from slot_engine import _run_state_engine_v13_safe,_v16_baseline_params

ROOT=Path(__file__).resolve().parents[1]

def evidence(start=0,confidence=.15,cameras=2):
    rows=[]
    for t in range(start,start+41):
        for camera in range(cameras):
            rows.append({'time_sec':float(t),'timestamp':str(t),'cctv':f'cctv{camera+1}',
                'local_id':f'S{camera+1}','global_id':'G1','initial_state':'UNKNOWN',
                'full_detected':1,'full_det_conf':confidence,'full_track_id':-1,
                'full_track_speed_px_s':0,'full_track_motion_span_px':0,
                'full_center_x':50,'full_center_y':50,'visual_diff_initial':0,
                'aux_requested':0,'aux_crop_detected':0,'aux_crop_det_conf':0,'aux_support_scales':0})
    return pd.DataFrame(rows)


class RestartTests(unittest.TestCase):
    def test_recovery_is_causal_stationary_and_bounded(self):
        settings=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
        base={**_v16_baseline_params(settings),'unlabeled_restart':True};rows=evidence(900)
        _,safe,_=_run_state_engine_v13_safe(rows,base)
        local,recovered,_=_run_state_engine_v13_safe(rows,{**base,'startup_recovery':DEFAULT_RECOVERY})
        self.assertTrue((safe.state=='EMPTY').all())
        self.assertTrue((recovered[recovered.time_sec<910].state=='EMPTY').all())
        self.assertEqual(recovered[recovered.time_sec==910].state.iloc[0],'OCCUPIED')
        self.assertTrue((local[local.time_sec>930].startup_phase=='NORMAL').all())
        _,prefix,_=_run_state_engine_v13_safe(rows[rows.time_sec<=920],{**base,'startup_recovery':DEFAULT_RECOVERY})
        self.assertEqual(prefix.state.tolist(),recovered[recovered.time_sec<=920].state.tolist())
        _,single,_=_run_state_engine_v13_safe(evidence(900,cameras=1),{**base,'startup_recovery':DEFAULT_RECOVERY})
        self.assertTrue((single.state=='EMPTY').all())
        moving=evidence(900);moving['full_center_x']=moving.time_sec*10
        self.assertFalse(recovery_gate(moving[moving.local_id=='S1'].iloc[:11].to_dict('records'),10,DEFAULT_RECOVERY))
        nohit=evidence(900);nohit['full_detected']=0
        self.assertFalse(recovery_gate(nohit.iloc[:11].to_dict('records'),10,DEFAULT_RECOVERY))

    def test_startup_metrics_do_not_exclude_errors(self):
        trace=pd.DataFrame({'time_sec':[900,910,920,930,940,950,960], 'error':[-3,-2,0,0,0,1,0]})
        m=recovery_metrics(trace,900)
        self.assertEqual(m['first_30s_N'],3)
        self.assertAlmostEqual(m['first_30s_mae'],5/3)
        self.assertEqual(m['time_to_first_exact_sec'],20)
        self.assertEqual(m['time_to_stable_exact_sec'],20)
        self.assertEqual(m['stable_confirmed_at_sec'],40)
        self.assertAlmostEqual(m['post_stable_exact_rate'],.8)
        self.assertEqual(m['time_to_first_relapse_sec'],50)
        self.assertEqual(m['relapse_count'],1)
        self.assertEqual(m['longest_exact_run_samples'],3)
        self.assertAlmostEqual(m['post_stable_mae'],.2)
        none=recovery_metrics(trace.assign(error=-1),900)
        self.assertTrue(none['never_stabilized'])
        self.assertIsNone(none['time_to_first_exact_sec'])
        self.assertIsNone(none['time_to_stable_exact_sec'])

    def test_recovery_does_not_block_normal_safe_entry(self):
        settings=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
        base={**_v16_baseline_params(settings),'unlabeled_restart':True}
        rows=evidence(900,confidence=.8,cameras=1)
        _,safe,_=_run_state_engine_v13_safe(rows,base)
        _,new,_=_run_state_engine_v13_safe(rows,{**base,'startup_recovery':DEFAULT_RECOVERY})
        _,old,_=_run_state_engine_v13_safe(rows,{**base,'startup_recovery':LEGACY_RECOVERY})
        self.assertEqual(safe.state.tolist(),new.state.tolist())
        self.assertEqual(new.iloc[0].state,'OCCUPIED')
        self.assertEqual(old.iloc[0].state,'EMPTY')

    def test_archived_restart_startup_regression(self):
        folder=ROOT/'paper/data/restart_v1655/original'
        rows=pd.read_csv(folder/'startup_evidence_900_910.csv')
        params=json.loads((folder/'selected_state_params.json').read_text(encoding='utf-8-sig'))
        params['unlabeled_restart']=True
        _,safe,_=_run_state_engine_v13_safe(rows,params)
        _,new,_=_run_state_engine_v13_safe(rows,{**params,'startup_recovery':DEFAULT_RECOVERY})
        _,old,_=_run_state_engine_v13_safe(rows,{**params,'startup_recovery':LEGACY_RECOVERY})
        for t in (900,910):
            self.assertEqual(new[new.time_sec==t].state.eq('OCCUPIED').sum(),17)
            self.assertEqual(safe[safe.time_sec==t].state.tolist(),new[new.time_sec==t].state.tolist())
        self.assertEqual(old[old.time_sec==900].state.eq('OCCUPIED').sum(),0)

    def test_live_extractor_reads_no_pre_restart_frames(self):
        from slot_engine import extract_evidence
        import cv2
        cfg=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
        cfg.update(inference_start_sec=900,eval_end_sec=902,evidence_sample_sec=1,unlabeled_restart=True)
        class Detector:
            def __init__(self,settings): pass
            def detect(self,image): return []
            def detect_batch(self,images): return [[] for _ in images]
        class Video:
            def release(self): pass
        times=[]
        def read(cap,t):
            times.append(t);return np.zeros((200,200,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'templates').mkdir()
            cv2.imwrite(str(root/'templates/initial.jpg'),np.zeros((100,100,3),dtype=np.uint8))
            slots=root/'slots.json';slots.write_text(json.dumps({'slots':[{'cctv':'cctv1','local_id':'S1',
                'global_id':'G1','point':[50,50],'region':[0,0,100,100],'initial_state':'UNKNOWN','template_initial':'templates/initial.jpg'}]}))
            with patch('slot_engine.VehicleDetector',Detector),patch('slot_engine.open_video',return_value=Video()), \
                 patch('slot_engine.read_frame_at',side_effect=read),patch('slot_engine.crop_roi',side_effect=lambda frame,roi:frame):
                rows=extract_evidence('video',{'cctv1':[]},str(slots),cfg,str(root),str(root/'output'))
            self.assertEqual(times,[900,901,902])
            self.assertEqual(rows.time_sec.tolist(),[900,901,902])
            self.assertTrue((rows.initial_state=='UNKNOWN').all())

    def test_restart_reextracts_unknown_state_and_freezes_fit(self):
        settings=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
        settings.update(eval_end_sec=940,dev_end_sec=900)
        settings['validation']['restart_experiments']={'starts_sec':[900],'duration_sec':40,'stable_samples':3}
        settings['detector_by_cctv']={'cctv1':{'model':'fixed'}}
        settings['active_split_protocol']={'test':[900,940]}
        before=copy.deepcopy(settings);seen=[]
        def extract(video,rois,slots,cfg,learned,out,progress):
            self.assertEqual(cfg['inference_start_sec'],900)
            self.assertEqual(cfg['frozen_segmentation_quality_thresholds'],{})
            self.assertTrue(all(s['initial_state']=='UNKNOWN' for s in json.loads(Path(slots).read_text())['slots']))
            seen.append(cfg['inference_start_sec']);return evidence(900,cameras=1)
        with tempfile.TemporaryDirectory() as tmp:
            fit=Path(tmp);params=b'{"frozen":true}';(fit/'selected_state_params.json').write_bytes(params)
            (fit/'selection_freeze.json').write_text(json.dumps({'sha256':hashlib.sha256(params).hexdigest(),'detectors':settings['detector_by_cctv']}))
            (fit/'slots.json').write_text(json.dumps({'slots':[{'local_id':'S1','global_id':'G1','cctv':'cctv1','initial_state':'OCCUPIED'}]}))
            continuous=pd.DataFrame({'time_sec':range(0,941),'timestamp':[str(t) for t in range(941)],'global_id':'G1','state':'OCCUPIED'})
            for tag in ('baseline','candidate_v164'):
                continuous.to_csv(fit/f'{tag}_global_slot_timeseries.csv',index=False)
            gt=pd.DataFrame({'time_sec':range(900,941,10),'timestamp':[str(t) for t in range(900,941,10)],'ground_truth_occupied_space_count':1})
            with patch('slot_engine.extract_evidence',side_effect=extract),patch('slot_engine.extract_segmentation_assist',side_effect=lambda *args:args[4]):
                result=run_restart_experiments('video',{'cctv1':[]},gt,settings,fit)
            metrics=pd.read_csv(result/'restart_metrics.csv')
            safe=metrics[(metrics.variant=='RESET_SAFE')&(metrics.scope=='INCLUSIVE_STARTUP')].iloc[0]
            self.assertEqual(safe.N,5)
            self.assertEqual(safe.mae,1)
            self.assertEqual(seen,[900])
            self.assertEqual((fit/'selected_state_params.json').read_bytes(),params)
            self.assertEqual(settings,before)


if __name__=='__main__':unittest.main()
