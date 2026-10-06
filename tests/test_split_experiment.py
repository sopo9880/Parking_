import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from split_protocol import (validate_protocol, protocol_context, labels, learning_times,
                            calibration_quality_rows)
from split_experiment import execute, protocols, run_suite

ROOT = Path(__file__).resolve().parents[1]


class SplitExperiments(unittest.TestCase):
    def test_boundaries_gaps_and_legacy(self):
        protocol = validate_protocol({'dev':[8,24], 'test':[0,8]},24)
        with protocol_context(protocol):
            self.assertEqual(labels([0,7.99,8,24,25]).tolist(),['TEST','TEST','DEV','DEV','IGNORED'])
            self.assertEqual(learning_times({},2),list(range(8,25,2)))
        self.assertEqual(labels([899,900]).tolist(),['DEV','TEST'])
        with protocol_context(validate_protocol({'dev':[10,20],'test':[0,8]},24)):
            self.assertEqual(labels([8,9,10,20]).tolist(),['IGNORED','IGNORED','DEV','IGNORED'])
            self.assertEqual(learning_times({},2),[10,12,14,16,18])
        for p in ({'dev':[0,10],'test':[9,24]}, {'dev':[0,25],'test':[0,8]},
                  {'dev':[8,float('nan')],'test':[0,8]}):
            with self.assertRaises(ValueError): validate_protocol(p,24)

    def test_quality_calibration_ignores_test_images(self):
        evidence=pd.DataFrame({'time_sec':[0,7,8,24]})
        rows=[{'idx':n,'cctv':'cctv1','sharpness':n} for n in range(4)]
        with protocol_context(validate_protocol({'dev':[8,24],'test':[0,8]},24)):
            self.assertEqual([r['idx'] for r in calibration_quality_rows(rows,evidence,{})],[2,3])
        self.assertEqual(calibration_quality_rows(rows,evidence,{}),rows)

    def fixture(self):
        rows=[]
        for t in range(25):
            hit=int(2<=t<=20)
            rows.append({'time_sec':float(t),'timestamp':f'0:{t:02d}',
                'cctv':'cctv1','local_id':'S1','global_id':'G1','initial_state':'EMPTY',
                'full_detected':hit,'full_det_conf':0.85 if hit else 0,
                'full_track_id':1 if hit else -1,'full_track_speed_px_s':0.5,
                'full_track_motion_span_px':2,'full_center_x':50 if hit else float('nan'),
                'full_center_y':50 if hit else float('nan'),'visual_diff_initial':0.25 if hit else 0,
                'aux_requested':int(t>=21),'aux_crop_detected':0,'aux_crop_det_conf':0,
                'aux_support_scales':0})
        return pd.DataFrame(rows)

    def test_actual_tuner_and_geometry_read_only_dev_frames(self):
        from detector_tuner import run_warped_detector_sweep
        from slot_engine import learn_slots_and_candidates, load_slots
        settings=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
        settings.update(dev_end_sec=15,eval_end_sec=24,learn_sample_sec=2)
        settings['active_split_protocol']=validate_protocol({'dev':[8,24],'test':[0,8]},24)
        settings['detector_by_cctv']={}
        settings['detector_sweep']={'models':['fake'],'imgsz':[64],'conf':[.1],'tiling':[False]}
        class Detector:
            def __init__(self,cfg): pass
            def detect(self,image): return []
            def detect_raw(self,image): return []
        class Video:
            def release(self): pass
        times=[]
        def read(cap,t):
            times.append(t)
            return np.full((200,200,3),int(t),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp, protocol_context(settings['active_split_protocol']):
            root=Path(tmp);gt=root/'gt.csv';slots=root/'slots.json'
            pd.DataFrame({'timestamp':list(range(25)), 'cctv1_count':[0]*25,
                'ground_truth_occupied_space_count':[0]*25}).to_csv(gt,index=False)
            slots.write_text(json.dumps({'slots':[{'cctv':'cctv1','local_id':'S1',
                'global_id':'G1','point':[100,100],'initial_state':'EMPTY'}]}))
            for module in ('detector_tuner','slot_engine'):
                with patch(module+'.VehicleDetector',Detector), patch(module+'.open_video',return_value=Video()), \
                     patch(module+'.read_frame_at',side_effect=read), patch(module+'.crop_roi',side_effect=lambda frame,roi:frame):
                    if module=='detector_tuner':
                        run_warped_detector_sweep('v',{'cctv1':[]},str(gt),settings,str(root/'tuning'))
                        self.assertEqual(times,list(range(8,25)))
                        times.clear()
                    else:
                        learn_slots_and_candidates('v',{'cctv1':[]},str(slots),settings,str(root/'learned'))
                        self.assertEqual(times,list(range(8,25,2)))
                        import cv2
                        template=cv2.imread(str(root/'learned'/load_slots(slots)[0]['template_initial']))
                        self.assertAlmostEqual(float(template.mean()),8,delta=1)

    def test_real_state_selection_is_invariant_to_test_labels(self):
        settings=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
        settings.update(dev_end_sec=15,eval_end_sec=24,evaluation_warmup_sec=0)
        settings['validation']['split_experiments']={'dev':[8,24],'test':[0,8]}
        settings['validation']['restart_experiments']={'enabled':False}
        original=copy.deepcopy(settings)
        seen=[]
        def tune(video,rois,gt_path,cfg,output,progress):
            gt=pd.read_csv(gt_path)
            self.assertTrue((gt.time_sec>=8).all())
            self.assertEqual(cfg['detector_by_cctv'],{})
            cfg['detector_by_cctv']={'cctv1':{'model':'DEV-only'}}
            seen.append(gt.ground_truth_occupied_space_count.tolist())
        def learn(video,rois,slots,cfg,output,progress):
            self.assertEqual(learning_times(cfg,2),list(range(8,25,2)))
            Path(slots).write_text('{"slots":[]}',encoding='utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);slots=root/'input_slots.json'
            slots.write_text('{"slots":[{"point":[50,50]}]}',encoding='utf-8')
            hashes=[];scores=[];experimental_hashes=[]
            for index in range(2):
                gt=pd.DataFrame({'timestamp':[f'0:{t:02d}' for t in range(25)],
                    'ground_truth_occupied_space_count':[int(2<=t<=20) if t>=8 or index==0 else 5 for t in range(25)]})
                path=root/f'gt{index}.csv';gt.to_csv(path,index=False)
                out=root/f'run{index}'
                request={'settings':settings,'protocol':protocols(settings)[1],
                         'video':'fixture','gt':str(path),'slots':str(slots),
                         'rois':{'cctv1':[]},'output':str(out)}
                with patch('detector_tuner.run_warped_detector_sweep',side_effect=tune), \
                     patch('slot_engine.learn_slots_and_candidates',side_effect=learn), \
                     patch('slot_engine.extract_evidence',return_value=self.fixture()), \
                     patch('slot_engine.extract_segmentation_assist',side_effect=lambda *args:args[4]):
                    execute(request)
                hashes.append(json.loads((out/'selection_freeze.json').read_text())['sha256'])
                experimental_hashes.append(json.loads((out/'dev_selection_freeze.json').read_text())['sha256'])
                final=pd.read_csv(out/'split_final_metrics.csv')
                safe=final[final.variant=='SAFE_BASELINE'].set_index('split')
                self.assertEqual(safe.loc['DEV','N'],17)
                self.assertEqual(safe.loc['TEST','N'],8)
                self.assertIn('DEV_SELECTED_EXPERIMENTAL',final.variant.values)
                self.assertIn('PRODUCTION_SAFE_SELECTED',final.variant.values)
                scores.append(safe.loc['TEST','mae'])
                board=pd.read_csv(out/'DEV_selection_variant_comparison.csv')
                self.assertTrue((board.test_N==0).all())
            self.assertEqual(hashes[0],hashes[1])
            self.assertEqual(experimental_hashes[0],experimental_hashes[1])
            self.assertEqual(seen[0],seen[1])
            self.assertNotEqual(scores[0],scores[1])
            self.assertEqual(settings,original)
            self.assertEqual(json.loads(slots.read_text())['slots'][0]['point'],[50,50])

    def test_failed_worker_is_not_reported_as_complete(self):
        class FailedProcess:
            stdout=iter(['fixture failure\n'])
            def wait(self): return 1
        with tempfile.TemporaryDirectory() as tmp, patch('split_experiment.subprocess.Popen',return_value=FailedProcess()):
            with self.assertRaises(RuntimeError):
                run_suite('v',{},'g','s',{'dev_end_sec':15,'eval_end_sec':24,
                    'validation':{'split_experiments':{'dev':[8,24],'test':[0,8]}}},tmp)
            summary=json.loads((Path(tmp)/'split_experiments/suite_summary.json').read_text())
            self.assertEqual(summary['status'],'failed')
            self.assertEqual(summary['failures'],['original','alternate'])


if __name__=='__main__': unittest.main()
