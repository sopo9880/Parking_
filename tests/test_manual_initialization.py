import copy,csv,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from manual_initialization import load_snapshot,write_snapshot,apply_snapshot,occupancy_bounds
from slot_engine import _run_state_engine_v13_safe,_v16_baseline_params
from restart_evaluation import run_restart_experiments
from test_restart_evaluation import evidence
ROOT=Path(__file__).resolve().parents[1]

class ManualInitializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.params={**_v16_baseline_params(json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))),'unlabeled_restart':True,'preserve_unknown':True,'manual_initialization':True}
    def tearDown(self): self.tmp.cleanup()
    def test_snapshot_only_exact_time_no_count_gt_inference(self):
        p=self.root/'states.csv';p.write_text('time_sec,global_slot_id,initial_state\n900,G1,O\n910,G1,E\n',encoding='utf-8')
        self.assertEqual(load_snapshot(p,900,['G1','G2']),{'G1':'OCCUPIED','G2':'UNKNOWN'})
        p.write_text(p.read_text().replace('910,G1,E','910,G1,INVALID_FUTURE_STATE'))
        self.assertEqual(load_snapshot(p,900,['G1'])['G1'],'OCCUPIED')
        self.assertIsNone(load_snapshot(p,899,['G1']))
        p.write_text('time_sec,ground_truth_occupied_space_count\n900,18\n')
        with self.assertRaisesRegex(ValueError,'Count-only'): load_snapshot(p,900,['G1'])
    def test_duplicate_unknown_id_and_invalid_state_rejected(self):
        p=self.root/'s.csv'
        for body in ('900,G1,O\n900,G1,E\n','900,G9,O\n','900,G1,MANEUVERING\n'):
            p.write_text('time_sec,global_slot_id,initial_state\n'+body)
            with self.assertRaises(ValueError): load_snapshot(p,900,['G1'])
    def test_manual_occupied_needs_no_motion_then_runs_automatically(self):
        rows=evidence(900,confidence=0,cameras=1);rows.full_detected=0
        local,global_,_= _run_state_engine_v13_safe(apply_snapshot(rows,{'G1':'OCCUPIED'}),self.params)
        self.assertTrue(global_[global_.time_sec<910].state.eq('OCCUPIED').all())
        self.assertEqual(global_[global_.time_sec==910].state.iloc[0],'EMPTY')
        self.assertFalse(local.appearance_occupied.any())
        _,prefix,_=_run_state_engine_v13_safe(apply_snapshot(rows[rows.time_sec<=905],{'G1':'OCCUPIED'}),self.params)
        self.assertEqual(prefix.state.tolist(),global_[global_.time_sec<=905].state.tolist())
    def test_unknown_is_not_empty_and_shared_views_receive_same_prior(self):
        rows=evidence(900,confidence=0,cameras=2);rows.full_detected=0
        local,global_,_=_run_state_engine_v13_safe(apply_snapshot(rows,{'G1':'UNKNOWN'}),self.params)
        self.assertTrue(local.state.eq('UNKNOWN').all());self.assertTrue(global_.state.eq('UNKNOWN').all())
        bounds=occupancy_bounds(global_);self.assertTrue(bounds.confirmed_occupied.eq(0).all());self.assertTrue(bounds.possible_occupied.eq(1).all())
        local,global_,_=_run_state_engine_v13_safe(apply_snapshot(rows,{'G1':'OCCUPIED'}),self.params)
        self.assertTrue(local[local.time_sec==900].state.eq('OCCUPIED').all())
    def test_empty_snapshot_is_honored_at_start_and_positive_entry_resolves_unknown(self):
        rows=evidence(900,confidence=.8,cameras=1)
        _,empty,_=_run_state_engine_v13_safe(apply_snapshot(rows,{'G1':'EMPTY'}),self.params)
        self.assertEqual(empty.iloc[0].state,'EMPTY');self.assertEqual(empty.iloc[1].state,'OCCUPIED')
        _,unknown,_=_run_state_engine_v13_safe(apply_snapshot(rows,{'G1':'UNKNOWN'}),self.params)
        self.assertEqual(unknown.iloc[0].state,'UNKNOWN');self.assertEqual(unknown.iloc[1].state,'OCCUPIED')
    def test_zero_restart_is_preserved(self):
        from validation_center import parse_time_list
        self.assertEqual(parse_time_list('0:00,5:30,11:00,15:00',include_zero=True),[0,330,660,900])
    def test_restart_integration_future_labels_do_not_change_manual_trace(self):
        cfg=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'));cfg.update(eval_end_sec=940,dev_end_sec=900)
        p=self.root/'snap.csv';write_snapshot(p,900,{'G1':'UNKNOWN'})
        cfg['validation']['restart_experiments']={'starts_sec':[900],'duration_sec':40,'manual_init':{'enabled':True,'path':str(p)}}
        cfg['active_split_protocol']={'test':[900,940]};cfg['detector_by_cctv']={'cctv1':{'model':'fixed'}}
        original=copy.deepcopy(cfg);freeze=b'{"fixed":true}'
        (self.root/'selected_state_params.json').write_bytes(freeze)
        (self.root/'selection_freeze.json').write_text(json.dumps({'sha256':hashlib.sha256(freeze).hexdigest(),'detectors':cfg['detector_by_cctv']}))
        (self.root/'slots.json').write_text(json.dumps({'slots':[{'local_id':'S1','global_id':'G1','cctv':'cctv1','initial_state':'OCCUPIED'}]}))
        continuous=pd.DataFrame({'time_sec':range(900,941),'timestamp':[str(t) for t in range(900,941)],'global_id':'G1','state':'OCCUPIED'})
        for tag in ('baseline','candidate_v164'): continuous.to_csv(self.root/f'{tag}_global_slot_timeseries.csv',index=False)
        gt=pd.DataFrame({'time_sec':range(900,941,10),'timestamp':[str(t) for t in range(900,941,10)],'ground_truth_occupied_space_count':1})
        rows=evidence(900,confidence=0,cameras=1);rows.full_detected=0
        with patch('slot_engine.extract_evidence',return_value=rows),patch('slot_engine.extract_segmentation_assist',side_effect=lambda *args:args[4]):
            out=run_restart_experiments('video',{},gt,cfg,self.root)
            changed=run_restart_experiments('video',{},gt.assign(ground_truth_occupied_space_count=99),cfg,self.root)
        self.assertEqual((out/'restart_900/RESET_MANUAL_INIT_global_slot_timeseries.csv').read_bytes(),(changed/'restart_900/RESET_MANUAL_INIT_global_slot_timeseries.csv').read_bytes())
        metrics=pd.read_csv(out/'restart_metrics.csv');manual=metrics[metrics.variant=='RESET_MANUAL_INIT']
        self.assertTrue(manual.exact_rate.isna().all());self.assertTrue(manual.mae.isna().all());self.assertTrue(manual.fully_resolved_N.eq(0).all())
        self.assertEqual(original,cfg);self.assertEqual(freeze,(self.root/'selected_state_params.json').read_bytes())
        audit=json.loads((out/'restart_900/manual_init_audit.json').read_text());self.assertFalse(audit['future_gt_used_for_inference'])

if __name__=='__main__': unittest.main()
