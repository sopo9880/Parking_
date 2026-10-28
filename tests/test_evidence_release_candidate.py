import unittest
import pandas as pd
from evidence_release_candidate import run_candidate


class EvidenceReleaseTests(unittest.TestCase):
    def setUp(self):
        self.base=pd.DataFrame([{'time_sec':t,'global_id':'G1','state':s} for t,s in [(0,'OCCUPIED'),(1,'EMPTY'),(2,'EMPTY'),(3,'OCCUPIED')]])
        self.obs=pd.DataFrame([{'time_sec':t,'global_id':'G1','local_id':'A','cctv':'C1','full_detected':int(t==3),'full_det_conf':.9 if t==3 else 0} for t in range(4)])
    def test_missed_detection_is_unknown_not_empty(self):
        d=run_candidate(self.base,self.obs)
        self.assertEqual(d.state.tolist(),['OCCUPIED','UNKNOWN','UNKNOWN','OCCUPIED'])
        self.assertEqual(d.reason.iloc[1],'NO_CONFIRMED_EXIT')
    def test_positive_departure_can_clear(self):
        cue=pd.DataFrame([{'time_sec':2,'global_id':'G1','outward_motion':True,'empty_visible':True}])
        d=run_candidate(self.base,self.obs,cue)
        self.assertEqual(d.state.tolist(),['OCCUPIED','UNKNOWN','EMPTY','OCCUPIED'])
    def test_other_camera_blocks_clearing(self):
        extra=pd.DataFrame([{'time_sec':2,'global_id':'G1','local_id':'B','cctv':'C2','full_detected':1,'full_det_conf':.8}])
        cue=pd.DataFrame([{'time_sec':2,'global_id':'G1','outward_motion':True,'empty_visible':True}])
        d=run_candidate(self.base,pd.concat([self.obs,extra],ignore_index=True),cue)
        self.assertEqual(d.state.iloc[2],'UNKNOWN');self.assertTrue(d.occupied_detection_any_camera.iloc[2])
    def test_strong_aux_can_protect_existing_occupancy(self):
        obs=self.obs.copy();obs['aux_crop_detected']=0;obs['aux_crop_det_conf']=0.;obs['aux_support_scales']=0
        obs.loc[obs.time_sec==1,['aux_crop_detected','aux_crop_det_conf','aux_support_scales']]=[1,.3,2]
        d=run_candidate(self.base,obs)
        self.assertEqual(d.state.iloc[1],'OCCUPIED');self.assertEqual(d.state.iloc[2],'UNKNOWN')
    def test_one_cue_alone_is_not_exit_evidence(self):
        for motion,empty in ((True,False),(False,True)):
            cue=pd.DataFrame([{'time_sec':2,'global_id':'G1','outward_motion':motion,'empty_visible':empty}])
            self.assertEqual(run_candidate(self.base,self.obs,cue).state.iloc[2],'UNKNOWN')
    def test_future_cue_does_not_change_past(self):
        cue=pd.DataFrame([{'time_sec':2,'global_id':'G1','outward_motion':True,'empty_visible':True}])
        original=run_candidate(self.base,self.obs);changed=run_candidate(self.base,self.obs,cue)
        self.assertEqual(original.state.iloc[:2].tolist(),changed.state.iloc[:2].tolist())
    def test_unknown_not_converted_to_empty(self):
        base=self.base.copy();base.loc[0,'state']='UNKNOWN'
        self.assertEqual(run_candidate(base,self.obs).state.iloc[1],'UNKNOWN')
    def test_rejects_ambiguous_cue(self):
        cue=pd.DataFrame([{'time_sec':2,'global_id':'G1','outward_motion':'False','empty_visible':True}])
        with self.assertRaises(ValueError):run_candidate(self.base,self.obs,cue)


if __name__=='__main__': unittest.main()
