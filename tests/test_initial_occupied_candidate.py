import unittest
import numpy as np
import pandas as pd
from initial_occupied_candidate import InitialOccupiedCue,run_initial_candidate


class InitialOccupiedTests(unittest.TestCase):
    def test_retains_stable_initial_appearance_and_detects_changed_motion(self):
        old=np.indices((40,40)).sum(axis=0).astype('uint8');old=np.repeat(old[:,:,None],3,axis=2)
        new=old.copy();new[10:30,10:30]=220
        d=InitialOccupiedCue('OCCUPIED',(0,0),0)
        first=d.observe(0,old,True,.9,1,(0,0),False,0,0);self.assertTrue(first['matches_initial_occupied'])
        d.observe(1,old,True,.9,1,(10,0),False,0,0)
        d.observe(2,new,False,0,-1,None,False,0,0);d.observe(3,new,False,0,-1,None,False,0,0)
        last=d.observe(4,new,False,0,-1,None,False,0,0)
        self.assertTrue(last['outward_motion_recent']);self.assertTrue(last['stable_changed_view'])
    def test_conflict_does_not_force_empty(self):
        base=pd.DataFrame([{'time_sec':0,'global_id':'G','state':'OCCUPIED'},{'time_sec':1,'global_id':'G','state':'EMPTY'}])
        f=pd.DataFrame([{'time_sec':t,'global_id':'G','local_id':'A','matches_initial_occupied':same,'full_observed':False,'strong_aux_observed':False,'outward_motion_recent':motion,'stable_changed_view':change} for t,same,motion,change in [(0,True,False,False),(1,True,True,True)]])
        self.assertEqual(run_initial_candidate(base,f).state.tolist(),['OCCUPIED','UNKNOWN'])
    def test_motion_and_changed_view_without_conflict_can_emit_empty(self):
        base=pd.DataFrame([{'time_sec':0,'global_id':'G','state':'OCCUPIED'},{'time_sec':1,'global_id':'G','state':'EMPTY'},{'time_sec':2,'global_id':'G','state':'OCCUPIED'}])
        f=pd.DataFrame([{'time_sec':t,'global_id':'G','local_id':'A','matches_initial_occupied':same,'full_observed':False,'strong_aux_observed':False,'outward_motion_recent':motion,'stable_changed_view':change} for t,same,motion,change in [(0,True,False,False),(1,False,True,True),(2,False,False,False)]])
        self.assertEqual(run_initial_candidate(base,f).state.tolist(),['OCCUPIED','EMPTY','OCCUPIED'])
    def test_future_features_do_not_change_prior_state(self):
        base=pd.DataFrame([{'time_sec':0,'global_id':'G','state':'OCCUPIED'},{'time_sec':1,'global_id':'G','state':'EMPTY'}])
        f=pd.DataFrame([{'time_sec':t,'global_id':'G','local_id':'A','matches_initial_occupied':False,'full_observed':False,'strong_aux_observed':False,'outward_motion_recent':False,'stable_changed_view':False} for t in (0,1)])
        prior=run_initial_candidate(base,f).state.tolist();f.loc[1,'stable_changed_view']=True;f.loc[1,'outward_motion_recent']=True
        self.assertEqual(prior[0],run_initial_candidate(base,f).state.iloc[0])
    def test_missing_linked_camera_view_is_rejected(self):
        base=pd.DataFrame([{'time_sec':0,'global_id':'G','state':'OCCUPIED'},{'time_sec':1,'global_id':'G','state':'EMPTY'}])
        f=pd.DataFrame([{'time_sec':t,'global_id':'G','local_id':lid,'matches_initial_occupied':False,'full_observed':False,
                         'strong_aux_observed':False,'outward_motion_recent':False,'stable_changed_view':False}
                        for t,lid in [(0,'A'),(0,'B'),(1,'A')]])
        with self.assertRaisesRegex(ValueError,'camera views'):run_initial_candidate(base,f)
    def test_other_camera_occupied_prevents_a_departure(self):
        base=pd.DataFrame([{'time_sec':0,'global_id':'G','state':'OCCUPIED'},{'time_sec':1,'global_id':'G','state':'EMPTY'}])
        rows=[]
        for t in (0,1):
            for lid in ('A','B'):
                rows.append({'time_sec':t,'global_id':'G','local_id':lid,'matches_initial_occupied':False,
                    'full_observed':t==1 and lid=='B','strong_aux_observed':False,
                    'outward_motion_recent':t==1 and lid=='A','stable_changed_view':t==1 and lid=='A'})
        self.assertEqual(run_initial_candidate(base,pd.DataFrame(rows)).state.iloc[1],'UNKNOWN')
    def test_no_motion_or_unstable_image_never_proposes_departure(self):
        initial=np.indices((40,40)).sum(axis=0).astype('uint8');initial=np.repeat(initial[:,:,None],3,axis=2)
        changed=initial.copy();changed[10:30,10:30]=220
        d=InitialOccupiedCue('OCCUPIED',(0,0),0);d.observe(0,initial,False,0,-1,None,False,0,0)
        for t in (1,2,3):out=d.observe(t,changed,False,0,-1,None,False,0,0)
        self.assertTrue(out['stable_changed_view']);self.assertFalse(out['outward_motion_recent'])
    def test_stationary_missed_vehicle_is_retained(self):
        base=pd.DataFrame([{'time_sec':0,'global_id':'G','state':'OCCUPIED'},{'time_sec':1,'global_id':'G','state':'EMPTY'}])
        features=pd.DataFrame([{'time_sec':t,'global_id':'G','local_id':'A','matches_initial_occupied':True,'full_observed':False,
            'strong_aux_observed':False,'outward_motion_recent':False,'stable_changed_view':False} for t in (0,1)])
        self.assertEqual(run_initial_candidate(base,features).state.tolist(),['OCCUPIED','OCCUPIED'])


if __name__=='__main__':unittest.main()
