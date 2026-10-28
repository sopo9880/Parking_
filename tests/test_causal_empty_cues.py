import unittest
import numpy as np
from causal_empty_cues import CausalEmptyCue


class EmptyCueTests(unittest.TestCase):
    def setUp(self):
        self.empty=np.indices((40,40)).sum(axis=0).astype('uint8')
        self.empty=np.repeat(self.empty[:,:,None],3,axis=2)
        self.occupied=self.empty.copy();self.occupied[10:30,10:30]=220
    def test_requires_empty_reference_motion_and_sustained_return(self):
        d=CausalEmptyCue('EMPTY',(0,0),0)
        d.observe(0,self.empty,False,0,-1,None,'EMPTY')
        for t,x in [(1,1),(2,4),(3,12)]: d.observe(t,self.occupied,True,.9,7,(x,0),'OCCUPIED')
        last=None
        for t in (4,5,6):last=d.observe(t,self.empty,False,0,-1,None,'EMPTY')
        self.assertTrue(last['outward_motion']);self.assertTrue(last['empty_visible'])
    def test_initial_occupied_has_no_empty_reference(self):
        d=CausalEmptyCue('OCCUPIED',(0,0),0)
        for t in range(5):result=d.observe(t,self.empty,False,0,-1,None,'EMPTY')
        self.assertFalse(result['empty_visible']);self.assertIsNone(result['empty_ref_difference'])
    def test_future_is_not_read_and_order_is_required(self):
        d=CausalEmptyCue('EMPTY',(0,0),0);a=d.observe(0,self.empty,False,0,-1,None,'EMPTY')
        with self.assertRaises(ValueError):d.observe(0,self.occupied,True,.9,1,(0,0),'OCCUPIED')
        self.assertFalse(a['outward_motion']);self.assertFalse(a['empty_visible'])


if __name__=='__main__':unittest.main()
