import unittest
import pandas as pd
from analyze_reviewed_events import summarize


class ReviewedEventMetricsTests(unittest.TestCase):
    def test_clip_label_does_not_create_departure_time(self):
        case={'case_id':'x','global_id':'G','clip_start':10,'clip_end':20}
        trace=pd.DataFrame([{'global_id':'G','time_sec':t,'state':s} for t,s in [(10,'OCCUPIED'),(15,'EMPTY'),(20,'EMPTY')]])
        result=summarize(case,'DEPARTURE',trace)
        self.assertEqual(result['first_empty_sec'],15)
        self.assertNotIn('departure_time',result);self.assertNotIn('delay_sec',result)
    def test_parked_interval_identifies_empty_proposal(self):
        case={'case_id':'x','global_id':'G','clip_start':10,'clip_end':20}
        trace=pd.DataFrame([{'global_id':'G','time_sec':t,'state':s} for t,s in [(10,'OCCUPIED'),(15,'EMPTY'),(20,'OCCUPIED')]])
        self.assertTrue(summarize(case,'PARKED',trace)['false_empty_in_parked_clip'])


if __name__=='__main__':unittest.main()
