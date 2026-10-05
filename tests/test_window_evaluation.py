import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from window_evaluation import parse_windows, write_window_evaluation, fingerprint

class WindowEvaluationTests(unittest.TestCase):
    def prepare(self,root):
        # Errors at boundaries differ to detect double inclusion and unequal denominators.
        pd.DataFrame({'time_sec':[0,10,20,30,40,50,60],'occupied_pred':[0,1,2,0,1,0,2],
                      'ground_truth_occupied_space_count':[1]*7}).to_csv(root/'baseline_count_timeseries.csv',index=False)
        pd.DataFrame({'time_sec':[0,10,20,30,40,60],'occupied_pred_candidate':[1,1,1,2,1,1]}).to_csv(root/'candidate_v164_count_timeseries.csv',index=False)
        return {'evaluation_warmup_sec':10,'validation':{'evaluation_windows':parse_windows('0-30;30-60')}}

    def test_shared_pairs_boundary_metrics_and_source_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);settings=self.prepare(root)
            hashes={p.name:fingerprint(p) for p in root.glob('*.csv')}
            write_window_evaluation(root,settings)
            m=pd.read_csv(root/'window_metrics.csv');c=pd.read_csv(root/'window_comparison.csv').set_index('window')
            self.assertEqual(c.loc['W1','N'],2);self.assertEqual(c.loc['W2','N'],3);self.assertEqual(c.loc['ALL','N'],5)
            self.assertEqual(c.loc['W1','exact_delta_pp'],50)
            self.assertAlmostEqual(c.loc['W2','MAE_delta'],-1/3)
            self.assertEqual(m[(m.window=='W2')&(m.algorithm=='CANDIDATE')].iloc[0].missing_pair_N,1)
            self.assertEqual(hashes,{name:fingerprint(root/name) for name in hashes})
            s=json.loads((root/'window_evaluation_summary.json').read_text());self.assertFalse(s['parameters_tuned'])
            self.assertEqual(s['candidate_exact_improved_windows'],2);self.assertEqual(s['compared_windows'],2)
            self.assertAlmostEqual(s['algorithms']['SAFE_BASELINE']['std_exact'],1/12)

    def test_cut_warmup_and_empty_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);settings=self.prepare(root)
            settings['validation'].update(cut_boundaries_sec=[30],cut_warmup_sec=20,evaluation_windows=parse_windows('30-40;100-110'))
            write_window_evaluation(root,settings)
            s=json.loads((root/'window_evaluation_summary.json').read_text());self.assertEqual(s['compared_windows'],0)
            self.assertIsNone(s['algorithms']['SAFE_BASELINE']['mean_exact'])
            self.assertEqual(pd.read_csv(root/'window_comparison.csv').set_index('window').loc['ALL','N'],3)

    def test_duplicates_and_gt_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);settings=self.prepare(root);p=root/'candidate_v164_count_timeseries.csv'
            c=pd.read_csv(p);pd.concat([c,c.iloc[:1]]).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'timestamps'): write_window_evaluation(root,settings)
            c['ground_truth_occupied_space_count']=2;c.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'ground truth differ'): write_window_evaluation(root,settings)

    def test_custom_times_and_invalid_intervals(self):
        self.assertEqual(parse_windows('5:30–11:00')[0]['start_sec'],330)
        self.assertEqual(parse_windows('1:00:00-1:05:30')[0]['end_sec'],3930)
        for text in ('','0-0','60-30','0:99-2:00','nan-30','0-10;0-10'):
            with self.assertRaises(ValueError): parse_windows(text)

if __name__=='__main__': unittest.main()
