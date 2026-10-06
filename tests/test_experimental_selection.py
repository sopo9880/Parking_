import unittest
import pandas as pd
from experimental_selection import choose_dev_variant


class ExperimentalSelectionTests(unittest.TestCase):
    def board(self):
        return pd.DataFrame([
            {'temporal_variant':n,'dev_N':91,'dev_exact_rate':e,'dev_MAE':m,
             'dev_over_rate':o,'dev_max_abs_error':1,'test_exact_rate':t}
            for n,e,m,o,t in [('SAFE_BASELINE',.318681,.681319,.197802,1),
                             ('TRANSITION_GUARD',.736264,.263736,.010989,0),
                             ('SEG_ASSIST',.736264,.263736,.010989,1)]])

    def test_dev_only_selection_and_simple_tie_break(self):
        b=self.board()
        self.assertEqual(choose_dev_variant(b),'TRANSITION_GUARD')
        b['test_exact_rate']=[0,1,0]
        self.assertEqual(choose_dev_variant(b),'TRANSITION_GUARD')
        for key in ('dev_exact_rate','dev_MAE','dev_over_rate'):
            b[key]=b[key].iloc[0]
        self.assertEqual(choose_dev_variant(b),'SAFE_BASELINE')

    def test_missing_or_invalid_dev_is_rejected(self):
        b=self.board();b.loc[0,'dev_N']=0
        with self.assertRaises(ValueError):choose_dev_variant(b)
        with self.assertRaises(ValueError):choose_dev_variant(self.board().iloc[:2])


if __name__=='__main__':unittest.main()
