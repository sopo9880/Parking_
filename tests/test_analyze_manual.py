import unittest
import pandas as pd
from analyze_manual_experiment import score


class ManualScoreTests(unittest.TestCase):
    def test_unknown_does_not_become_empty_or_an_ordinary_error(self):
        d = pd.DataFrame({'error': [-2, 0], 'unknown_slots': [2, 0], 'gt_within_bounds': [True, True]})
        value = score(d, True)
        self.assertIsNone(value['exact']); self.assertIsNone(value['mae'])
        self.assertEqual(value['fully_resolved_N'], 1)
        self.assertEqual(value['gt_within_bounds_rate'], 1)

    def test_resolved_checkpoint_uses_ordinary_count_metrics(self):
        d = pd.DataFrame({'error': [-1, 0], 'unknown_slots': [0, 0], 'gt_within_bounds': [False, True]})
        value = score(d, True)
        self.assertEqual(value['exact'], .5); self.assertEqual(value['mae'], .5)


if __name__ == '__main__': unittest.main()
