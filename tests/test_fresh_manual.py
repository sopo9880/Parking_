import unittest
import pandas as pd
from fresh_manual_experiment import pseudo_snapshot


class PseudoSnapshotTests(unittest.TestCase):
    def test_only_exact_time_used_and_missing_is_unknown(self):
        rows = pd.DataFrame([{'time_sec': 900, 'global_id': 'A', 'full_detected': 1, 'full_det_conf': .5},
                             {'time_sec': 900, 'global_id': 'B', 'full_detected': 0, 'full_det_conf': 0},
                             {'time_sec': 901, 'global_id': 'B', 'full_detected': 1, 'full_det_conf': .9}])
        self.assertEqual(pseudo_snapshot(rows, {'A', 'B', 'C'}, 900), {'A': 'OCCUPIED', 'B': 'UNKNOWN', 'C': 'UNKNOWN'})

    def test_unknown_global_id_rejected(self):
        rows = pd.DataFrame([{'time_sec': 900, 'global_id': 'wrong', 'full_detected': 1, 'full_det_conf': .8}])
        with self.assertRaises(ValueError):
            pseudo_snapshot(rows, {'A'}, 900)


if __name__ == '__main__': unittest.main()
