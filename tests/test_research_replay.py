import unittest
from research_replay import select_dev


class ReplaySelectionTests(unittest.TestCase):
    def test_test_labels_cannot_change_selection(self):
        rows = [dict(split='a', partition='DEV', variant='SAFE', exact_rate=.5, mae=1),
                dict(split='a', partition='DEV', variant='NEW', exact_rate=.6, mae=1)]
        first = select_dev(rows, 'a', ['SAFE', 'NEW'])
        rows.append(dict(split='a', partition='TEST', variant='SAFE', exact_rate=1, mae=0))
        self.assertEqual(first, select_dev(rows, 'a', ['SAFE', 'NEW']))
        self.assertEqual(first['selected_on_dev'], 'NEW')

    def test_incomplete_dev_is_rejected(self):
        rows = [dict(split='a', partition='DEV', variant='SAFE', exact_rate=1, mae=0)]
        with self.assertRaises(ValueError):
            select_dev(rows, 'a', ['SAFE', 'NEW'])


if __name__ == '__main__':
    unittest.main()
