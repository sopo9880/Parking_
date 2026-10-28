import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from research_manager import resource_sample


class ResourceMonitoringTests(unittest.TestCase):
    def test_records_live_process_tree_and_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])
            try:
                resource_sample(child, Path(tmp))
                with (Path(tmp)/'resources.csv').open(encoding='utf-8', newline='') as f:
                    row = next(csv.DictReader(f))
                self.assertEqual(int(row['pid']), child.pid)
                self.assertGreater(int(row['process_tree_rss_bytes']), 0)
                self.assertGreater(int(row['free_disk_bytes']), 0)
            finally:
                child.terminate(); child.wait(timeout=10)


if __name__ == '__main__': unittest.main()
