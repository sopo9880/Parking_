"""Recalculate published retained-video window evidence and verify fingerprints."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from window_evaluation import fingerprint,write_window_evaluation

def main():
    manifest=json.loads((ROOT/'paper_manifest.json').read_text(encoding='utf-8'))
    source=ROOT/'paper/data/windows_v1653'
    summary=json.loads((source/'window_evaluation_summary.json').read_text(encoding='utf-8'))
    for item in manifest['window_evidence']['artifacts']:
        assert fingerprint(ROOT/item['path'])==item['sha256'],item['path']
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)
        for name,digest in summary['input_sha256'].items():
            if (source/name).is_file():
                assert fingerprint(source/name)==digest
                shutil.copy2(source/name,out/name)
        settings={'evaluation_warmup_sec':summary['global_warmup_sec'],'validation':{
            'evaluation_windows':summary['windows'],'cut_boundaries_sec':summary['cuts_sec'],'cut_warmup_sec':summary['cut_warmup_sec']}}
        write_window_evaluation(out,settings)
        for name in ('window_metrics.csv','window_comparison.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(source/name),pd.read_csv(out/name))
        actual=json.loads((out/'window_evaluation_summary.json').read_text())
        for key in ('algorithms','compared_windows','candidate_exact_improved_windows','candidate_MAE_improved_windows'):
            assert actual[key]==summary[key],key
    print('Frozen window evidence PASS: complete paired traces, per-window metrics and summaries recomputed')

if __name__=='__main__': main()
