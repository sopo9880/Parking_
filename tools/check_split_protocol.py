"""Validate implemented split methodology without claiming measured performance."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from split_experiment import protocols

manifest=json.loads((ROOT/'paper_manifest.json').read_text(encoding='utf-8'))
path=ROOT/manifest['split_experiments']['protocol_record']
record=json.loads(path.read_text(encoding='utf-8'))
assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['split_experiments']['sha256']
assert record['full_real_video_results'] is None
assert record['status']=='implemented_functionally_validated_full_video_pending'
assert record['neural_network_retraining'] is False
assert record['independent_validation'] is False
assert record['default_protocols']==protocols(json.loads((ROOT/'settings.json').read_text(encoding='utf-8')))
for filename,digest in record['implementation_sha256_lf_utf8'].items():
    text=(ROOT/filename).read_text(encoding='utf-8')
    assert hashlib.sha256(text.encode('utf-8')).hexdigest()==digest, filename
print('Fresh split protocol: implementation provenance/defaults PASS; full-video performance remains pending')
