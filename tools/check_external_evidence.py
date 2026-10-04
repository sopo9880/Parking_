"""Verify frozen archive hashes and recompute metrics without external images."""
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_evidence import binary_metrics, count_metrics, deduplicate_spots, image_counts

def compare(actual, expected):
    for key, value in actual.items():
        assert math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-12), (key, value, expected[key])

def main():
    manifest = json.loads((ROOT/'paper_manifest.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(ROOT/'paper/data/external_v1652/EXTERNAL_VALIDATION_TO_CHATGPT.zip') as z:
        assert z.testzip() is None
        for name, expected in json.loads(z.read('external_artifact_hashes.json')).items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
        summary = json.loads(z.read('external_summary.json'))
        df = pd.read_csv(io.BytesIO(z.read('external_slot_predictions.csv')))
        unique, audit = deduplicate_spots(df)
        assert audit['duplicates_removed'] == 0
        compare(binary_metrics(unique), summary['overall'])
        counts = image_counts(unique)
        compare(count_metrics(counts), summary['count_metrics'])
        gt = pd.read_csv(io.BytesIO(z.read('external_gt_spots.csv')))
        assert gt[['image_path','spot_id','occupied_gt']].equals(df[['image_path','spot_id','occupied_gt']])
        for group in ('camera','weather'):
            for filename, calculator, data in (
                ('external_metrics_by_camera_weather.csv', binary_metrics, unique),
                ('external_count_metrics_by_camera_weather.csv', count_metrics, counts)):
                rows = pd.read_csv(io.BytesIO(z.read(filename)))
                for name, values in data.groupby(group):
                    row = rows[(rows.scope == group.upper()) & (rows.name == name)].iloc[0]
                    compare(calculator(values), row.to_dict())
        source = summary['upstream_deduplication']['archived_predictions']
        assert source['input_rows'] == source['unique_pairs'] + source['duplicates_removed']
        assert source['unique_pairs'] == len(unique)
        assert sum(source['duplicates_by_camera'].values()) == source['duplicates_removed']
        index = pd.read_csv(io.BytesIO(z.read('screenshot_index.csv')))
        assert len(index) == summary['screenshots']['generated']
        assert set(index.category) == {'tp','tn','fp','fn'}
        for row in index.itertuples():
            assert hashlib.sha256(z.read(row.output_path)).hexdigest() == row.sha256
        boxes = [json.loads(line) for line in z.read('external_image_detections.jsonl').decode().splitlines()]
        assert len(boxes) == summary['baseline']['selected_images_verified']
        assert set(index.image_path).issubset({d['image_path'] for d in boxes})
    for asset in manifest['external_evidence']['artifacts']:
        assert hashlib.sha256((ROOT/asset['path']).read_bytes()).hexdigest() == asset['sha256']
    print(f"External evidence PASS: {len(unique):,} unique pairs, {len(counts):,} images; hashes, GT, metrics, groups and screenshot provenance verified")

if __name__ == '__main__':
    main()
