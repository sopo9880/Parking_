import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from detector import Detection
from external_evidence import deduplicate_spots, binary_metrics, image_counts, count_metrics, freeze_baseline, write_results
from public_dataset import evaluate_external_cnr, convert_spot_annotations


class ExternalEvidenceTests(unittest.TestCase):
    def fixture(self):
        # Synthetic test-only evidence; never copied into the paper.
        return pd.DataFrame([
            dict(image_path='images/a.jpg',camera='camera1',weather='SUNNY',spot_id=i,
                 occupied_gt=gt,occupied_pred=pred,polygon_json=json.dumps([[5+i*45,5],[40+i*45,5],[40+i*45,60],[5+i*45,60]]))
            for i,(gt,pred) in enumerate([(1,1),(0,0),(0,1),(1,0)])
        ]+[dict(image_path='images/b.jpg',camera='camera2',weather='RAINY',spot_id=1,occupied_gt=0,occupied_pred=1,polygon_json='[[10,10],[50,10],[50,60],[10,60]]')])

    def test_duplicate_keys_and_conflict_rejection(self):
        df=self.fixture();dup=df.iloc[[0]].copy();dup['image_path']='./images\\a.jpg'
        unique,audit=deduplicate_spots(pd.concat([df,dup]))
        self.assertEqual(len(unique),5);self.assertEqual(audit['duplicates_removed'],1)
        dup['occupied_gt']=0
        with self.assertRaisesRegex(ValueError,'Conflicting'): deduplicate_spots(pd.concat([df,dup]))
        for path in ('../escape.jpg','C:/bad.jpg','/bad.jpg'):
            wrong=df.copy();wrong.loc[0,'image_path']=path
            with self.assertRaises(ValueError): deduplicate_spots(wrong)

    def test_binary_and_count_metrics_have_distinct_denominators(self):
        df=self.fixture();m=binary_metrics(df);c=count_metrics(image_counts(df))
        self.assertEqual((m['TP'],m['TN'],m['FP'],m['FN']),(1,1,2,1))
        self.assertEqual(m['accuracy'],.4)
        self.assertEqual(c['images'],2);self.assertEqual(c['count_exact'],.5)
        self.assertEqual(c['count_mae'],.5);self.assertEqual(c['count_bias'],.5)
        self.assertEqual(c['over_count_rate'],.5);self.assertEqual(c['under_count_rate'],0)
        self.assertEqual(c['count_p90_error'],.9)

    def test_baseline_ignores_changed_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            first,changed=freeze_baseline(temp,{'conf':.15},.12)
            self.assertFalse(changed)
            second,changed=freeze_baseline(temp,{'conf':.99},.8)
            self.assertTrue(changed);self.assertEqual(first,second)

    def test_conversion_deduplicates_two_extraction_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'images').mkdir();cv2.imwrite(str(root/'images/a.jpg'),np.zeros((100,200,3),np.uint8))
            (root/'manifest.json').write_text(json.dumps({'images':[{'repo_path':'CNRPark-EXT/camera1/SUNNY/a.jpg','local_path':'images/a.jpg'}]}))
            annotation={'images':[{'id':1,'file_name':'CNRPark-EXT/camera1/SUNNY/a.jpg'}],'annotations':[{'id':10,'image_id':1,'category_id':1,'segmentation':[[1,1,20,1,20,20,1,20]]}]}
            for folder in ('1_spots.tar','camera1'):
                p=root/'annotations_extracted'/folder;p.mkdir(parents=True);(p/'spots.json').write_text(json.dumps(annotation))
            gt=convert_spot_annotations(root)
            self.assertEqual(len(pd.read_csv(gt)),1)
            audit=json.loads((root/'annotation_deduplication.json').read_text());self.assertEqual(audit['duplicates_removed'],1)

    def test_inference_evaluates_each_unique_slot_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'images').mkdir();cv2.imwrite(str(root/'images/a.jpg'),np.zeros((100,200,3),np.uint8))
            df=self.fixture().iloc[:4].drop(columns='occupied_pred');pd.concat([df,df.iloc[[0]]]).to_csv(root/'external_gt_spots.csv',index=False)
            class FakeDetector:
                def __init__(self,cfg): pass
                def detect(self,image): return [Detection(5,5,40,60,.9,2)]
            with patch('public_dataset.VehicleDetector',FakeDetector): result=evaluate_external_cnr(root,{'detector':{'conf':.15}},root/'results')
            self.assertEqual(result['samples'],4)
            summary=json.loads((root/'results/external_summary.json').read_text())
            self.assertEqual(summary['upstream_deduplication']['evaluation_input']['duplicates_removed'],1)

    def test_real_image_overlay_package_and_unavailable_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'images').mkdir()
            for name in ('a','b'): cv2.imwrite(str(root/f'images/{name}.jpg'),np.full((100,200,3),120,np.uint8))
            df=self.fixture();boxes={p:[dict(x1=5,y1=5,x2=40,y2=60,conf=.9,cls=2)] for p in df.image_path.unique()}
            result=write_results(df,root/'results',root,'synthetic-test-only',detections=boxes)
            self.assertEqual(set(result['screenshots']['categories_present']),{'tp','tn','fp','fn'})
            with zipfile.ZipFile(result['package']) as package:
                names=package.namelist();self.assertIn('paper_ready/external_validation_montage.jpg',names)
                self.assertIn('external_image_counts.csv',names);self.assertIn('external_artifact_hashes.json',names)
            result=write_results(df,root/'missing',root,'test',detections={})
            self.assertEqual(result['screenshots']['generated'],0)
            self.assertFalse((root/'missing/paper_ready/external_validation_montage.jpg').exists())

if __name__=='__main__': unittest.main()
