"""Recompute an archived spatial baseline; optionally reproduce selected detection overlays.

Never edits the archived dataset or predictions. All 0/1 predictions remain frozen.
Selected inference must exactly reproduce the archived slot decisions before images
are accepted as overlay evidence. This is not a full 4,073-image inference rerun.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import pandas as pd
from detector import VehicleDetector
from external_evidence import deduplicate_spots, select_representatives, write_results, freeze_baseline, file_hash
from public_dataset import _poly_array, _det_intersects_spot, SOURCE_PAGE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--infer-selected',action='store_true')
    args=parser.parse_args();root=args.dataset_root.resolve();out=args.output_dir.resolve()
    if out==args.predictions.resolve().parent or out==root:
        raise ValueError('Output must be a new folder; archived evidence is read-only')
    out.mkdir(parents=True,exist_ok=False)
    predictions,pred_audit=deduplicate_spots(pd.read_csv(args.predictions,encoding='utf-8-sig'),'archived_prediction_recalculation')
    print('Archived predictions:',json.dumps(pred_audit),flush=True)
    gt_path=root/'external_gt_spots.csv'
    gt,gt_audit=deduplicate_spots(pd.read_csv(gt_path,encoding='utf-8-sig'),'archived_gt_reconciliation')
    reference=predictions[['image_path','spot_id','occupied_gt']].merge(gt[['image_path','spot_id','occupied_gt']],on=['image_path','spot_id'],how='left',suffixes=('_prediction','_source'),validate='one_to_one')
    if reference.occupied_gt_source.isna().any() or (reference.occupied_gt_source!=reference.occupied_gt_prediction).any():
        raise ValueError('Archived predictions do not match the retained GT')
    enrich=[c for c in ('polygon_json','bbox_json','source_json') if c in gt and c not in predictions]
    predictions=predictions.merge(gt[['image_path','spot_id']+enrich],on=['image_path','spot_id'],how='left',validate='one_to_one')
    detections={};baseline=None
    if args.infer_selected:
        if not args.settings or not args.weights: raise ValueError('Selected inference needs settings and explicit original weights')
        import torch
        torch.set_num_threads(4)
        settings=json.loads(args.settings.read_text(encoding='utf-8'))
        config=dict(settings.get('detector',{}));external=settings.get('external_validation',{});config.update(external.get('detector',{}))
        if Path(config.get('model','yolov8m.pt')).name!=args.weights.name:
            raise ValueError('Weights name differs from the saved settings')
        baseline,_=freeze_baseline(out,config,external.get('spot_overlap_threshold',.12))
        baseline.update({'model_sha256':file_hash(args.weights),'evaluation_status':'recomputed_from_frozen_predictions_selected_inference_verified',
                         'original_run_settings_snapshot_available':False,
                         'configuration_provenance':'current supplied settings; selected images must reproduce archived predictions',
                         'prediction_sha256':file_hash(args.predictions),'gt_sha256':file_hash(gt_path),'settings_sha256':file_hash(args.settings)})
        infer_config={**config,'model':str(args.weights.resolve())};detector=VehicleDetector(infer_config)
        paths=sorted(set(c['image_path'] for c in select_representatives(predictions)))
        for i,path in enumerate(paths):
            image_path=(root/path).resolve()
            if not image_path.is_relative_to(root): raise ValueError('Image escapes dataset root')
            image=cv2.imread(str(image_path))
            if image is None: raise ValueError(f'Missing selected image {path}')
            dets=detector.detect(image);spots=predictions[predictions.image_path==path]
            for r in spots.itertuples():
                predicted=int(any(_det_intersects_spot(d,_poly_array(r.polygon_json),baseline['parameters']['spot_overlap_threshold']) for d in dets))
                if predicted!=int(r.occupied_pred): raise ValueError(f'Selected replay disagrees with archived predictions: {path} / {r.spot_id}')
            if 'detections_in_image' in spots and not (spots.detections_in_image==len(dets)).all():
                raise ValueError(f'Selected detection count disagrees: {path}')
            detections[path]=[d.as_dict() for d in dets]
            print(f'Selected inference verified {i+1}/{len(paths)}: {path}',flush=True)
        baseline['selected_images_verified']=len(paths)
        (out/'external_baseline.json').write_text(json.dumps(baseline,indent=2)+'\n',encoding='utf-8')
    result=write_results(predictions,out,root,SOURCE_PAGE,baseline,detections,
                         upstream_audit={'archived_predictions':pred_audit,'archived_gt':gt_audit})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
