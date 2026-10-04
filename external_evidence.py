"""Auditable external spatial metrics and real-image evidence; no parameter tuning."""
from __future__ import annotations
import hashlib
import json
import posixpath
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def canonical_image(value):
    text=posixpath.normpath(str(value).strip().replace('\\','/'))
    if text in ('','.') or text.startswith(('../','/')) or re.match(r'^[A-Za-z]:',text):
        raise ValueError(f'Image path must be relative to dataset root: {value}')
    return text


def deduplicate_spots(frame, context='annotations'):
    """Keep each image/spot once; conflicting duplicates must never be silently scored."""
    required={'image_path','spot_id','occupied_gt'}
    if not required.issubset(frame.columns): raise ValueError(f'Missing columns: {required-set(frame.columns)}')
    if frame[list(required)].isna().any().any(): raise ValueError('Missing image/spot/GT identity')
    df=frame.copy();df['image_path']=df['image_path'].map(canonical_image)
    ids=pd.to_numeric(df['spot_id'],errors='raise')
    if ((ids%1)!=0).any(): raise ValueError('spot_id must be an integer')
    df['spot_id']=ids.astype('int64')
    for column in ('camera','weather'):
        if column not in df: df[column]='unknown'
    for column in ('occupied_gt','occupied_pred'):
        if column in df:
            values=pd.to_numeric(df[column],errors='raise')
            if not values.isin([0,1]).all(): raise ValueError(f'{column} must be binary 0/1')
            df[column]=values.astype(int)
    conflicts=[];duplicate=df.duplicated(['image_path','spot_id'],keep=False)
    compare=[c for c in ('occupied_gt','occupied_pred','camera','weather','polygon_json') if c in df]
    for key,group in df.loc[duplicate].groupby(['image_path','spot_id'],sort=False):
        for column in compare:
            values=group[column].map(lambda x:json.dumps(json.loads(x),sort_keys=True) if column=='polygon_json' else str(x))
            if values.nunique(dropna=False)>1: conflicts.append({'image_path':key[0],'spot_id':int(key[1]),'column':column})
    if conflicts: raise ValueError(f'Conflicting duplicate image/spot annotations: {conflicts[:5]}')
    removed=df[df.duplicated(['image_path','spot_id'],keep='first')]
    unique=df.drop_duplicates(['image_path','spot_id'],keep='first').reset_index(drop=True)
    counts=removed.groupby('camera').size().to_dict() if 'camera' in removed else {}
    audit={'schema_version':1,'context':context,'key':['canonical_image_path','spot_id'],
           'input_rows':len(df),'unique_pairs':len(unique),'duplicates_removed':len(removed),
           'unique_images':int(unique['image_path'].nunique()),'duplicates_by_camera':{str(k):int(v) for k,v in counts.items()},'conflicting_pairs':0}
    return unique,audit


def freeze_baseline(root, detector, overlap_threshold):
    path=Path(root)/'external_baseline.json'
    proposed={'detector':detector,'spot_overlap_threshold':float(overlap_threshold)}
    def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if path.exists():
        data=json.loads(path.read_text(encoding='utf-8'))
        if digest(data['parameters'])!=data['parameter_sha256']: raise ValueError('Frozen external baseline was modified')
        return data, digest(proposed)!=data['parameter_sha256']
    weights=Path(str(detector.get('model','')))
    data={'schema_version':1,'name':'EXTERNAL_BASELINE','frozen_by_version':'v16.5.2',
          'purpose':'spatial occupancy generalization only; no target-label tuning',
          'parameters':proposed,'parameter_sha256':digest(proposed),
          'model_sha256':file_hash(weights) if weights.is_file() else None}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)
    return data,False


def binary_metrics(df):
    y=df['occupied_gt'];p=df['occupied_pred']
    tp=int(((y==1)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());n=len(df)
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    specificity=tn/(tn+fp) if tn+fp else 0.
    return {'N':n,'accuracy':(tp+tn)/n if n else 0.,'precision':precision,'occupied_recall':recall,
            'empty_specificity':specificity,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'balanced_accuracy':(recall+specificity)/2 if tp+fn and tn+fp else None,
            'TP':tp,'TN':tn,'FP':fp,'FN':fn}


def image_counts(predictions):
    rows=[]
    for path,g in predictions.groupby('image_path',sort=True):
        for col in ('camera','weather'):
            if col in g and g[col].nunique(dropna=False)!=1: raise ValueError(f'Inconsistent {col} for {path}')
        true=int(g.occupied_gt.sum());pred=int(g.occupied_pred.sum())
        rows.append({'image_path':path,'camera':str(g.iloc[0].get('camera','')),'weather':str(g.iloc[0].get('weather','')),
                     'evaluated_spots':len(g),'gt_occupied_spots':true,'pred_occupied_spots':pred,
                     'count_error':pred-true,'abs_error':abs(pred-true),
                     'detections_in_image':g.iloc[0].get('detections_in_image',None)})
    return pd.DataFrame(rows)


def count_metrics(counts):
    errors=counts.count_error.to_numpy(dtype=float);absolute=np.abs(errors)
    if not len(errors): raise ValueError('No evaluated image counts')
    return {'images':len(errors),'count_exact':float(np.mean(errors==0)),'count_mae':float(np.mean(absolute)),
            'count_median_abs_error':float(np.median(absolute)),'count_p90_error':float(np.percentile(absolute,90)),
            'count_p95_error':float(np.percentile(absolute,95)),'count_max_error':int(np.max(absolute)),
            'over_count_rate':float(np.mean(errors>0)),'under_count_rate':float(np.mean(errors<0)),
            'count_bias':float(np.mean(errors))}


def select_representatives(predictions,limit=3):
    """Deterministic descriptive case selection, never used to tune evaluation."""
    p=predictions.copy();p['category']=np.select([
        (p.occupied_gt==1)&(p.occupied_pred==1),(p.occupied_gt==0)&(p.occupied_pred==0),
        (p.occupied_gt==0)&(p.occupied_pred==1)],['tp','tn','fp'],default='fn')
    error=p[p.occupied_gt!=p.occupied_pred].groupby('image_path').size().to_dict()
    p['_errors']=p.image_path.map(error).fillna(0)
    selected=[]
    def pick(group,kind,name,n=1):
        group=group.sort_values(['_errors','image_path','spot_id'],ascending=[False,True,True]).drop_duplicates('image_path').head(n)
        for row in group.to_dict('records'):
            selected.append({'group':kind,'name':name,**{k:row[k] for k in ('image_path','spot_id','camera','weather','category','occupied_gt','occupied_pred')}})
    for cat in ('tp','tn','fp','fn'): pick(p[p.category==cat],cat,cat,max(1,min(12,int(limit))))
    for camera,group in p.groupby('camera'):
        failures=group[group.category=='fp']
        pick(failures if not failures.empty else group,'by_camera',str(camera))
        correct=group[group.occupied_gt==group.occupied_pred]
        if not correct.empty: pick(correct.sort_values('_errors',ascending=True).head(1),'by_camera',str(camera)+'_success')
    for weather,group in p.groupby('weather'): pick(group,'by_weather',str(weather))
    return selected


def _safe_name(value): return re.sub(r'[^A-Za-z0-9_-]','_',str(value))[:100]


def _letterbox(image,width,height):
    h,w=image.shape[:2];scale=min(width/max(1,w),height/max(1,h))
    resized=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
    out=np.full((height,width,3),25,np.uint8);x=(width-resized.shape[1])//2;y=(height-resized.shape[0])//2
    out[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return out


def render_overlay(image,spots,detections,target):
    canvas=image.copy();height,width=canvas.shape[:2];thickness=max(1,round(max(height,width)/650))
    colors={'tp':(70,210,70),'tn':(220,185,60),'fp':(50,60,240),'fn':(230,80,180)}
    target_poly=None
    for r in spots.to_dict('records'):
        poly=np.asarray(json.loads(r['polygon_json']),dtype=float).reshape(-1,2)
        if not np.isfinite(poly).all() or len(poly)<3: raise ValueError('Invalid screenshot polygon')
        points=np.rint(poly).astype(np.int32)
        gt=int(r['occupied_gt']);pred=int(r['occupied_pred']);cat=('tp' if pred else 'fn') if gt else ('fp' if pred else 'tn')
        chosen=int(r['spot_id'])==int(target['spot_id'])
        cv2.polylines(canvas,[points],True,(255,255,255) if chosen else colors[cat],thickness*(3 if chosen else 1))
        if chosen: target_poly=points
    for d in detections:
        x1,y1,x2,y2=[int(round(float(d[k]))) for k in ('x1','y1','x2','y2')]
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,210,255),thickness)
    if target_poly is None: raise ValueError('Selected slot missing polygon')
    x,y,w,h=cv2.boundingRect(target_poly);pad=max(24,round(max(w,h)*.8))
    crop=canvas[max(0,y-pad):min(height,y+h+pad),max(0,x-pad):min(width,x+w+pad)]
    result=np.full((780,1440,3),245,np.uint8)
    result[110:710,:1000]=_letterbox(canvas,1000,600);result[110:710,1000:]=_letterbox(crop,440,600)
    title=f"{target['category'].upper()} | {target['camera']} | {target['weather']} | spot {target['spot_id']} | GT={target['occupied_gt']} Pred={target['occupied_pred']}"
    cv2.putText(result,title,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),2,cv2.LINE_AA)
    name=str(target['image_path']);name=name if len(name)<115 else '...'+name[-110:]
    cv2.putText(result,name,(20,70),cv2.FONT_HERSHEY_SIMPLEX,.48,(50,50,50),1,cv2.LINE_AA)
    cv2.putText(result,'All evaluated polygons | yellow: detections | white: selected spot | right: selected spot zoom',(20,95),cv2.FONT_HERSHEY_SIMPLEX,.53,(30,30,30),1,cv2.LINE_AA)
    cv2.putText(result,'TP=green TN=blue FP=red FN=purple | GT and Pred refer to parking-space occupancy',(20,751),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),1,cv2.LINE_AA)
    return result


def screenshots(predictions,root,out,detections,limit=3):
    root=Path(root).resolve();out=Path(out)
    for sub in ('tp','tn','fp','fn','by_camera','by_weather'): (out/'screenshots'/sub).mkdir(parents=True,exist_ok=True)
    (out/'paper_ready').mkdir(exist_ok=True)
    selected=select_representatives(predictions,limit);index=[]
    for case in selected:
        path=(root/case['image_path']).resolve()
        if not path.is_relative_to(root): raise ValueError('Image escapes dataset root')
        if case['image_path'] not in detections or 'polygon_json' not in predictions:
            index.append({**case,'status':'missing_detection_or_polygon_evidence','output_path':''});continue
        image=cv2.imread(str(path))
        if image is None:
            index.append({**case,'status':'missing_image','output_path':''});continue
        panel=render_overlay(image,predictions[predictions.image_path==case['image_path']],detections[case['image_path']],case)
        unique=hashlib.sha256(case['image_path'].encode()).hexdigest()[:10]
        relative=f"screenshots/{case['group']}/{_safe_name(case['name'])}_{unique}_spot{case['spot_id']}.jpg"
        if not cv2.imwrite(str(out/relative),panel,[int(cv2.IMWRITE_JPEG_QUALITY),92]): raise IOError(relative)
        index.append({**case,'status':'generated','output_path':relative,'source_image_sha256':file_hash(path),'sha256':file_hash(out/relative)})
    generated=[r for r in index if r['status']=='generated']
    chosen=[]
    for category in ('fp','fn','tp','tn'):
        row=next((r for r in generated if r['group']==category),None)
        if row:
            target=f"paper_ready/representative_{category}.jpg";shutil.copy2(out/row['output_path'],out/target);chosen.append(row)
    if generated:
        # Include requested camera1/camera9 failures and camera2 representative when available.
        for camera in ('camera1','camera9','camera2'):
            row=next((r for r in generated if r['group']=='by_camera' and r['camera'].lower()==camera),None)
            if row and not any(c['output_path']==row['output_path'] for c in chosen): chosen.append(row)
        priority=[]
        for camera in ('camera1','camera9','camera2'):
            row=next((r for r in generated if r['group']=='by_camera' and
                      r['camera'].lower()==camera and
                      (r['name'].endswith('_success') if camera=='camera2' else r['category']=='fp')),None)
            if row: priority.append(row)
        for row in chosen:
            if not any(r['image_path']==row['image_path'] and r['spot_id']==row['spot_id'] for r in priority): priority.append(row)
        chosen=priority[:6];montage=np.full((400*((len(chosen)+1)//2),1440,3),255,np.uint8)
        for i,row in enumerate(chosen): montage[(i//2)*400:(i//2+1)*400,(i%2)*720:(i%2+1)*720]=_letterbox(cv2.imread(str(out/row['output_path'])),720,400)
        cv2.imwrite(str(out/'paper_ready/external_validation_montage.jpg'),montage,[int(cv2.IMWRITE_JPEG_QUALITY),94])
    pd.DataFrame(index).to_csv(out/'screenshot_index.csv',index=False,encoding='utf-8-sig')
    return {'requested':len(selected),'generated':len(generated),'unavailable':len(selected)-len(generated),'categories_present':sorted(set(r['category'] for r in generated)),
            'selection_policy':'deterministic category/camera/weather cases; maximum observed errors for failure illustrations, not a random sample'}


def charts(predictions,counts,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)/'paper_ready';out.mkdir(parents=True,exist_ok=True)
    cameras=sorted(predictions.camera.unique(),key=lambda name:int(re.search(r'\d+',str(name)).group()) if re.search(r'\d+',str(name)) else 999)
    weather=sorted(predictions.weather.unique())
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,key,labels,title in ((axes[0],'camera',cameras,'Spatial occupancy accuracy by camera'),(axes[1],'weather',weather,'Spatial occupancy accuracy by weather')):
        values=[binary_metrics(predictions[predictions[key]==name])['accuracy']*100 for name in labels]
        bars=ax.bar(labels,values,color=['#d15f50' if name in ('camera1','camera9') else '#397ba8' for name in labels])
        ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_title(title,fontsize=11);ax.tick_params(axis='x',rotation=45)
        ax.bar_label(bars,fmt='%.2f',fontsize=7);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(out/'external_accuracy_by_camera_weather.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    values=[count_metrics(counts[counts.camera==name])['count_mae'] for name in cameras]
    bars=axes[0].bar(cameras,values,color='#397ba8');axes[0].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[0].set_ylabel('MAE (annotated occupied spaces)');axes[0].set_title('Image-level count error by camera');axes[0].tick_params(axis='x',rotation=45)
    axes[1].hist(counts.count_error,bins=np.arange(counts.count_error.min()-.5,counts.count_error.max()+1.5),color='#397ba8')
    axes[1].axvline(0,color='#d15f50',linestyle='--');axes[1].set_xlabel('Predicted minus GT occupied spaces');axes[1].set_ylabel('Images');axes[1].set_title('Signed image-level count error')
    fig.savefig(out/'external_count_error.png',dpi=180);plt.close(fig)


def write_results(predictions,out,root,source,baseline=None,detections=None,upstream_audit=None,skipped=None,screenshot_limit=3):
    out=Path(out);root=Path(root);out.mkdir(parents=True,exist_ok=True)
    df,audit=deduplicate_spots(predictions,'evaluation_predictions')
    if df.empty: raise ValueError('No evaluated predictions')
    df['correct']=(df.occupied_gt==df.occupied_pred).astype(int)
    df.to_csv(out/'external_slot_predictions.csv',index=False,encoding='utf-8-sig')
    overall=binary_metrics(df);counts=image_counts(df);cm=count_metrics(counts)
    counts.to_csv(out/'external_image_counts.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'scope':'ALL',**overall}]).to_csv(out/'external_metrics_overall.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'scope':'ALL',**cm}]).to_csv(out/'external_count_metrics_overall.csv',index=False,encoding='utf-8-sig')
    by=[];by_count=[]
    for column in ('camera','weather'):
        for name,g in df.groupby(column): by.append({'scope':column.upper(),'name':str(name),**binary_metrics(g)})
        for name,g in counts.groupby(column): by_count.append({'scope':column.upper(),'name':str(name),**count_metrics(g)})
    pd.DataFrame(by).to_csv(out/'external_metrics_by_camera_weather.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(by_count).to_csv(out/'external_count_metrics_by_camera_weather.csv',index=False,encoding='utf-8-sig')
    detections=detections or {}
    with (out/'external_image_detections.jsonl').open('w',encoding='utf-8') as f:
        for image,boxes in sorted(detections.items()): f.write(json.dumps({'image_path':image,'detections':boxes})+'\n')
    screen=screenshots(df,root,out,detections,screenshot_limit)
    charts(df,counts,out)
    source_audit=next((upstream_audit.get(k) for k in ('archived_predictions','annotation_conversion','evaluation_input')
                       if upstream_audit and upstream_audit.get(k)),audit)
    skip=skipped or [];pd.DataFrame(skip,columns=['image_path','reason']).to_csv(out/'external_skipped_images.csv',index=False,encoding='utf-8-sig')
    summary={'schema_version':1,'app_version':'v16.5.2','dataset':'MetaPKLot/CNRPark-EXT','evidence_scope':'external_spatial_occupancy',
             'evaluation_status':baseline.get('evaluation_status','recomputed') if baseline else 'recomputed_from_predictions_parameters_unverified',
             'source':source,'baseline':baseline,'overall':overall,'count_metrics':cm,'deduplication':audit,
             'upstream_deduplication':upstream_audit,'source_duplicate_rows_removed':int(source_audit['duplicates_removed']),
             'skipped_images':len(skip),'screenshots':screen,
             'count_definition':'sum of occupied annotated parking spaces per image; not all vehicles or cross-camera unique vehicles'}
    camera_metrics=[r for r in by if r['scope']=='CAMERA'];weather_metrics=[r for r in by if r['scope']=='WEATHER']
    summary['descriptive_analysis']={'camera_accuracy_range_pp':100*(max(r['accuracy'] for r in camera_metrics)-min(r['accuracy'] for r in camera_metrics)) if camera_metrics else None,
        'weather_accuracy_range_pp':100*(max(r['accuracy'] for r in weather_metrics)-min(r['accuracy'] for r in weather_metrics)) if weather_metrics else None,
        'fp_to_fn_ratio':overall['FP']/overall['FN'] if overall['FN'] else None,
        'warning':'Group ranges are descriptive and confounded; they do not prove camera geometry causes performance differences.'}
    (out/'external_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['Parking Research Agent v16.5.2 external spatial baseline','NOT a continuous temporal-transition benchmark.',
           f"Evaluation status: {summary['evaluation_status']}",f"Images: {cm['images']} | Unique image/spot pairs: {overall['N']}",
           f"Evaluation duplicate rows removed: {audit['duplicates_removed']}",
           f"Source duplicate rows removed: {summary['source_duplicate_rows_removed']}",
           f"Preparation deduplication audit: {json.dumps(upstream_audit,ensure_ascii=False)}",f"Skipped images: {len(skip)}",'Count means occupied annotated parking spaces; not total unique vehicles.',
           *[f'{k}: {v}' for k,v in overall.items()],*[f'{k}: {v}' for k,v in cm.items()],
           f"Screenshot evidence: {json.dumps(screen)}",f"Frozen baseline: {json.dumps(baseline,ensure_ascii=False)}",f'Source: {source}',
           'See upstream SOURCE_AND_LICENSE.txt for dataset usage terms.']
    (out/'EXTERNAL_VALIDATION_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name in ('manifest.json','SOURCE_AND_LICENSE.txt'):
        if (root/name).is_file(): shutil.copy2(root/name,out/name)
    # Package the actual deduplicated evaluated GT, not the stale duplicated CSV.
    gt_cols=[c for c in df if c not in ('occupied_pred','correct','detections_in_image')]
    df[gt_cols].to_csv(out/'external_gt_spots.csv',index=False,encoding='utf-8-sig')
    assets={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='external_artifact_hashes.json'}
    (out/'external_artifact_hashes.json').write_text(json.dumps(assets,indent=2)+'\n',encoding='utf-8')
    package=out/'EXTERNAL_VALIDATION_TO_CHATGPT.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p!=package: z.write(p,p.relative_to(out).as_posix())
    return {'overall':overall,'count_metrics':cm,'deduplication':audit,'source_duplicate_rows_removed':summary['source_duplicate_rows_removed'],
            'screenshots':screen,'output_dir':str(out),'samples':overall['N'],'package':str(package)}
