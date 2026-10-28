"""Evaluate the research exit policy on prior human-initialized GPU traces."""
import argparse
import json
from pathlib import Path
import pandas as pd
from evidence_release_candidate import run_candidate
from human_slot_review import sha
from manual_initialization import occupancy_bounds
from utils import load_ground_truth


def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh',required=True,type=Path);p.add_argument('--comparison',required=True,type=Path)
    p.add_argument('--gt',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--cues',type=Path);args=p.parse_args()
    fresh=json.loads((args.fresh/'manifest.json').read_text(encoding='utf-8'))
    comp=json.loads((args.comparison/'manifest.json').read_text(encoding='utf-8'))
    if fresh['status']!='completed' or comp['status']!='completed' or comp['fresh_manifest_sha256']!=sha(args.fresh/'manifest.json'):
        raise ValueError('Fresh/comparison provenance mismatch')
    start=int(fresh['start_sec']);source=Path(fresh['result'])/f'restart_{start}'/'evidence'/'segmentation_evidence.csv'
    if sha(source)!=comp['evidence_sha256']:raise ValueError('Saved observations changed')
    if sha(args.gt)!=comp['gt_sha256']:raise ValueError('Count GT changed')
    obs=pd.read_csv(source,low_memory=False);baseline=pd.read_csv(args.comparison/'HUMAN_MANUAL_INIT_global.csv')
    if args.cues:
        cue_manifest=json.loads((args.cues/'manifest.json').read_text(encoding='utf-8'))
        if cue_manifest['status']!='completed' or cue_manifest['evidence_sha256']!=sha(source) or cue_manifest['comparison_manifest_sha256']!=sha(args.comparison/'manifest.json'):
            raise ValueError('Cue provenance mismatch')
        cues=pd.read_csv(args.cues/'exit_cues.csv')
    else:cues=None
    candidate=run_candidate(baseline,obs,cues)
    args.output.mkdir(parents=True,exist_ok=False)
    candidate.to_csv(args.output/'candidate_global.csv',index=False)
    gt=load_ground_truth(args.gt); end=min(1230,start+330)
    window=gt[(gt.time_sec>=start)&((gt.time_sec<=end) if end==1230 else (gt.time_sec<end))]
    b=occupancy_bounds(candidate).merge(window[['time_sec','ground_truth_occupied_space_count']],on='time_sec',validate='one_to_one')
    b['gt_within_bounds']=(b.ground_truth_occupied_space_count>=b.confirmed_occupied)&(b.ground_truth_occupied_space_count<=b.possible_occupied)
    b.to_csv(args.output/'candidate_bounds.csv',index=False)
    rows=[]
    for name,d in [('initial',b[b.time_sec==start]),('after_initial',b[b.time_sec>start]),
                   *[(f'first_{n}s',b[(b.time_sec>start)&(b.time_sec<start+n)]) for n in (30,60,120)]]:
        resolved=d[d.unknown_slots==0];err=resolved.confirmed_occupied-resolved.ground_truth_occupied_space_count
        rows.append({'scope':name,'N':len(d),'fully_resolved_N':len(resolved),'unknown_mean':float(d.unknown_slots.mean()) if len(d) else None,
                     'gt_in_bounds_rate':float(d.gt_within_bounds.mean()) if len(d) else None,
                     'exact_all':float(err.eq(0).mean()) if len(d) and len(resolved)==len(d) else None,
                     'mae_all':float(err.abs().mean()) if len(d) and len(resolved)==len(d) else None,
                     'exact_resolved_subset':float(err.eq(0).mean()) if len(resolved) else None})
    pd.DataFrame(rows).to_csv(args.output/'candidate_metrics.csv',index=False)
    manifest={'status':'completed','mode':'research_only_unknown_on_unsupported_clear','start_sec':start,
              'code_sha256':sha(Path(__file__).with_name('evidence_release_candidate.py')),
              'observation_sha256':sha(source),'human_snapshot_sha256':comp['human_snapshot_sha256'],
              'gt_sha256':sha(args.gt),'exit_evidence_source':'uncalibrated_initial_empty_crop_proxy' if args.cues else 'none: no verified empty or direction labels supplied',
              'cues_sha256':sha(args.cues/'exit_cues.csv') if args.cues else None,
              'scope':'same historically reviewed video; extra human initialization; exploratory; not independent validation',
              'warning':'Cues are uncalibrated proposals or absent; they are not verified slot labels and cannot establish actual exit-detection accuracy.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
