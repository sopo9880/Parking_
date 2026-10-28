"""Summarize user-reviewed clip intervals without inventing event timestamps."""
import argparse
import json
from pathlib import Path
import pandas as pd
from human_slot_review import sha


def summarize(case,label,trace):
    d=trace[(trace.global_id==case['global_id']) & trace.time_sec.between(case['clip_start'],case['clip_end'])].sort_values('time_sec')
    if d.empty:raise ValueError('No candidate observations for reviewed clip')
    empty=d[d.state=='EMPTY'];occupied=d[d.state=='OCCUPIED'];unknown=d[d.state=='UNKNOWN']
    result={'case_id':case['case_id'],'global_id':case['global_id'],'label':label,'clip_start':case['clip_start'],'clip_end':case['clip_end'],
        'samples':len(d),'empty_samples':len(empty),'unknown_samples':len(unknown),'occupied_samples':len(occupied),
        'first_empty_sec':float(empty.time_sec.min()) if len(empty) else None,
        'first_occupied_sec':float(occupied.time_sec.min()) if len(occupied) else None,
        'false_empty_in_parked_clip':bool(len(empty)) if label=='PARKED' else None,
        'interpretation':'Clip-level label only; no exact event time or departure delay established'}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--reviews',type=Path,required=True)
    p.add_argument('--candidate-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    case_data=json.loads(args.cases.read_text(encoding='utf8'));reviews=json.loads(args.reviews.read_text(encoding='utf8'))
    if reviews['cases_sha256']!=sha(args.cases) or reviews['video_sha256']!=case_data['video_sha256']:raise ValueError('Review provenance mismatch')
    manifest=json.loads((args.candidate_dir/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='completed' or manifest['video_sha256']!=case_data['video_sha256']:raise ValueError('Candidate provenance mismatch')
    results=[]
    for case in case_data['cases']:
        record=reviews['reviews'].get(case['case_id'])
        if record is None:continue
        if record.get('source')!='operator_review':raise ValueError('Invalid review source')
        trace=pd.read_csv(args.candidate_dir/f"{case['start_sec']}_candidate.csv")
        results.append(summarize(case,record['label'],trace))
    if args.output.exists():raise FileExistsError(args.output)
    pd.DataFrame(results).to_csv(args.output,index=False)


if __name__=='__main__':main()
