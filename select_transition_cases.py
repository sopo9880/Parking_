"""Select a small set of unverified DEV review cases, without reading GT."""
import argparse
import json
from pathlib import Path
import pandas as pd
from human_slot_review import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--extras-only',action='store_true');args=p.parse_args()
    manifest=json.loads((args.dev/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='completed':raise ValueError('DEV analysis incomplete')
    runs=sorted((args.fit/'restart_experiments').glob('run_*'));run=runs[-1]
    cases=[];extras=[];used=set()
    for start,end in [(330,660),(660,900)]:
        trace=pd.read_csv(args.dev/f'{start}_candidate.csv').sort_values(['global_id','time_sec'])
        trace['previous']=trace.groupby('global_id').state.shift()
        changes=trace[(trace.previous.notna())&(trace.state!=trace.previous)]
        for previous,current,kind in [('OCCUPIED','UNKNOWN','unconfirmed_release'),('UNKNOWN','OCCUPIED','possible_entry')]:
            options=changes[(changes.previous==previous)&(changes.state==current)].sort_values(['time_sec','global_id'])
            if len(options):
                row=options.iloc[0];gid=str(row.global_id);t=int(row.time_sec);used.add((start,gid,t))
                cases.append({'start_sec':start,'event_sec':t,'global_id':gid,'kind':kind,'ai_state_before':previous,'ai_state_after':current})
        source=run/f'restart_{start}'/'evidence'/'segmentation_evidence.csv'
        if sha(source)!=manifest['inputs'][str(start)]['evidence_sha256']:raise ValueError('DEV evidence changed')
        evidence=pd.read_csv(source,low_memory=False)
        motion=evidence[(evidence.time_sec>=start+10)&(evidence.time_sec<end-10)&(evidence.full_detected==1)&
                        (evidence.full_det_conf>=.25)&(evidence.full_track_speed_px_s>8)&(evidence.full_track_motion_span_px>25)&
                        (~evidence.global_id.isin(['G018','G021']))]
        if len(motion):
            row=motion.sort_values(['full_track_speed_px_s','time_sec'],ascending=[False,True]).iloc[0]
            cases.append({'start_sec':start,'event_sec':int(row.time_sec),'global_id':str(row.global_id),'kind':'moving_detection_unverified',
                          'local_id':str(row.local_id),'ai_state_before':'not inferred','ai_state_after':'motion candidate only'})
        for gid,g in trace.groupby('global_id',sort=True):
            unknown=g[g.state=='UNKNOWN']
            if len(unknown)>=61:
                samples=unknown.time_sec.to_numpy()
                for i in range(len(samples)-60):
                    if samples[i+60]-samples[i]==60:
                        t=int(samples[i+30]);extras.append({'start_sec':start,'event_sec':t,'global_id':str(gid),'kind':'long_unknown',
                            'ai_state_before':'UNKNOWN','ai_state_after':'UNKNOWN'});break
                else:continue
                break
        ev=evidence[(evidence.time_sec>=start+30)&(evidence.time_sec<end-10)].copy()
        ev['hit']=(ev.full_detected==1)&(ev.full_det_conf>=.25)
        grouped=ev.groupby(['time_sec','global_id']).agg(cameras=('cctv','nunique'),hits=('hit','sum')).reset_index()
        conflicts=grouped[(grouped.cameras>=2)&(grouped.hits>0)&(grouped.hits<grouped.cameras)].sort_values(['time_sec','global_id'])
        if len(conflicts):
            row=conflicts.iloc[0];extras.append({'start_sec':start,'event_sec':int(row.time_sec),'global_id':str(row.global_id),'kind':'camera_detection_disagreement',
                                                 'ai_state_before':'not inferred','ai_state_after':'camera observations disagree'})
    if args.extras_only:cases=extras
    else:cases.extend(extras)
    for item in cases:
        item['case_id']=f"{item['start_sec']}_{item['global_id']}_{item['event_sec']}_{item['kind']}"
        item['clip_start']=max(item['start_sec'],item['event_sec']-15);item['clip_end']=min(899 if item['start_sec']==660 else 659,item['event_sec']+15)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps({'status':'unreviewed_ai_selected','scope':'same historically reviewed video, DEV only; not human GT',
        'video_sha256':sha(args.video),'fit_slots_sha256':sha(args.fit/'slots.json'),'roi_source_sha256':sha(args.fit/'worker_input.json'),
        'dev_manifest_sha256':sha(args.dev/'manifest.json'),'cases':cases},ensure_ascii=False,indent=2),encoding='utf8')
    print('Selected',len(cases),'unreviewed cases')


if __name__=='__main__':main()
