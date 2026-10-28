"""Operator/per-slot-GT snapshots. Count GT cannot initialize individual slots."""
import csv
import io
import hashlib
import math
from pathlib import Path

STATES={'O':'OCCUPIED','E':'EMPTY','U':'UNKNOWN','OCCUPIED':'OCCUPIED','EMPTY':'EMPTY','UNKNOWN':'UNKNOWN'}

def read_snapshots(path, raw=None):
    if not str(path).strip(): raise ValueError('Select a snapshot file or folder')
    path=Path(path)
    if path.is_dir() and raw is None:
        files=sorted(path.glob('*.csv'))
        if not files: raise ValueError('No snapshot CSV files in directory')
        rows=[]; times=set()
        for item in files:
            part,part_times=read_snapshots(item)
            rows.extend(part);times.update(part_times)
        return rows,sorted(times)
    with (io.StringIO(raw.decode('utf-8-sig')) if raw is not None else path.open(encoding='utf-8-sig',newline='')) as f:
        reader=csv.DictReader(f)
        if not {'time_sec','global_slot_id','initial_state'}.issubset(reader.fieldnames or []):
            raise ValueError('Use slot snapshots: time_sec,global_slot_id,initial_state. Count-only GT is not supported.')
        rows=list(reader)
    times=[]
    for row in rows:
        t=float(row['time_sec'])
        if not math.isfinite(t) or t<0: raise ValueError('Invalid snapshot time')
        times.append(t)
    return rows,sorted(set(times))

def read_snapshot_source(path):
    """Read a file or directory once and hash exactly the bytes used."""
    if not str(path).strip(): raise ValueError('Select a snapshot file or folder')
    path=Path(path)
    files=sorted(path.glob('*.csv')) if path.is_dir() else [path]
    if not files: raise ValueError('No snapshot CSV files in directory')
    rows=[];times=set();h=hashlib.sha256()
    for file in files:
        raw=file.read_bytes();part,part_times=read_snapshots(file,raw=raw)
        rows.extend(part);times.update(part_times)
        if len(files)==1 and not path.is_dir(): return rows,sorted(times),hashlib.sha256(raw).hexdigest()
        h.update(file.name.encode('utf-8'));h.update(b'\0');h.update(hashlib.sha256(raw).digest())
    return rows,sorted(times),h.hexdigest()

def load_snapshot(path,start,global_ids,rows=None):
    if rows is None: rows,_=read_snapshots(path)
    selected=[r for r in rows if float(r['time_sec'])==float(start)]
    if not selected: return None
    result={str(g):'UNKNOWN' for g in global_ids}
    seen=set()
    for row in selected:
        gid=row['global_slot_id'].strip()
        if gid not in result or gid in seen: raise ValueError('Unknown or duplicate global slot at selected snapshot: '+gid)
        value=row['initial_state'].strip().upper()
        if value not in STATES: raise ValueError('Use O/E/U or OCCUPIED/EMPTY/UNKNOWN')
        result[gid]=STATES[value];seen.add(gid)
    return result

def write_snapshot(path,start,states):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time_sec','global_slot_id','initial_state'])
        for gid,state in sorted(states.items()): writer.writerow([start,gid,STATES[state.upper()]])

def snapshot_audit(path,start,states,source,source_sha256=None):
    return {'source':source,'source_sha256':source_sha256 or hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'snapshot_time_sec':start,'states':states,'future_gt_used_for_inference':False,
        'used_for_dev_selection':False,'runtime_history_reused':False,
        'occupied':list(states.values()).count('OCCUPIED'),'empty':list(states.values()).count('EMPTY'),
        'unknown':list(states.values()).count('UNKNOWN')}

def apply_snapshot(evidence,states):
    data=evidence.copy()
    unknown=set(data.global_id.astype(str))-set(states)
    if unknown: raise ValueError('Snapshot/evidence global IDs differ')
    data['initial_state']=data.global_id.astype(str).map(states)
    return data

def occupancy_bounds(global_trace):
    data=global_trace.assign(occupied=global_trace.state.eq('OCCUPIED').astype(int),unknown=global_trace.state.eq('UNKNOWN').astype(int))
    result=data.groupby('time_sec',as_index=False)[['occupied','unknown']].sum()
    result=result.rename(columns={'occupied':'confirmed_occupied','unknown':'unknown_slots'})
    result['possible_occupied']=result.confirmed_occupied+result.unknown_slots
    return result
