"""Exploratory causal candidate using an initial occupied image, NOT a validated vacancy detector."""
import math
import numpy as np
import pandas as pd
from utils import normalized_visual_diff


class InitialOccupiedCue:
    def __init__(self, initial_state, point, start):
        self.initial=initial_state;self.point=np.asarray(point,dtype=float);self.start=float(start)
        self.ref=None;self.prev_image=None;self.prev_track=None;self.last_time=None;self.motion_until=-1;self.stable=0

    def observe(self,t,crop,full_detected,confidence,track,center,aux_detected,aux_conf,aux_scales):
        t=float(t)
        if self.last_time is None and t!=self.start:raise ValueError('Must start at snapshot time')
        if self.last_time is not None and t<=self.last_time:raise ValueError('Time must increase')
        if crop is None or crop.size==0:raise ValueError('Missing crop')
        if self.last_time is not None and t-self.last_time>1.01:self.prev_track=None;self.stable=0
        self.last_time=t
        if t==self.start and self.initial=='OCCUPIED':self.ref=crop.copy()
        full=bool(full_detected and confidence>=.25)
        aux=bool(aux_detected and (aux_conf>=.45 or (aux_conf>=.2 and aux_scales>=2)))
        if full and center is not None and all(math.isfinite(float(v)) for v in center):
            dist=float(np.linalg.norm(np.asarray(center,dtype=float)-self.point))
            if self.prev_track and int(track)>=0 and int(track)==self.prev_track[0] and dist-self.prev_track[1]>=8:
                self.motion_until=t+20
            self.prev_track=(int(track),dist)
        else:self.prev_track=None
        diff=normalized_visual_diff(crop,self.ref) if self.ref is not None else None
        delta=normalized_visual_diff(crop,self.prev_image) if self.prev_image is not None else None
        candidate=bool(diff is not None and delta is not None and diff>=.08 and delta<=.025 and not full and not aux)
        self.stable=self.stable+1 if candidate else 0
        self.prev_image=crop.copy()
        return {'matches_initial_occupied':bool(diff is not None and diff<=.04),
                'outward_motion_recent':bool(t<=self.motion_until), 'stable_changed_view':bool(self.stable>=2),
                'changed_difference':diff,'frame_difference':delta,'full_observed':full,'strong_aux_observed':aux,
                'source':'unvalidated_initial_occupied_visual_motion_proxy'}


def run_initial_candidate(baseline,features):
    keys=['time_sec','global_id']
    if baseline.duplicated(keys).any() or features.duplicated(['time_sec','local_id']).any():raise ValueError('Duplicate observations')
    groups={(float(t),str(g)):d for (t,g),d in features.groupby(keys)}
    previous={};expected_views={};rows=[]
    for row in baseline.sort_values(keys).itertuples(index=False):
        t=float(row.time_sec);gid=str(row.global_id);base=str(row.state);k=(t,gid)
        if k not in groups:raise ValueError('Missing visual features')
        f=groups[k]; old=previous.get(gid)
        views=set(f.local_id.astype(str))
        if gid not in expected_views:expected_views[gid]=views
        elif views!=expected_views[gid]:raise ValueError('Missing or changed camera views')
        same=bool(f.matches_initial_occupied.any());positive=bool((f.full_observed | f.strong_aux_observed).any())
        departure=bool((f.outward_motion_recent & f.stable_changed_view).any())
        if old is None:state=base;reason='INITIAL_SNAPSHOT'
        elif base=='OCCUPIED':state='OCCUPIED';reason='BASELINE_OCCUPIED'
        elif departure and not positive and not same:state='EMPTY';reason='EXPLORATORY_MOTION_CHANGED_VIEW'
        elif departure and (positive or same):state='UNKNOWN';reason='CONFLICTING_VIEWS_OR_DETECTIONS'
        elif old in ('OCCUPIED','UNKNOWN') and (same or (old=='OCCUPIED' and positive)):
            state='OCCUPIED';reason='INITIAL_APPEARANCE_OR_DETECTION_SUPPORT'
        elif old in ('OCCUPIED','UNKNOWN'):state='UNKNOWN';reason='UNRESOLVED'
        elif base=='UNKNOWN':state='UNKNOWN';reason='BASELINE_UNKNOWN'
        else:state='EMPTY';reason='PRIOR_EMPTY'
        rows.append({'time_sec':t,'global_id':gid,'state':state,'baseline_state':base,'previous_state':old,'reason':reason,
                     'matches_initial_occupied':same,'occupancy_detection':positive,'motion_changed_view_cue':departure})
        previous[gid]=state
    return pd.DataFrame(rows)
