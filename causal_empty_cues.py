"""Uncalibrated research proxy: compare current crop to causal empty/occupied references.

This is NOT a validated classifier of a clear parking space. Only an operator's
initial EMPTY snapshot can establish the empty reference. Outputs are proposals.
"""
import math
import numpy as np
from utils import normalized_visual_diff


class CausalEmptyCue:
    def __init__(self, initial_state, point, start):
        self.initial=initial_state;self.point=point;self.start=float(start);self.last_time=None
        self.empty_ref=None;self.occupied_ref=None;self.hit_streak=0;self.empty_streak=0
        self.prev_track=None;self.motion_until=-1

    def observe(self,t,crop,full_detected,confidence,track,center,baseline_state):
        t=float(t)
        if self.last_time is not None and t<=self.last_time:raise ValueError('Observations must be chronological')
        if self.last_time is None and t!=self.start:raise ValueError('Must start at exact snapshot time')
        if crop is None or crop.size==0:raise ValueError('Missing crop')
        if self.last_time is not None and t-self.last_time>1.01:self.hit_streak=0;self.empty_streak=0;self.prev_track=None
        self.last_time=t
        hit=bool(full_detected and confidence>=.25)
        if t==self.start and self.initial=='EMPTY' and not hit:self.empty_ref=crop.copy()
        if hit and center is not None and all(math.isfinite(float(x)) for x in center):
            distance=float(np.linalg.norm(np.asarray(center,dtype=float)-np.asarray(self.point,dtype=float)))
            if self.prev_track and int(track)>=0 and self.prev_track[0]==int(track) and distance-self.prev_track[1]>=6:
                self.motion_until=t+15
            self.prev_track=(int(track),distance)
        else:self.prev_track=None
        self.hit_streak=self.hit_streak+1 if hit and baseline_state=='OCCUPIED' else 0
        if self.empty_ref is not None and self.occupied_ref is None and self.hit_streak>=3:self.occupied_ref=crop.copy()
        d_empty=normalized_visual_diff(crop,self.empty_ref) if self.empty_ref is not None else None
        d_occupied=normalized_visual_diff(crop,self.occupied_ref) if self.occupied_ref is not None else None
        candidate=(d_empty is not None and d_occupied is not None and d_empty<=.035 and d_occupied>=.06 and not hit)
        self.empty_streak=self.empty_streak+1 if candidate else 0
        motion=t<=self.motion_until
        return {'outward_motion':motion,'empty_visible':self.empty_streak>=3,'empty_ref_difference':d_empty,
                'occupied_ref_difference':d_occupied,'empty_streak':self.empty_streak,'occupied_reference_available':self.occupied_ref is not None,
                'source':'uncalibrated_initial_empty_crop_proxy_not_verified_gt'}
