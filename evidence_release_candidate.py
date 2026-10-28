"""Research-only uncertainty-preserving exit policy; SAFE remains unchanged.

Positive empty evidence must be validated before deployment. Research inputs may
be uncalibrated proxies and must be labeled as such. A detector miss alone can
never establish that a space is empty.
"""
import pandas as pd


def run_candidate(baseline_global, observations, exit_evidence=None):
    required={'time_sec','global_id','state'}
    if not required.issubset(baseline_global.columns): raise ValueError('Missing baseline columns')
    if baseline_global.duplicated(['time_sec','global_id']).any(): raise ValueError('Duplicate global states')
    required_obs={'time_sec','global_id','local_id','cctv','full_detected','full_det_conf'}
    if not required_obs.issubset(observations.columns): raise ValueError('Missing observations')
    if observations.duplicated(['time_sec','local_id']).any(): raise ValueError('Duplicate local observations')
    if exit_evidence is None:
        exit_evidence=pd.DataFrame(columns=['time_sec','global_id','outward_motion','empty_visible'])
    if not {'time_sec','global_id','outward_motion','empty_visible'}.issubset(exit_evidence.columns): raise ValueError('Missing exit evidence')
    if exit_evidence.duplicated(['time_sec','global_id']).any(): raise ValueError('Duplicate exit evidence')
    def as_flag(v):
        if isinstance(v,(bool,int)) and v in (0,1):return bool(v)
        raise ValueError('Exit evidence flags must be booleans or 0/1')
    cues={(float(r.time_sec),str(r.global_id)):(as_flag(r.outward_motion),as_flag(r.empty_visible)) for r in exit_evidence.itertuples()}
    groups={(float(t),str(g)):d for (t,g),d in observations.groupby(['time_sec','global_id'])}
    previous={};rows=[]
    for row in baseline_global.sort_values(['time_sec','global_id']).itertuples(index=False):
        time=float(row.time_sec);gid=str(row.global_id);base=str(row.state)
        if base not in {'OCCUPIED','EMPTY','UNKNOWN'}: raise ValueError('Invalid baseline state')
        key=(time,gid)
        if key not in groups:raise ValueError('Baseline/observation time and global IDs do not align')
        local=groups[key]
        full_observed=bool(((local.full_detected==1)&(local.full_det_conf>=.25)).any())
        if {'aux_crop_detected','aux_crop_det_conf','aux_support_scales'}.issubset(local.columns):
            aux_observed=bool(((local.aux_crop_detected==1)&((local.aux_crop_det_conf>=.45)|
                ((local.aux_crop_det_conf>=.20)&(local.aux_support_scales>=2)))).any())
        else:aux_observed=False
        observed=full_observed or aux_observed
        motion,empty=cues.get(key,(False,False));positive_exit=motion and empty and not observed
        old=previous.get(gid)
        if old is None:state=base;reason='INITIAL_SNAPSHOT'
        elif base=='OCCUPIED':state='OCCUPIED';reason='BASELINE_OCCUPIED'
        elif old=='OCCUPIED' and observed:state='OCCUPIED';reason='RETAIN_CURRENT_FULL_OR_STRONG_AUX'
        elif positive_exit:state='EMPTY';reason='MOTION_AND_VISIBLE_EMPTY_CUE'
        elif old=='OCCUPIED' and base!='OCCUPIED':state='UNKNOWN';reason='NO_CONFIRMED_EXIT'
        elif old=='UNKNOWN':state='UNKNOWN';reason='UNRESOLVED'
        elif base=='UNKNOWN':state='UNKNOWN';reason='BASELINE_UNKNOWN'
        else:state='EMPTY';reason='PRIOR_EMPTY'
        rows.append({'time_sec':time,'global_id':gid,'state':state,'previous_state':old,'baseline_state':base,
                     'reason':reason,'motion_evidence':motion,'visible_empty_evidence':empty,
                     'occupied_detection_any_camera':observed,'full_observed':full_observed,'strong_aux_observed':aux_observed,'positive_exit':positive_exit,
                     'cameras_observed':';'.join(sorted(set(local.cctv.astype(str))))})
        previous[gid]=state
    return pd.DataFrame(rows)
