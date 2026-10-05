"""Explicit DEV/TEST membership, scoped to an isolated experiment process."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import numpy as np

ACTIVE = ContextVar('split_protocol',default=None)

def validate_protocol(protocol,horizon):
    horizon=float(horizon)
    if not math.isfinite(horizon) or horizon<=0: raise ValueError('Invalid experiment horizon')
    result={'name':str(protocol.get('name','alternate'))}
    for key in ('dev','test'):
        start,end=map(float,protocol[key])
        if not all(math.isfinite(t) for t in (start,end)) or not 0<=start<end<=horizon:
            raise ValueError('DEV/TEST must be nonempty intervals within video evaluation duration')
        result[key]=[start,end]
    a,b=result['dev'],result['test']
    if max(a[0],b[0])<min(a[1],b[1]): raise ValueError('DEV and TEST intervals must not overlap')
    result['horizon_sec']=horizon
    return result

@contextmanager
def protocol_context(protocol):
    token=ACTIVE.set(protocol)
    try: yield
    finally: ACTIVE.reset(token)

def interval_mask(times,interval,horizon):
    times=np.asarray(times,dtype=float);start,end=interval
    return (times>=start)&((times<=end) if end==horizon else (times<end))

def labels(times,legacy_end=900):
    protocol=ACTIVE.get();times=np.asarray(times,dtype=float)
    if protocol is None: return np.where(times<float(legacy_end),'DEV','TEST')
    dev=interval_mask(times,protocol['dev'],protocol['horizon_sec'])
    test=interval_mask(times,protocol['test'],protocol['horizon_sec'])
    return np.where(dev,'DEV',np.where(test,'TEST','IGNORED'))

def dev_mask(times,settings):
    protocol=settings.get('active_split_protocol') or ACTIVE.get()
    if protocol: return interval_mask(times,protocol['dev'],protocol['horizon_sec'])
    return np.asarray(times,dtype=float)<float(settings.get('dev_end_sec',900))

def learning_times(settings,sample_sec):
    protocol=settings.get('active_split_protocol') or ACTIVE.get()
    if not protocol: return np.arange(0,float(settings.get('dev_end_sec',900))+1e-6,sample_sec).tolist()
    if not math.isfinite(sample_sec) or sample_sec<=0: raise ValueError('Invalid learning sample interval')
    start,end=protocol['dev'];times=np.arange(start,end+1e-6,sample_sec)
    return times[interval_mask(times,protocol['dev'],protocol['horizon_sec'])].tolist()

def calibration_quality_rows(rows, evidence, settings):
    if not (settings.get('active_split_protocol') or ACTIVE.get()):
        return rows
    return [row for row in rows if dev_mask([evidence.at[row['idx'],'time_sec']],settings)[0]]
