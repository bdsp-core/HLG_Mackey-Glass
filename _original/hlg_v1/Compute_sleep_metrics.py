import sys, os
import numpy as np

# import utils functions
from Event_array_modifiers import find_events

# metric computation fuctions
def compute_sleep_metrics(resp, stage, exclude_wake=True):
    resp, stage = np.array(resp), np.array(stage)
    
    # compute sleep time
    stage[~np.isfinite(stage)] = 0
    patient_asleep = np.logical_and(stage<5, stage>0)
    sleep_time = np.sum(patient_asleep==1) / 36000
    if sleep_time==0:
        return 0, 0, 0, 0

    # compute RDI
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    metric = len(find_events(vals>0)) if exclude_wake else np.sum(vals>0)
    RDI = round(metric / sleep_time, 2)

    # compute AHI
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    vals[vals==7] = 4
    vals[vals>4] = 0
    vals[vals<0] = 0
    metric = len(find_events(vals>0)) if exclude_wake else np.sum(vals>0)
    AHI = round(metric / sleep_time, 2)

    # compute CAI
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    vals[vals==2] = 8
    vals[vals<=7] = 0
    metric = len(find_events(vals>0)) if exclude_wake else np.sum(vals>0)
    CAI = round(metric / sleep_time, 2)

    return RDI, AHI, CAI, round(sleep_time, 2)

