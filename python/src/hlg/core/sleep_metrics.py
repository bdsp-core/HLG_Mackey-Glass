"""
Clinical sleep-disordered-breathing severity metrics.

This module computes the three canonical indices used to quantify the severity
of sleep-disordered breathing (SDB) from a scored polysomnography recording:

* **RDI** (Respiratory Disturbance Index) -- total respiratory events per hour
  of sleep.  Includes apneas, hypopneas, and respiratory-effort-related
  arousals (RERAs).
* **AHI** (Apnoea-Hypopnoea Index) -- apneas and hypopneas only (labels 1-4
  and 7, with 7 remapped to 4), excluding RERAs and central-only events.
  This is the primary metric used for OSA severity grading (mild >= 5,
  moderate >= 15, severe >= 30).
* **CAI** (Central Apnoea Index) -- central apneas only (label 2, remapped to
  8 for isolation).  A CAI >= 5 is the diagnostic threshold for central sleep
  apnea and treatment-emergent central sleep apnea.

All indices are expressed as **events per hour of sleep** and are rounded to
two decimal places for clinical reporting.

Label encoding (AASM-derived):
    0 = no event, 1 = obstructive apnea, 2 = central apnea,
    3 = mixed apnea, 4 = hypopnea, 5 = RERA (?), 7 = central hypopnea.

Sleep staging encoding:
    0 = wake (or artefact), 1-4 = NREM stages, 5 = wake stage marker.

Local dependency: ``hlg.core.events.find_events`` for event counting.
"""

import numpy as np

from hlg.core.events import find_events


def compute_sleep_metrics(
    resp: np.ndarray,
    stage: np.ndarray,
    exclude_wake: bool = True,
) -> tuple[float, float, float, float]:
    """Compute RDI, AHI, and CAI from respiratory and staging arrays.

    The function first determines total sleep time (TST) by counting samples
    where the patient is asleep (stage 1-4).  If TST is zero, all indices
    are returned as zero.

    Event counting uses ``find_events`` on the masked respiratory array,
    which groups contiguous non-zero samples into discrete events.  This
    matches the clinical convention of counting each apnea/hypopnea as a
    single event regardless of its duration.

    Args:
        resp: 1-D array of respiratory event labels at the analysis sample
            rate (typically 10 Hz, i.e. 36 000 samples per hour).
        stage: 1-D array of sleep stage labels at the same sample rate.
            Stages 1-4 are considered sleep; 0 and 5 are wake/artefact.
        exclude_wake: If ``True`` (default), only events occurring during
            sleep are counted and TST is used as the denominator.  If
            ``False``, all events are counted and total recording time is
            used (non-standard, included for research flexibility).

    Returns:
        A 4-tuple ``(RDI, AHI, CAI, sleep_time_hours)`` where each index is
        rounded to 2 decimal places and sleep time is in hours.
    """
    resp, stage = np.array(resp), np.array(stage)

    # Replace NaN / Inf in staging with 0 (wake) to avoid masking errors.
    stage[~np.isfinite(stage)] = 0

    # Sleep mask: NREM stages 1-4 are "asleep" (stage < 5 and stage > 0).
    # Stage 5 in this encoding is wake; stage 0 is unscored / artefact.
    patient_asleep = np.logical_and(stage < 5, stage > 0)

    # Total sleep time in hours.  At 10 Hz, one hour = 36 000 samples.
    sleep_time = np.sum(patient_asleep == 1) / 36000

    if sleep_time == 0:
        return 0, 0, 0, 0

    # ------------------------------------------------------------------
    # RDI -- all respiratory events during sleep
    # ------------------------------------------------------------------
    # Any non-zero label is counted as a respiratory disturbance.
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    metric = len(find_events(vals > 0)) if exclude_wake else np.sum(vals > 0)
    RDI = round(metric / sleep_time, 2)

    # ------------------------------------------------------------------
    # AHI - apneas (1, 3, 4) + hypopneas + central hypopneas (7 -> 4)
    # ------------------------------------------------------------------
    # Remap label 7 (central hypopnea) to 4 (generic hypopnea) so it is
    # included in the AHI count, then zero out any label > 4 (RERAs,
    # central-only markers) and any negative artefact.
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    vals[vals == 7] = 4
    vals[vals > 4] = 0
    vals[vals < 0] = 0
    metric = len(find_events(vals > 0)) if exclude_wake else np.sum(vals > 0)
    AHI = round(metric / sleep_time, 2)

    # ------------------------------------------------------------------
    # CAI -- central apneas only
    # ------------------------------------------------------------------
    # Isolate label 2 (central apnea) by remapping it to 8, then zeroing
    # everything <= 7.  This keeps only the remapped central events.
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    vals[vals == 2] = 8
    vals[vals <= 7] = 0
    metric = len(find_events(vals > 0)) if exclude_wake else np.sum(vals > 0)
    CAI = round(metric / sleep_time, 2)

    return RDI, AHI, CAI, round(sleep_time, 2)
