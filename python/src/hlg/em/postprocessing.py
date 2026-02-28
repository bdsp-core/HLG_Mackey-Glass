"""
Post-processing of Estimation Model (EM) output.

After the EM has fit loop gain (LG), controller gain (G), and plant gain
(D) to every 8-minute sleep segment, the raw per-segment estimates need
several corrections before they are suitable for clinical interpretation:

1. **Outlier smoothing** -- ``post_process_EM_output`` applies a sliding-
   window median/mean filter that replaces spurious LG spikes (often
   caused by artifacts or brief arousals within a segment) with a
   neighbourhood average.

2. **Arousal recovery** -- ``post_process_estimated_arousals`` disentangles
   the arousal component from the modelled ventilatory drive, producing a
   "corrected" ventilation estimate that can be compared against the raw
   observed signal.

3. **Metadata linkage** -- ``match_EM_with_SS_output`` and ``add_arousals``
   link the EM CSV back to the Self-Similarity (SS) HDF5 output so that
   arousal annotations and patient demographics can be merged.

4. **Trailing-wake removal** -- ``remove_excessive_wake`` trims long
   post-sleep wake epochs that inflate recording duration without
   contributing useful respiratory data.

Source functions: ``EM_output_to_Figures.py`` (data-processing subset).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hlg.io.readers import load_sim_output


# ---------------------------------------------------------------------------
# SS <-> EM linkage helpers
# ---------------------------------------------------------------------------


def match_EM_with_SS_output(
    data: pd.DataFrame,
    dataset: str,
    csv_file: str,
) -> tuple[str, pd.DataFrame]:
    """Locate the SS pipeline HDF5 output that corresponds to an EM recording.

    The EM and SS pipelines produce separate output files.  This function
    reads the cohort-level CSV index (``csv_file``) and uses the patient
    tag embedded in ``data`` to find the matching SS path.  Different
    datasets store the patient identifier in different CSV columns, so
    the lookup logic branches by ``dataset``.

    Args:
        data: EM output DataFrame.  Must contain a ``patient_tag`` column
            whose first value is the recording identifier.
        dataset: Dataset identifier string -- one of ``'mgh'``,
            ``'redeker'``, ``'rt'``, or ``'bdsp'``.  Controls which
            CSV column and matching strategy is used.
        csv_file: Path to the cohort-level CSV metadata file that maps
            patient tags to SS output paths.

    Returns:
        A 2-tuple ``(sim_path, sim_df)`` where:

        * **sim_path** -- the SS output path stem (without ``.hf5``
          extension) for this recording.
        * **sim_df** -- the full cohort DataFrame loaded from ``csv_file``.

    Raises:
        AssertionError: If zero or more than one matching path is found.
    """
    sim_df: pd.DataFrame = pd.read_csv(csv_file)
    tag = data.patient_tag[0]

    # Each dataset stores the patient identifier in a different column
    # and may embed it differently in the filename.
    if dataset == "mgh":
        sim_path: list[str] = [p for p in sim_df.SS_path if tag in p]
    elif dataset == "redeker":
        sim_path = [p for p in sim_df.SS_path if str(tag) in p]
    elif dataset == "rt":
        # RT dataset appends '.hf5' to the patient_num column; strip it
        # before matching so the resulting path can be re-suffixed later.
        sim_path = [p.split(".hf5")[0] for p in sim_df.patient_num if tag in p]
    elif dataset == "bdsp":
        sim_path = [p for p in sim_df.subjectID if tag in p]

    assert len(sim_path) == 1, f".hf5 file not found for patient {tag}"
    return sim_path[0], sim_df


def add_arousals(
    data: pd.DataFrame,
    version: str,
    dataset: str,
    hf5_folder: str,
    csv_file: str,
) -> tuple[pd.DataFrame, str]:
    """Merge arousal annotations from the SS output into the EM DataFrame.

    The EM CSV does not natively contain arousal labels -- those live in
    the SS HDF5 output.  This function reads them and attaches an
    ``Arousals`` column to ``data``.  It also returns the clinical
    "SS group" label (e.g. CPAP success/failure) from the cohort CSV.

    For simulation runs (where ``version`` contains ``'Simulation'``)
    arousals are not loaded because the simulated data already has
    deterministic arousal timing.

    Args:
        data: EM output DataFrame to augment.
        version: Pipeline version string.  If it contains ``'Simulation'``
            the arousal merge is skipped.
        dataset: Dataset identifier (``'mgh'``, ``'redeker'``, etc.).
        hf5_folder: Directory containing the SS ``.hf5`` files (with
            trailing separator).
        csv_file: Path to the cohort-level CSV metadata file.

    Returns:
        A 2-tuple ``(data, group)`` where ``data`` now has an
        ``Arousals`` column and ``group`` is the string SS group label.

    Raises:
        AssertionError: If the arousal array length does not match the
            EM data length (within a 1 % tolerance for minor resampling
            differences).
    """
    sim_path, sim_df = match_EM_with_SS_output(data, dataset, csv_file)
    path: str = hf5_folder + sim_path + ".hf5"

    if "Simulation" not in version:
        # Load only the arousal channel from the SS output.
        arousals, _ = load_sim_output(path, ["arousal"])

        # Sanity check: the two files should represent the same
        # recording duration (allow 1 % slack for rounding).
        assert len(arousals) > 0.99 * len(data), "Something is going wrong when adding arousals!"

        data.loc[: len(arousals) - 1, "Arousals"] = arousals.values

    # Retrieve the clinical grouping (e.g. CPAP responder vs.
    # non-responder) from the cohort metadata CSV.
    group: str = sim_df.loc[np.where(sim_df.SS_path == sim_path), "SS group"].values[0]

    return data, group


# ---------------------------------------------------------------------------
# Patient demographics extraction
# ---------------------------------------------------------------------------


def extract_patient_metrics(
    hdr: dict[str, Any],
    dataset: str,
    csv_file: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Enrich the recording header with patient demographics from the cohort CSV.

    Different datasets store demographic and clinical index columns under
    different names.  This function maps dataset-specific column names to
    a canonical set (Sex, Age, AHI, CAI, etc.) and writes them into
    ``hdr``.

    The RT dataset has no external demographics CSV, so this function is
    a no-op for that dataset.

    Args:
        hdr: Mutable header dict for the current recording.  Will be
            augmented in-place with demographic fields.
        dataset: Dataset identifier (``'mgh'``, ``'bdsp'``, ``'redeker'``,
            or ``'rt'``).
        csv_file: Path to the cohort-level CSV metadata file.

    Returns:
        A 2-tuple ``(hdr, metric_map)`` where:

        * **hdr** -- the enriched header dict.
        * **metric_map** -- a dict mapping CSV column names to canonical
          names (empty dict for ``'rt'``).
    """
    if dataset == "rt":
        return hdr, {}

    sim_df: pd.DataFrame = pd.read_csv(csv_file)
    tag = hdr["patient_tag"]

    # Locate the row in the cohort CSV that matches this patient.
    # The lookup column differs by dataset.
    if dataset in ["mgh", "bdsp"]:
        ind: int = np.where(sim_df.SS_path == tag)[0][0]
        # Canonical mapping: CSV column -> header key.
        # AHI and sub-indices are the standard AASM definitions:
        #   OAI = obstructive apnea index, CAI = central apnea index,
        #   MAI = mixed apnea index, HI = hypopnea index.
        metric_map: dict[str, str] = {
            "Sex": "Sex",
            "age": "Age",
            "ahi_3%": "AHI",
            "Obs_i": "OAI",
            "cai_3%": "CAI",
            "Mix_i": "MAI",
            "Hyp_i": "HI",
        }
    elif dataset == "redeker":
        # Redeker IDs are numeric; match on the first 4 characters.
        ind = np.where(sim_df.ID.astype(str) == str(tag)[:4])[0][0]
        metric_map = {"Sex": "Sex", "age": "Age", "ahi_3%": "AHI", "cai": "CAI"}

        # Redeker encodes sex as 1/2; remap to human-readable strings.
        for i in range(len(sim_df)):
            sim_df.loc[i, "Sex"] = {1: "Male", 2: "Female"}[sim_df.loc[i, "Sex"]]

    # Transfer each metric from the cohort CSV into the header dict,
    # applying appropriate rounding: ages are integer, indices are
    # rounded to one decimal place.
    for metric in metric_map.keys():
        if isinstance(sim_df.loc[ind, metric], str):
            hdr[metric_map[metric]] = sim_df.loc[ind, metric]
        else:
            if metric == "age":
                hdr[metric_map[metric]] = sim_df.loc[ind, metric].astype(int)
            else:
                hdr[metric_map[metric]] = sim_df.loc[ind, metric].round(1)

    return hdr, metric_map


# ---------------------------------------------------------------------------
# Arousal disentanglement
# ---------------------------------------------------------------------------


def post_process_estimated_arousals(
    data: pd.DataFrame,
    arousal_dur: float,
) -> pd.DataFrame:
    """Separate arousal contributions from the modelled ventilatory drive.

    The EM produces two parallel ventilation estimates (tagged '1' and '2')
    that include an arousal component.  Only one of these is valid (the
    other starts at zero).  This function:

    1. Selects the valid estimate by checking which tag has a non-zero
       initial value.
    2. Subtracts the arousal component to recover the "drive-only"
       ventilation (``Vd_est``).
    3. Computes an "unscaled" arousal by removing the scaling difference
       between the raw and scaled ventilation estimates.  This isolates
       the true physiological arousal response from the model's amplitude
       scaling.
    4. Adds the unscaled arousal back to produce a "corrected" observed
       ventilation (``Vo_est_corrected``) suitable for error analysis.

    Args:
        data: EM output DataFrame containing columns ``Vo_est_scaled1``,
            ``Vo_est_scaled2``, ``Vo_est1``, ``Vo_est2``, ``Arousal1``,
            ``Arousal2``.
        arousal_dur: Expected arousal duration in seconds (used for
            context; not consumed in this function but kept for API
            compatibility with the original script).

    Returns:
        The input DataFrame augmented with:

        * ``Vd_est_scaled`` -- scaled ventilatory drive (arousal removed).
        * ``Vd_est`` -- unscaled ventilatory drive.
        * ``Aest_loc`` -- boolean mask of arousal locations.
        * ``Arousal_unscaled`` -- physiological arousal magnitude.
        * ``Vo_est_corrected`` -- corrected observed ventilation.
    """
    # Determine which of the two parallel estimates is the valid one.
    # The invalid estimate starts at exactly zero (within floating-point
    # tolerance of 5 decimal places).
    tag: str = "1" if data.loc[0, "Vo_est_scaled1"].round(5) == 0 else "2"

    # Subtract the arousal component to isolate pure ventilatory drive.
    data["Vd_est_scaled"] = data["Vo_est_scaled" + tag] - data["Arousal" + tag]
    data["Vd_est"] = data["Vo_est" + tag] - data["Arousal" + tag]

    # Boolean mask where arousals are absent (used to zero-out the
    # scaling correction at non-arousal locations).
    locs: pd.Series = data["Arousal" + tag] == 0
    data["Aest_loc"] = data["Arousal" + tag] > 0

    # The difference between scaled and unscaled drive estimates captures
    # the amplitude scaling applied by the model.  At non-arousal
    # locations this difference is meaningless, so zero it out.
    Vo_diff: pd.Series = data["Vd_est_scaled"] - data["Vd_est"]
    Vo_diff[locs] = 0

    # Remove the scaling offset from the arousal to get the physiological
    # arousal magnitude.  Clamp negative values to zero (can occur from
    # numerical noise when the scaling barely exceeds the arousal).
    data["Arousal_unscaled"] = data["Arousal" + tag] - Vo_diff
    data.loc[data["Arousal_unscaled"] < 0, "Arousal_unscaled"] = 0

    # Final corrected ventilation: drive + physiological arousal.
    data["Vo_est_corrected"] = data["Vd_est_scaled"] + data["Arousal_unscaled"]

    return data


# ---------------------------------------------------------------------------
# LG outlier smoothing
# ---------------------------------------------------------------------------


def post_process_EM_output(
    data: pd.DataFrame,
    thresh: float = 0.8,
) -> pd.DataFrame:
    """Apply sliding-window outlier correction to per-segment loop gain estimates.

    Raw per-segment LG values can exhibit isolated spikes caused by
    within-segment artifacts or brief arousals that confuse the model
    fit.  This function detects such outliers using a hybrid
    median-and-mean criterion over a 5-segment sliding window and
    replaces them with the neighbourhood average.

    The algorithm works as follows for each interior segment ``k``:

    1. Gather the 4 nearest neighbours: [k-2, k-1, k+1, k+2].
    2. Compute neighbourhood median and mean (ignoring NaN).
    3. If LG[k] exceeds **both** the median and the mean by more than
       ``thresh``, **and** both the median and mean are below 1.0 (i.e.
       the neighbourhood is in the physiologically stable LG < 1 range),
       replace LG[k] with the mean of its neighbours.
    4. After any replacement, restart from the beginning of the window
       so that cascading outliers are re-evaluated with the updated
       values.

    The corrected values are written back into the DataFrame under the
    ``LG_{stage}_corrected`` columns.

    Args:
        data: EM output DataFrame containing columns ``nrem_starts``,
            ``nrem_ends``, ``LG_nrem``, ``rem_starts``, ``rem_ends``,
            ``LG_rem``.
        thresh: Outlier detection threshold.  A segment is flagged if
            it exceeds both the neighbourhood median and mean by at
            least this amount.  Default 0.8 was empirically tuned to
            catch artifactual spikes while preserving genuine high-LG
            segments.

    Returns:
        The input DataFrame with new columns ``LG_nrem_corrected`` and
        ``LG_rem_corrected`` populated for every segment.
    """
    # Collect start/end/LG for NREM and REM, then merge and sort by
    # chronological order so the sliding window operates across both
    # sleep stages in temporal sequence.
    starts_nrem: np.ndarray = data["nrem_starts"].dropna().values.astype(int)
    ends_nrem: np.ndarray = data["nrem_ends"].dropna().values.astype(int)
    LGs_nrem: np.ndarray = np.round(data["LG_nrem"].values[: len(starts_nrem)], 2)

    starts_rem: np.ndarray = data["rem_starts"].dropna().values.astype(int)
    ends_rem: np.ndarray = data["rem_ends"].dropna().values.astype(int)
    LGs_rem: np.ndarray = np.round(data["LG_rem"].values[: len(starts_rem)], 2)

    starts: np.ndarray = np.concatenate([starts_nrem, starts_rem])
    ends: np.ndarray = np.concatenate([ends_nrem, ends_rem])
    LGs: np.ndarray = np.concatenate([LGs_nrem, LGs_rem])

    # Sort all segments chronologically (they were concatenated
    # NREM-first, REM-second, but temporally they are interleaved).
    loc: np.ndarray = np.argsort(starts)
    starts = starts[loc]
    ends = ends[loc]
    LGs = LGs[loc]

    LGs_corrected: np.ndarray = np.array(LGs)

    # Sliding window parameters.  A window of 5 gives 2 neighbours on
    # each side, which is wide enough to establish a local baseline
    # without oversmoothing genuine within-night LG trends.
    win: int = 5
    cnt: int = win // 2
    len_: int = len(LGs[win // 2 : -win // 2])

    while cnt < len_:
        LG: float = LGs_corrected[cnt]
        window: np.ndarray = LGs_corrected[cnt - win // 2 : cnt + 1 + win // 2]

        # If the entire local window is NaN (e.g. a long wake epoch in
        # the middle of the night), nothing can be inferred.
        if all(np.isnan(window)):
            LGs_corrected[cnt] = np.nan
            cnt += 1
            continue

        # The 4 nearest neighbours (excluding the center segment itself).
        inds: np.ndarray = np.array([cnt - 2, cnt - 1, cnt + 1, cnt + 2])
        neighbors: np.ndarray = LGs_corrected[inds]
        median: float = np.nanmedian(neighbors)
        mean: float = np.nanmean(neighbors)

        # Outlier criterion: the segment's LG must exceed both the
        # median and mean by more than ``thresh``, AND the local
        # neighbourhood must be in the physiologically "stable" range
        # (LG < 1).  This prevents false positives in patients with
        # genuinely high loop gain throughout the night.
        if (LG - median) > thresh and (LG - mean) > thresh and np.logical_and(median < 1, mean < 1):
            val: float = round(np.mean(neighbors), 2)
            LGs_corrected[cnt] = val
            # Restart the sweep from the beginning so that any cascade
            # effects (adjacent outliers) are re-evaluated with the
            # newly corrected value.
            cnt = win // 2
            continue
        else:
            LGs_corrected[cnt] = LG

        cnt += 1

    # Write corrected values back into the appropriate stage-specific
    # column in the DataFrame.
    for st, _end, _LG, LG_c in zip(starts, ends, LGs, LGs_corrected):
        tag: str = "nrem"
        loc = np.where(data["nrem_starts"] == st)[0]
        if len(loc) == 0:
            tag = "rem"
            loc = np.where(data["rem_starts"] == st)[0]
        data.loc[loc[0], f"LG_{tag}_corrected"] = LG_c

    return data


# ---------------------------------------------------------------------------
# Trailing-wake trimming
# ---------------------------------------------------------------------------


def remove_excessive_wake(
    EM_data: pd.DataFrame,
    SS_data: pd.DataFrame,
    Fs: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trim trailing wake epochs that do not contribute respiratory data.

    Many PSG recordings continue for 30-60 minutes after the patient's
    final awakening.  This "trailing wake" inflates recording duration
    and can bias per-hour indices.  The function:

    1. Removes any trailing rows beyond the last scored epoch.
    2. If the recording ends in wake (stage 5) and the last sleep
       epoch is more than 30 minutes (``3600 * Fs / 2`` samples) from
       the end, truncates to 30 minutes past the last sleep sample.

    Both the EM and SS DataFrames are trimmed identically to keep them
    aligned sample-for-sample.

    Args:
        EM_data: EM output DataFrame with a ``Stage`` column.
        SS_data: Matching SS output DataFrame (same length as ``EM_data``).
        Fs: Sampling frequency in Hz.

    Returns:
        A 2-tuple ``(EM_data, SS_data)`` with trailing wake removed and
        indices reset.
    """
    SS: np.ndarray = EM_data["Stage"].values

    # Find the last sample that has a valid (finite) stage annotation.
    end: int = np.where(np.isfinite(SS))[0][-1]
    EM_data = EM_data.loc[:end, :].reset_index(drop=True)
    SS_data = SS_data.loc[:end, :].reset_index(drop=True)

    # Maximum allowed trailing wake: half an hour (3600 / 2 seconds).
    thresh: int = int(3600 * Fs / 2)

    # First and last samples where the patient is actually asleep
    # (stage < 5; stages 0-4 are wake/NREM/REM in AASM encoding,
    # where stage 0 = wake is still "scored" -- only 5 = unscored).
    _start: int = np.where(SS < 5)[0][0]
    end = np.where(SS < 5)[0][-1]

    # If the recording ends in unscored/movement (stage 5) and the
    # gap between the last scored epoch and the recording end exceeds
    # the threshold, truncate.
    if SS[-1] == 5 and end < len(SS) - thresh:
        EM_data = EM_data.loc[: end + thresh, :].reset_index(drop=True)
        SS_data = SS_data.loc[: end + thresh, :].reset_index(drop=True)

    return EM_data, SS_data
