"""
Cohort pipeline for patient selection and EM input preparation.

This module contains the batch-processing functions that were originally in
``SS_output_to_EM_input.py`` -- specifically the patient selection, file
sorting, and recording export routines.  The two analysis functions that
previously lived alongside them have been migrated to dedicated modules:

* ``load_sim_output`` → :mod:`hlg.io.readers`
* ``segment_data_based_on_nrem`` / ``compute_SS_score_per_segement``
  → :mod:`hlg.ss.segmentation`

The functions retained here orchestrate cohort-level workflows:

1. **Quality filtering** -- ``remove_bad_signal_recordings`` removes recordings
   whose IDs appear in the centrally maintained bad-signal list.
2. **Metric extraction** -- ``extract_latest_SS_outputs`` iterates over HF5
   files and populates a metadata DataFrame with respiratory indices, sleep
   time, and SS / oscillation fractions.
3. **Patient selection** -- ``patient_selection`` creates clinically meaningful
   patient subsets (SS-cases, high-CAI, HLG-OSA, REM-OSA, NREM-OSA, CPAP)
   driven by a version string.
4. **File sorting** -- ``sort_input_files`` and ``sort_altitude_files`` order
   HF5 paths according to version-specific criteria so that the exported CSV
   studies have a consistent numbering.
5. **Export** -- ``segment_and_export_recordings`` and its per-recording
   helper ``segment_and_export_recording`` convert each recording into a
   segment-level CSV suitable for the Estimation Model (EM).
"""

from __future__ import annotations

import os
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from hlg.config import config
from hlg.core.events import find_events
from hlg.core.ventilation import create_ventilation_trace
from hlg.io.readers import load_sim_output
from hlg.ss.segmentation import (
    compute_SS_score_per_segement,
    segment_data_based_on_nrem,
)


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------


def remove_bad_signal_recordings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove recordings with known bad signal quality.

    Recordings are matched by comparing the leading characters of their
    ``SS_path`` column to the canonical bad-recording list stored in
    :pyattr:`hlg.config.config.bad_recording_ids`.  The prefix length is
    determined by the length of the first ID in the bad-recording list
    (typically 7 hex characters from the truncated SHA hash used as the
    file name).

    Args:
        df: Recording metadata DataFrame.  Must contain an ``SS_path``
            column whose values start with the recording hash ID.

    Returns:
        A copy of *df* with bad-signal rows removed and the index reset.
    """
    bad_recs = list(config.bad_recording_ids)
    prefix_len = len(bad_recs[0])

    IDs = df.SS_path.values
    good_ids = [i for i, ID in enumerate(IDs) if ID[:prefix_len] not in bad_recs]
    df = df.loc[good_ids].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_latest_SS_outputs(
    sim_df: pd.DataFrame,
    input_files: list[str],
) -> pd.DataFrame:
    """Populate *sim_df* with per-recording respiratory metrics.

    For every HF5 file whose hash ID appears in ``sim_df.path_name``, the
    following header-level and derived metrics are written back into the
    corresponding row:

    * **RDI_3 %**, **AHI_3 %**, **CAI_3 %** -- respiratory disturbance,
      apnea-hypopnea, and central apnea indices at the 3 % desaturation
      threshold.
    * **sleep_time** -- total sleep time (hours) from the SS header.
    * **T_SS_new** -- fraction of sleep time during which the self-similarity
      signal is active (i.e. periodic breathing is detected).
    * **T_osc_new** -- number of tagged oscillation events per hour of sleep.

    Args:
        sim_df: Metadata DataFrame indexed by ``path_name`` (recording hash).
        input_files: List of absolute paths to ``.hf5`` output files.

    Returns:
        The updated *sim_df* (modified in place, also returned for chaining).
    """
    for p, path in enumerate(input_files):
        print(f"extracting SS output {p}/{len(input_files)} ..", end="\r")

        ID = path.split("/")[-1].split(".hf5")[0]
        if ID not in sim_df.path_name.values:
            continue

        cols = ["sleep_stages", "apnea", "self similarity", "tagged"]
        data, hdr = load_sim_output(path, cols=cols)

        loc = np.where(ID == sim_df.path_name)[0][0]

        sim_df.loc[loc, "RDI_3%"] = hdr["RDI"]
        sim_df.loc[loc, "AHI_3%"] = hdr["AHI"]
        sim_df.loc[loc, "CAI_3%"] = hdr["CAI"]
        sim_df.loc[loc, "sleep_time"] = hdr["sleep_time"]

        # T_SS: fraction of asleep time spent in self-similar (periodic
        # breathing) state.  Denominator converts sample count → hours via
        # the downsampled rate (newFs) and 3600 s/h.
        SS_time = np.sum(np.logical_and(data.patient_asleep, data["self similarity"])) / (3600 * hdr["newFs"])
        sim_df.loc[loc, "T_SS_new"] = round(SS_time / hdr["sleep_time"], 2)

        # T_osc: count of discrete tagged oscillation events normalised by
        # total sleep time.  find_events returns one tuple per contiguous
        # non-zero run, so len() gives the event count.
        osc_time = len(find_events(np.logical_and(data.patient_asleep, data.tagged > 0))) / hdr["sleep_time"]
        sim_df.loc[loc, "T_osc_new"] = round(osc_time, 2)

    return sim_df


# ---------------------------------------------------------------------------
# Patient selection
# ---------------------------------------------------------------------------


def patient_selection(
    sim_df: pd.DataFrame,
    version: str,
    sim_info_subset_path: str,
) -> None:
    """Create a clinically defined patient subset and save to CSV.

    Depending on *version*, this function applies different inclusion criteria
    and column-level filters to *sim_df*, producing a smaller ``selection_df``
    that is written to *sim_info_subset_path*.

    Supported versions:

    ``SS_cases``
        Stratified random sample: 20 patients from each of five T_SS
        percentage bands (0-5 %, 5-10 %, 10-20 %, 20-30 %, 30-100 %).
        This ensures balanced representation across the periodic-breathing
        severity spectrum.

    ``high_CAI``
        Patients with AHI > 10 **and** CAI > 5 (at 3 % desat) whose central
        apnea index exceeds their obstructive apnea index.  Sorted by CAI
        descending so the most central-dominant patients come first.

    ``HLG_OSA``
        Obstructive-predominant patients: CAI < 5, AHI > 15, sorted by
        descending T_SS.  Used for high loop-gain OSA analyses.

    ``REM_OSA``
        REM-dominant OSA: sufficient REM time (>10 % of total sleep), AHI > 15,
        and REM-AHI at least 3× higher than NREM-AHI.

    ``NREM_OSA``
        NREM-dominant OSA: sufficient REM time, AHI > 15, and NREM-AHI at
        least 2× higher than REM-AHI.

    ``CPAP_success`` / ``CPAP_failure``
        CPAP treatment response cohorts, cross-referenced against a separate
        CPAP outcomes CSV.  Success and failure groups are each trimmed to
        200 patients.  Failure patients are selected from those with the
        lowest CAI (< 10) and highest pre-treatment SS.

    Args:
        sim_df: Full cohort metadata DataFrame.
        version: Version string selecting the inclusion strategy.
        sim_info_subset_path: Output CSV path for the selected subset.
    """
    random.seed(0)

    if version == "SS_cases":
        # Stratified sampling across five SS-percentage bands.
        ranges = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1)]
        for i, maxi in ranges:
            ran = np.where(np.logical_and(sim_df["T_SS"] > i, sim_df["T_SS"] < maxi))[0]
            ran = random.sample(ran.tolist(), k=20)
            sim_df.loc[ran, "SS group"] = f"SS {i}-{maxi}"
        selection_df = sim_df.dropna(subset=["SS group"]).reset_index(drop=True)

    elif version == "high_CAI":
        # Central-apnea-dominant patients.
        selection_df = sim_df.loc[sim_df.ahi > 10]
        selection_df = selection_df.loc[selection_df["cai_3%"] > 5].sort_values(by=["cai"], ascending=False)
        selection_df = selection_df.query("cai > oai").reset_index(drop=True)
        selection_df.loc[:, "SS group"] = "N/A"

    elif version == "HLG_OSA":
        # Obstructive-predominant, high loop-gain OSA.
        selection_df = sim_df.loc[sim_df["cai_3%"] < 5]
        selection_df = selection_df.loc[selection_df["ahi_3%"] > 15].sort_values(by=["T_SS"], ascending=False)
        selection_df.loc[:, "SS group"] = "N/A"

    elif version == "REM_OSA":
        # REM-predominant OSA: REM time > 10 % of total sleep.
        selection_df = sim_df.loc[sim_df.REM_time > 0.1 * sim_df.h_sleep]
        selection_df = selection_df.loc[selection_df.ahi > 15]
        selection_df = selection_df.query("AHI_REM > 3*AHI_NREM").reset_index(drop=True)
        selection_df.loc[:, "SS group"] = "N/A"

    elif version == "NREM_OSA":
        # NREM-predominant OSA: REM time > 10 % of total sleep.
        selection_df = sim_df.loc[sim_df.REM_time > 0.1 * sim_df.h_sleep]
        selection_df = selection_df.loc[selection_df.ahi > 15]
        selection_df = selection_df.query("AHI_NREM > 2*AHI_REM").reset_index(drop=True)
        selection_df.loc[:, "SS group"] = "N/A"

    elif "CPAP" in version:
        # CPAP treatment response analysis.  Build both success and failure
        # subsets, then select based on version suffix.
        selection_dict: dict[str, pd.DataFrame] = {}
        for tag in ["failure", "success"]:
            cpap_csv_path = "csv_files/sim_df_03_20_2023_all.csv"
            cpap_df = pd.read_csv(cpap_csv_path)
            if tag == "success":
                cpap_df = cpap_df.loc[cpap_df["CPAP success 3%"] == "True"].reset_index(drop=True)
            else:
                cpap_df = cpap_df.loc[cpap_df["CPAP success 3%"] == "False"].reset_index(drop=True)

            subjectIDs = np.array([s.split("_")[0] for s in cpap_df["subjectID"]])

            # Cross-reference sim_df with CPAP outcomes to pull in
            # treatment-specific metrics.
            remove_locs: list[int] = []
            metrics = ["T_SS1", "AHI1_3%", "CAI1_3%", "subjectID"]
            for i, ID in enumerate(sim_df["HashID"].values):
                loc = np.where(ID == subjectIDs)[0]
                if len(loc) == 0:
                    remove_locs.append(i)
                    continue
                for metric in metrics:
                    sim_df.loc[i, metric] = cpap_df.loc[loc[0], metric]

            selection_dict[tag] = sim_df.drop(remove_locs)

            # Exclude patients with very high central apnea indices on CPAP
            # (CAI ≥ 10 likely indicates treatment-emergent central apnea).
            cai_mask = selection_dict[tag]["CAI1_3%"] < 10
            if tag == "failure":
                # Take the 300 with lowest CAI, then the 200 with highest
                # pre-treatment SS from that subset.
                low_cai_df = selection_dict[tag][cai_mask].sort_values(by=["CAI1_3%"], ascending=True)
                selection_dict[tag] = (
                    low_cai_df[:300].sort_values(by=["T_SS1"], ascending=False)[:200].reset_index(drop=True)
                )
            else:
                selection_dict[tag] = selection_dict[tag][cai_mask].sample(n=200, random_state=1).reset_index(drop=True)

        if "success" in version:
            selection_df = selection_dict["success"]
        else:
            selection_df = selection_dict["failure"]

    selection_df.to_csv(
        sim_info_subset_path,
        header=selection_df.columns,
        index=None,
        mode="w+",
    )


# ---------------------------------------------------------------------------
# File sorting
# ---------------------------------------------------------------------------


def sort_input_files(
    all_paths: list[str],
    sim_df: pd.DataFrame,
    version: str,
) -> list[str]:
    """Sort HF5 paths according to the version-specific ordering.

    Each version defines a column by which *sim_df* is sorted; the function
    then maps the sorted IDs back to the original *all_paths* list so that
    the exported CSV files are numbered in a clinically meaningful order.

    Args:
        all_paths: Unsorted list of absolute HF5 file paths.
        sim_df: Metadata DataFrame (already filtered to the desired subset).
        version: Version string that determines the sort column.

    Returns:
        *all_paths* re-ordered to match *sim_df*'s sort order.
    """
    # Strip directory and extension to get bare recording IDs.
    stripped_paths = np.array([p.split("/")[-1].split("\\")[-1].split(".hf5")[0] for p in all_paths])
    sorted_paths: list[str] = []

    if version == "SS_cases":
        for ID in sim_df.sort_values(by=["SS group"]).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    elif version == "high_CAI":
        for ID in sim_df.sort_values(by=["T_SS"]).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    elif version == "HLG_OSA":
        for ID in sim_df.sort_values(by=["T_SS"]).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    elif version == "REM_OSA":
        for ID in sim_df.sort_values(by=["T_SS"]).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    elif version == "NREM_OSA":
        for ID in sim_df.sort_values(by=["T_SS"]).SS_path:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    elif version == "Heart_Failure":
        # Heart-failure IDs are numeric (first 4 characters of path).
        stripped_paths = np.array([int(p[:4]) for p in stripped_paths])
        for ID in sim_df.sort_values(by=["EF"], ascending=False).ID:
            if ID not in stripped_paths:
                print(f"{ID} from redeker table not found among recordings")
                continue
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    elif "CPAP" in version:
        for ID in sim_df.sort_values(by=["T_SS1"])["subjectID"]:
            assert ID in stripped_paths, 'Somehow ID not found in "all_paths".'
            loc = np.where(ID == stripped_paths)[0][0]
            sorted_paths.append(all_paths[loc])

    return sorted_paths


def sort_altitude_files(
    all_paths: list[str],
    date: str,
) -> tuple[list[str], pd.DataFrame]:
    """Sort altitude-study recordings by patient and altitude level.

    The altitude study follows a naming convention ``P40-{patient}-{altitude}``
    where patient ∈ [1, 11] and altitude ∈ [1, 4].  This function iterates
    through all patient/altitude combinations, loads each recording's SS
    output to extract respiratory metrics, and builds a summary DataFrame.

    Args:
        all_paths: List of HF5 file paths for the altitude study.
        date: Date string that appears in the path before the patient folder
            (used to split the path and extract the patient identifier).

    Returns:
        A tuple of (sorted_paths, sim_df) where *sorted_paths* is ordered by
        patient then altitude, and *sim_df* contains one row per recording
        with respiratory metrics.
    """
    sim_df = pd.DataFrame([], columns=["num", "patient_num", "altitude"])
    sorted_paths: list[str] = []

    for n, num in enumerate(range(1, 12)):
        patient_paths = [p for p in all_paths if f"P40-{num}" in p]
        for a, alt in enumerate(range(1, 5)):
            print(f"extracting SS output P40-{num}-{alt}..", end="\r")
            path = [p for p in patient_paths if f"P40-{num}-{alt}" in p]
            if len(path) == 0:
                continue

            sorted_paths += path

            loc = len(sim_df)
            sim_df.loc[loc, "num"] = n * 10 + a
            sim_df.loc[loc, "patient_num"] = path[0].split(date)[-1].split("/")[1]
            sim_df.loc[loc, "altitude"] = alt

            data, hdr = load_sim_output(path[0])

            sim_df.loc[loc, "RDI_3%"] = hdr["RDI"]
            sim_df.loc[loc, "AHI_3%"] = hdr["AHI"]
            sim_df.loc[loc, "CAI_3%"] = hdr["CAI"]
            sim_df.loc[loc, "sleep_time"] = hdr["sleep_time"]

            # Same T_SS / T_osc computation as extract_latest_SS_outputs.
            SS_time = np.sum(np.logical_and(data.patient_asleep, data["self similarity"])) / (3600 * hdr["newFs"])
            sim_df.loc[loc, "T_SS_new"] = round(SS_time / hdr["sleep_time"], 2)

            osc_time = len(find_events(np.logical_and(data.patient_asleep, data.tagged > 0))) / hdr["sleep_time"]
            sim_df.loc[loc, "T_osc_new"] = round(osc_time, 2)

    return sorted_paths, sim_df


# ---------------------------------------------------------------------------
# Recording export
# ---------------------------------------------------------------------------


def segment_and_export_recordings(
    all_paths: list[str],
    version: str,
    dataset: str,
    save_folder: str,
) -> None:
    """Export all recordings as segment-level CSVs using multiprocessing.

    Creates a versioned output directory under *save_folder* and launches a
    process pool that converts each recording in *all_paths* into an EM-input
    CSV via :func:`segment_and_export_recording`.

    Args:
        all_paths: Sorted list of HF5 recording paths.
        version: Version string (included in the output folder name).
        dataset: Dataset identifier (included in the output folder name).
        save_folder: Parent directory for the output folder.
    """
    save_folder = os.path.join(save_folder, f"{dataset}_{version}_V7/")
    os.makedirs(save_folder, exist_ok=True)

    out_paths = [save_folder + f"Study {p + 1}.csv" for p in range(len(all_paths))]

    num_workers = cpu_count()
    pool = Pool(num_workers)
    process_args = [(path, out_path) for path, out_path in zip(all_paths, out_paths)]
    for _i, _ in enumerate(pool.starmap(segment_and_export_recording, process_args)):
        pass


def segment_and_export_recording(path: str, out_path: str) -> None:
    """Load a single recording, build ventilation, segment, and export CSV.

    Processing steps for each recording:

    1. Load the SS pipeline output (HF5) via :func:`~hlg.io.readers.load_sim_output`.
    2. Rename legacy column names to canonical schema names.
    3. Compute the breath-by-breath ventilation trace and fractional
       ventilation (d_i) via :func:`~hlg.core.ventilation.create_ventilation_trace`.
    4. Map each sample back to the original PSG sample indices (``ind0`` /
       ``ind1``) so that later analysis can cross-reference raw signals.
    5. Tag breaths as *ptaf* (pre-therapeutic airflow) or *cflow* (CPAP flow)
       based on the CPAP-start index stored in the header.
    6. Segment the night into NREM/REM blocks and compute per-segment SS
       scores via :mod:`hlg.ss.segmentation`.
    7. Write the export DataFrame to *out_path* as CSV.

    If *out_path* already exists the function returns immediately (idempotent
    re-runs).

    Args:
        path: Absolute path to the ``.hf5`` recording file.
        out_path: Destination CSV path.
    """
    if os.path.exists(out_path):
        return

    data, hdr = load_sim_output(path)

    # Rename legacy columns to the canonical schema used by the EM pipeline.
    data = data.rename(columns={"apnea": "Apnea"})
    data = data.rename(columns={"flow_reductions": "Apnea_algo"})
    data = data.rename(columns={"sleep_stages": "Stage"})
    data = data.rename(columns={"abd": "ABD"})
    data = data.rename(columns={"spo2": "SpO2"})

    data = create_ventilation_trace(data, hdr["newFs"], plot=False)

    # Map each downsampled sample to the original PSG sample range.
    # factor = Fs_original / Fs_downsampled gives the number of original
    # samples per downsampled sample.
    factor = hdr["Fs"] // hdr["newFs"]
    ind0 = np.arange(0, data.shape[0]) * factor
    ind1 = np.concatenate([ind0[1:], [ind0[-1] + factor]])
    data["ind0"] = ind0
    data["ind1"] = ind1

    # Tag breaths as ptaf (diagnostic/pre-therapeutic airflow) or cflow
    # (CPAP flow).  cpap_start == 0 means no CPAP was applied.
    tags = ["ptaf"] * len(data)
    if hdr["cpap_start"] > 0:
        tags[hdr["cpap_start"] :] = ["cflow"] * (len(data) - hdr["cpap_start"])
    data["breath_tag"] = tags

    export_cols = [
        "ind0",
        "ind1",
        "ABD",
        "Ventilation_ABD",
        "Eupnea_ABD",
        "d_i_ABD",
        "d_i_ABD_smooth",
        "arousal_locs",
        "Apnea",
        "Apnea_algo",
        "Stage",
        "SpO2",
        "breath_tag",
    ]
    export_data = pd.DataFrame([], columns=export_cols)
    for c in export_cols:
        export_data[c] = data[c]

    # Segment the night into NREM/REM blocks for EM input.
    seg_dic = segment_data_based_on_nrem(data, hdr["newFs"])
    for key in seg_dic.keys():
        export_data.loc[: len(seg_dic[key]) - 1, key] = seg_dic[key]

    # Compute per-segment SS quality scores.
    SS_seg_scores = compute_SS_score_per_segement(data, seg_dic)
    for key in SS_seg_scores.keys():
        export_data.loc[: len(SS_seg_scores[key]) - 1, f"{key}_SS_score"] = SS_seg_scores[key]

    export_data["patient_tag"] = hdr["patient_tag"]
    export_data["Fs"] = hdr["newFs"]
    export_data["original_Fs"] = hdr["Fs"]

    export_data.to_csv(out_path, header=export_data.columns, index=None, mode="w+")
