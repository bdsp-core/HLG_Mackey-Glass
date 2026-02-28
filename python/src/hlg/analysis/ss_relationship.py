"""
Self-Similarity (SS) score vs. Loop Gain (LG) relationship analysis.

This module extracts per-segment EM estimates grouped by SS score and
by patient SS group, producing intermediate CSV files that are then
used for scatterplot and polynomial regression figures showing the
relationship between segment-level self-similarity and estimated loop
gain.

The key insight from this analysis is that higher self-similarity
(more periodic / repetitive breathing patterns) is associated with
higher loop gain, and the relationship is well described by a
second-order polynomial.  This supports the use of SS as a non-invasive
proxy for loop gain.

The extraction function ``extract_EM_output_old`` is the "old-style"
variant that groups results by *patient-level* SS group (e.g. low-SS,
medium-SS, high-SS) rather than the flat all-segments format used by
``hlg.analysis.group.extract_EM_output``.  Both the per-group and
per-SS-bin CSVs are written as intermediate output.

Curve-fitting utilities (``quadratic_model``, ``prediction_band``) and
dictionary sorting are imported from ``hlg.analysis.statistics`` to
avoid duplication.

Source: ``EM_output_to_SS_Relationship.py`` (``extract_EM_output_old``,
plus the ``__main__`` plotting logic which is not included here).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from hlg.analysis.statistics import sort_dic_keys
from hlg.core.events import find_events
from hlg.em.loop_gain import create_total_LG_array
from hlg.io.readers import load_sim_output
from hlg.ss.scoring import convert_ss_seg_scores_into_arrays

from hlg.em.postprocessing import (
    add_arousals,
    match_EM_with_SS_output,
    post_process_EM_output,
)


def extract_EM_output_old(
    input_files: list[str],
    interm_folder: str,
    hf5_folder: str,
    version: str,
    dataset: str,
    csv_file: str,
    error_thresh: float = 1.8,
) -> None:
    """Extract per-segment EM estimates grouped by patient SS group.

    This is the "old-style" extraction that produces two levels of
    grouping:

    1. **Per patient SS group** -- each patient is assigned to an SS
       severity group (e.g. ``'low-SS'``, ``'high-SS'``) based on the
       arousal/SS annotation in the metadata CSV.  Within each group,
       all valid segments are concatenated and written to
       ``{interm_folder}/{group}.csv``.

    2. **Per SS score bin** -- segments are also binned into 0.2-wide
       SS-score ranges (0.0-0.2, 0.2-0.4, …, 0.8-1.0) regardless of
       patient group, and written to ``{interm_folder}/seg_{lo}-{hi}.csv``.

    For each study file, the pipeline:
      a) Reads the EM output CSV and converts sparse SS scores to dense.
      b) Post-processes the EM output (LG outlier smoothing).
      c) Loads arousal annotations and retrieves the patient's SS group.
      d) Matches with the SS pipeline HDF5 output and validates alignment.
      e) Reconstructs the LG hypnogram and saves it to ``hypnograms/``.
      f) Iterates over NREM/REM segments, collecting parameters.
      g) Applies the RMSE error threshold to exclude poor fits.
      h) Accumulates results into per-group and per-bin dictionaries.

    The per-bin output also includes sleep stage labels (``St``) to
    support NREM-vs-REM sub-analyses of the SS-LG relationship.

    Args:
        input_files: List of EM output CSV file paths.
        interm_folder: Output directory for intermediate CSVs.
        hf5_folder: Directory containing SS pipeline ``.hf5`` files.
        version: Analysis version string (passed to ``add_arousals``
            and ``match_EM_with_SS_output``).
        dataset: Dataset identifier (``'mgh'``, ``'redeker'``, etc.).
        error_thresh: RMSE threshold; segments with higher error are
            excluded (default 1.8).

    Side Effects:
        Writes CSV files to ``interm_folder``:
        - ``{group}.csv`` for each patient SS group
        - ``seg_{lo}-{hi}.csv`` for each 0.2-wide SS-score bin
        - ``hypnograms/Study {num}.csv`` for each study
    """
    # --- Initialise per-group dictionaries ---
    # Each dictionary maps SS group label -> numpy array of parameter values.
    LG_data: dict[str, np.ndarray] = {}
    G_data: dict[str, np.ndarray] = {}
    D_data: dict[str, np.ndarray] = {}
    GxD_data: dict[str, np.ndarray] = {}
    Valid_data: dict[str, np.ndarray] = {}
    Stage_data: dict[str, np.ndarray] = {}

    # --- Initialise per-SS-bin dictionary ---
    # Bins span [0.0, 0.2), [0.2, 0.4), ..., [0.8, 1.0).
    # Each bin stores separate lists for SS, LG, G, D, validity (VV),
    # and sleep stage (St).
    SS_data: dict[str, list] = {}
    for i in np.arange(0, 10, 2):
        for param in ["SS", "LG", "G", "D", "VV", "St"]:
            SS_data[f"seg_{i / 10}-{(i + 2) / 10}_{param}"] = []

    # --- Main extraction loop ---
    for idx, input_file in enumerate(input_files):
        num = input_file.split("/Study")[-1].split(".csv")[0]
        print(
            f"Extracting Study {num} ({idx + 1}/{len(input_files)}) ..",
            end="\r",
        )

        # Read and prepare the EM output DataFrame.
        data = pd.read_csv(input_file)
        data = convert_ss_seg_scores_into_arrays(data)
        data = post_process_EM_output(data)

        # Add arousal annotations and retrieve the patient's SS group.
        hdr: dict[str, Any] = {"Study_num": f"Study {num}"}
        _, hdr["SS group"] = add_arousals(data, version, "mgh", hf5_folder, csv_file)

        # Match with SS pipeline output for segment validity checks.
        sim_path, _ = match_EM_with_SS_output(data, dataset, csv_file)
        path = hf5_folder + sim_path + ".hf5"
        SS_df, SS_hdr = load_sim_output(path, ["flow_reductions"])
        assert len(SS_df) > 0.99 * len(data), "matching SS output does not match EM data"

        # Reconstruct sample-level LG array and save as CSV.
        total_LG = create_total_LG_array(data)
        total_LG_df = pd.DataFrame(total_LG, columns=["LG_hypno"])
        hypno_folder = interm_folder + "/hypnograms/"
        os.makedirs(hypno_folder, exist_ok=True)
        out_path = f"{hypno_folder}Study {num}.csv"
        total_LG_df.to_csv(
            out_path,
            header=total_LG_df.columns,
            index=None,
            mode="w+",
        )

        # Extract per-column header fields.
        for col in ["patient_tag", "Fs", "original_Fs"]:
            hdr[col] = data.loc[0, col]
            data = data.drop(columns=col)

        # --- Collect per-segment parameters ---
        Errors: list[float] = []
        Vmaxs: list[float] = []
        LGs: list[float] = []
        Gs: list[float] = []
        Ds: list[float] = []
        Ls: list[float] = []
        SSs_seg: list[float] = []
        valid_seg: list[bool] = []
        Stages: list[str] = []

        for stage in ["nrem", "rem"]:
            starts = data[f"{stage}_starts"].dropna().values.astype(int)
            ends = data[f"{stage}_ends"].dropna().values.astype(int)
            group = hdr["SS group"]

            for start, end in zip(starts, ends):
                loc = np.where(data[f"{stage}_starts"] == start)[0][0]
                Errors.append(round(data.loc[loc, "rmse_Vo"], 2))
                Ls.append(data.loc[loc, f"L_{stage}"])
                Vmaxs.append(round(data.loc[loc, "Vmax"], 2))
                LGs.append(data.loc[loc, f"LG_{stage}_corrected"])
                Gs.append(data.loc[loc, f"G_{stage}"])
                Ds.append(data.loc[loc, f"D_{stage}"])
                # Segment is valid if it contains >= 5 flow-reduction events.
                valid_seg.append(len(find_events(SS_df.loc[start:end, "flow_reductions"] > 0)) >= 5)
                SSs_seg.append(data.loc[start, "SS_score"])
                Stages.append(stage)

        # --- Apply error threshold ---
        inds = np.array(Errors) < error_thresh
        LGs_arr = np.array(LGs)[inds]
        Gs_arr = np.array(Gs)[inds]
        Ds_arr = np.array(Ds)[inds]
        SSs_arr = np.array(SSs_seg)[inds]
        valids_arr = np.array(valid_seg)[inds]
        Stages_arr = np.array(Stages)[inds]

        # --- Accumulate into per-group dictionaries ---
        # First encounter with a group creates the entry; subsequent
        # encounters concatenate.
        if group not in LG_data:
            LG_data[group] = LGs_arr
            G_data[group] = Gs_arr
            D_data[group] = Ds_arr
            GxD_data[group] = Gs_arr * Ds_arr
            Valid_data[group] = valids_arr
            Stage_data[group] = Stages_arr
        else:
            LG_data[group] = np.concatenate([LG_data[group], LGs_arr])
            G_data[group] = np.concatenate([G_data[group], Gs_arr])
            D_data[group] = np.concatenate([D_data[group], Ds_arr])
            GxD_data[group] = np.concatenate([GxD_data[group], Gs_arr * Ds_arr])
            Valid_data[group] = np.concatenate([Valid_data[group], valids_arr])
            Stage_data[group] = np.concatenate([Stage_data[group], Stages_arr])

        # --- Bin segments by SS score ---
        for LG, G, D, SS, VV, St in zip(
            LGs_arr,
            Gs_arr,
            Ds_arr,
            SSs_arr,
            valids_arr,
            Stages_arr,
        ):
            for i in np.arange(0, 10, 2):
                ran = f"seg_{i / 10}-{(i + 2) / 10}"
                if SS >= i and SS < i + 0.2:
                    break
            SS_data[ran + "_SS"].append(SS)
            SS_data[ran + "_LG"].append(LG)
            SS_data[ran + "_G"].append(G)
            SS_data[ran + "_D"].append(D)
            SS_data[ran + "_VV"].append(VV)
            SS_data[ran + "_St"].append(St)

    # --- Sort dictionaries for reproducible output ---
    sorted_dics = sort_dic_keys([LG_data, G_data, D_data, GxD_data, Valid_data, Stage_data])
    names = ["LG_data", "G_data", "D_data", "GxD_data", "Valid_data", "Stage_data"]

    # --- Write per-group CSVs ---
    for group in LG_data.keys():
        out_path = f"{interm_folder}/{group}.csv"
        df = pd.DataFrame([], dtype=float)
        for dic, name in zip(sorted_dics, names):
            df[name] = dic[group]
        df.to_csv(
            f"{interm_folder}/{group}.csv",
            header=df.columns,
            index=None,
            mode="w+",
        )

    # --- Write per-SS-bin CSVs ---
    for i in np.arange(0, 10, 2):
        df = pd.DataFrame([], dtype=float)
        ran = f"seg_{i / 10}-{(i + 2) / 10}"
        for param in ["SS", "LG", "G", "D", "VV", "St"]:
            df[param] = SS_data[f"{ran}_{param}"]
        df.to_csv(
            f"{interm_folder}/{ran}.csv",
            header=df.columns,
            index=None,
            mode="w+",
        )
