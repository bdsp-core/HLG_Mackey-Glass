#!/usr/bin/env python
"""
Update the CAISR MGH clinical table with per-recording HLG metrics.

Iterates over HDF5 output files from the Self-Similarity pipeline,
matches each recording to the CAISR MGH clinical table (by HashID and
date-of-visit), and enriches the table with computed sleep metrics:

  * **3 % severity indices** -- AHI, RDI, CAI (from the pipeline header).
  * **Individual event indices** -- obstructive, central, mixed, hypopnea,
    and RERA indices (events per hour of sleep).
  * **Self-similarity metrics** -- total SS time as a fraction of sleep,
    and the oscillation event rate.
  * **REM / NREM-specific indices** -- AHI and CAI computed separately
    for REM and NREM sleep, enabling stage-specific severity profiling.

Recordings are excluded if the patient HashID is not in the clinical
table or if the date-of-visit does not match (date format differences
between the HDF5 filename and the CSV column are normalised before
comparison).

Usage:
    python -m scripts.update_mgh_info
"""

import glob
import os

import numpy as np
import pandas as pd

from hlg.core.events import find_events
from hlg.core.sleep_metrics import compute_sleep_metrics
from hlg.io.readers import load_sim_output


def main():
    # ── Configuration ─────────────────────────────────────────────────
    dataset = "mgh"
    date = "11_09_2023"
    exp = "Expansion" if os.path.exists("/media/cdac/Expansion/CAISR data1/Rule_based") else "Expansion1"
    input_folder = f"/media/cdac/{exp}/LG project/hf5data/{dataset}_{date}/"
    input_files = glob.glob(input_folder + "*.hf5")

    # Extract bare recording identifiers (HashID_DOV) from file paths.
    stripped_paths = np.array([p.split("/")[-1].split(".hf5")[0] for p in input_files])

    # ── Load the CAISR MGH clinical table ─────────────────────────────
    table_path = "csv_files/caisr_mgh_v4_table1.csv"
    caisr_table = pd.read_csv(table_path)
    caisr_table = caisr_table.drop(columns=["mgh_v4", "path_prepared"])

    sim_df = pd.DataFrame([])
    not_in_T1, DOV_mismatch = 0, 0

    for i, path in enumerate(stripped_paths):
        print(f"extracting SS output {i}/{len(stripped_paths)} ..", end="\r")

        ID = path.split("_")[0]
        DOV = path.split("_")[1]

        # Skip recordings whose patient HashID is not in the clinical
        # table (e.g. test recordings or patients excluded from the
        # study cohort).
        if ID not in caisr_table.HashID.values:
            not_in_T1 += 1
            continue

        loc = np.where(ID == caisr_table.HashID.values)[0]

        # The clinical table stores the date-of-visit as
        # "MM/DD/YYYY HH:MM" -- strip the time part and slashes to
        # match the YYYYMMDD format used in the HDF5 filename.
        table_DOV = caisr_table.loc[loc[0], "DOVshifted"].split(" ")[0].replace("/", "")
        if DOV != table_DOV:
            DOV_mismatch += 1
            continue

        # Merge the clinical row into the output DataFrame.
        sim_df = pd.concat([sim_df, caisr_table.loc[loc, :]], ignore_index=True)
        sim_df.loc[len(sim_df) - 1, "SS_path"] = path

        # ── Read the SS pipeline HDF5 output ──────────────────────────
        try:
            cols = ["apnea", "sleep_stages", "self similarity", "tagged"]
            data, hdr = load_sim_output(input_files[i], cols=cols)
        except Exception:
            continue

        # ── 3 % severity indices (from the pipeline header) ───────────
        for metric in ["AHI", "RDI", "CAI"]:
            sim_df.loc[len(sim_df) - 1, f"{metric.lower()}_3%"] = hdr[metric]

        # ── Individual respiratory event indices ──────────────────────
        # Apnea label encoding: 1=Obs, 2=Cen, 3=Mix, 4=Hyp, 5=RERA.
        resp_map = {1: "Obs", 2: "Cen", 3: "Mix", 4: "Hyp", 5: "RERA"}
        for key in resp_map.keys():
            num = len(find_events(np.logical_and(data.patient_asleep, data.apnea == key))) / hdr["sleep_time"]
            tag = resp_map[key] + "_i"
            sim_df.loc[len(sim_df) - 1, tag] = num

        # ── Self-similarity metrics ───────────────────────────────────
        # T_SS: fraction of total sleep time that is self-similar
        # (periodic breathing), expressed as a ratio (0-1).
        SS_time = np.sum(np.logical_and(data.patient_asleep, data["self similarity"])) / (3600 * hdr["newFs"])
        sim_df.loc[len(sim_df) - 1, "T_SS"] = round(SS_time / hdr["sleep_time"], 2)

        # T_osc: number of tagged oscillation events per hour of sleep.
        osc_num = len(find_events(np.logical_and(data.patient_asleep, data.tagged > 0))) / hdr["sleep_time"]
        sim_df.loc[len(sim_df) - 1, "T_osc"] = osc_num

        # ── Stage-specific severity indices (REM vs NREM) ─────────────
        # Separate sleep into REM (stage 4) and NREM (stages 1-3) to
        # compute AHI and CAI independently for each stage, enabling
        # phenotyping of stage-dependent respiratory instability.
        REM_region = np.where(data.sleep_stages == 4)[0]
        NREM_region = np.where(np.logical_and(data.sleep_stages > 0, data.sleep_stages < 4))[0]
        regions, tags = [REM_region, NREM_region], ["REM", "NREM"]

        for region, tag in zip(regions, tags):
            if len(region) > 0:
                apneas = data.loc[region, "apnea"].values
                stages = data.loc[region, "sleep_stages"].values
                RDI, AHI, CAI, sleep_time = compute_sleep_metrics(apneas, stages, exclude_wake=True)
            else:
                RDI, AHI, CAI, sleep_time = 0, 0, 0, 0

            sim_df.loc[len(sim_df) - 1, f"RDI_{tag}"] = RDI
            sim_df.loc[len(sim_df) - 1, f"AHI_{tag}"] = AHI
            sim_df.loc[len(sim_df) - 1, f"CAI_{tag}"] = CAI
            sim_df.loc[len(sim_df) - 1, f"{tag}_time"] = sleep_time

    print(f"{not_in_T1} not in Table 1")
    print(f"{DOV_mismatch} DOV mismatch")

    # ── Save the enriched clinical table ──────────────────────────────
    sim_path_updated = table_path.replace(".csv", "_updated.csv")
    sim_df.to_csv(sim_path_updated, header=sim_df.columns, index=None, mode="w+")


if __name__ == "__main__":
    main()
