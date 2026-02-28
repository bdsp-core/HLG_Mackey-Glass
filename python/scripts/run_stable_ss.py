#!/usr/bin/env python
"""
Run the stable self-similarity detector on a cohort of recordings.

For each recording, computes oscillation chains and stable SS regions
via change-point detection, then generates per-recording visualizations
and cohort-level length histograms.

Usage:
    python -m scripts.run_stable_ss
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlg.io.readers import load_sim_output
from hlg.ss.stable import compute_osc_chains, compute_change_points_ruptures
from hlg.visualization.stable_ss import plot_SS, create_length_histogram
from hlg.config import config

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def run_stable_SS_detector(i, path, sim_df, output_folder, dataset):
    """Process a single recording: load, compute SS features, plot."""
    subjectID = path.split("/")[-1].split(".hf")[0]
    out_path = f"{output_folder}{subjectID}.png"
    already = len(os.listdir(output_folder))
    ratio = f"{i + 1}/{len(sim_df)}"
    print(f"Assessing {dataset.upper()} recording: {ratio} ({already})")

    try:
        data, hdr = load_sim_output(path)
        hdr["group"] = sim_df.loc[np.where(sim_df.SS_path == subjectID)[0][0], "SS group"]
        hdr["SS_threshold"] = config.ss_threshold
    except Exception as error:
        print("Loading error: ", error)
        return (None, None)

    win = int(3 * 60 * hdr["newFs"])
    data["SS_trace"] = data["ss_conv_score"].rolling(win, min_periods=1, center=True).median().fillna(0)

    data, hdr = compute_osc_chains(data, hdr)
    data = compute_change_points_ruptures(data, hdr)

    if not os.path.exists(out_path):
        plot_SS(data, hdr, out_path=out_path)

    return (data, hdr)


def main():
    dataset = "mgh"
    version = "SS_cases"

    input_folder = config.hf5_dir or "SS paper files/"
    all_paths = glob.glob(input_folder + "*.hf5")

    output_folder = os.path.join(FIGURES_DIR, "stable_ss", f"{dataset}_{version}")
    os.makedirs(output_folder, exist_ok=True)
    output_folder += "/"  # trailing slash expected by run_stable_SS_detector

    sim_df = pd.read_csv(os.path.join(config.csv_dir, "mgh_table1 100_SS_cases.csv"))

    result = []
    for i, p in enumerate(all_paths):
        r = run_stable_SS_detector(i, p, sim_df, output_folder, dataset)
        result.append(r)

    hist_dir = os.path.join(FIGURES_DIR, "stable_ss")
    os.makedirs(hist_dir, exist_ok=True)

    create_length_histogram(sim_df, result, version="Osc_chain")
    out1 = os.path.join(hist_dir, "Osc_chain_length_histogram.png")
    plt.savefig(out1, dpi=1200, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    create_length_histogram(sim_df, result, version="stable_SS")
    out2 = os.path.join(hist_dir, "Stable_SS_length_histogram.png")
    plt.savefig(out2, dpi=1200, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()


if __name__ == "__main__":
    main()
