#!/usr/bin/env python
"""
Run the CPAP treatment outcome prediction analysis.

Computes logistic regression models comparing CAI, LG (with IQR range),
and combined features for predicting CPAP treatment failure.

Usage:
    python -m scripts.run_cpap_analysis
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlg.analysis.cpap import (
    compute_logistic_regression,
    set_AUC_curve_layout,
    compute_calibration_curve,
    set_calibration_curve_layout,
)
from hlg.em.histograms import load_histogram_bars, predict_CPAP_SUCCESS_from_bars
from hlg.config import config

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    smooth = "non-smooth"
    interm_base = os.path.join(config.interm_dir, smooth)

    success_results = pd.read_csv(os.path.join(interm_base, "bdsp_CPAP_success", "all_segments.csv"))
    failure_results = pd.read_csv(os.path.join(interm_base, "bdsp_CPAP_failure", "all_segments.csv"))

    LG_data = np.concatenate([success_results["LG_data"].values, failure_results["LG_data"].values])
    valid_data = np.concatenate([success_results["valid_data"].values, failure_results["valid_data"].values])
    ID_data = np.concatenate([success_results["ID_data"].values * -1, failure_results["ID_data"].values])

    success_info_df = pd.read_csv(os.path.join(config.csv_dir, "bdsp_table1 200_CPAP_success_cases.csv"))
    failure_info_df = pd.read_csv(os.path.join(config.csv_dir, "bdsp_table1 200_CPAP_failure_cases.csv"))
    maxi = len(success_info_df)
    success_info_df["ID"] = range(-1, -maxi - 1, -1)
    failure_info_df["ID"] = range(1, maxi + 1)
    cols = ["ID", "CAI1_3%", "T_SS1", "AHI1_3%"]
    info_df = pd.concat([success_info_df[cols], failure_info_df[cols]])

    bar_folder = os.path.join(config.bars_dir, "bdsp_")
    bars_success, bars_failure = load_histogram_bars(bar_folder)
    all_bars = np.array(bars_success + bars_failure)
    info_df = predict_CPAP_SUCCESS_from_bars(info_df, all_bars, bars_success, bars_failure)

    IDs = ID_data[valid_data.astype(bool)]
    unique_IDs = np.unique(IDs)
    LG_median = np.array([np.median(LG_data[valid_data.astype(bool)][IDs == ID]) for ID in unique_IDs])
    LG_25 = np.array([np.percentile(LG_data[valid_data.astype(bool)][IDs == ID], 25) for ID in unique_IDs])
    LG_75 = np.array([np.percentile(LG_data[valid_data.astype(bool)][IDs == ID], 75) for ID in unique_IDs])
    yy = np.array([0 if ID < 0 else 1 for ID in unique_IDs])
    cai = np.array([info_df.loc[info_df["ID"] == ID, "CAI1_3%"].values[0] for ID in unique_IDs])
    ss = np.array([info_df.loc[info_df["ID"] == ID, "T_SS1"].values[0] for ID in unique_IDs])

    LG_range = np.column_stack([LG_25, LG_median, LG_75])
    cai = np.expand_dims(cai, axis=1)
    ss = np.expand_dims(ss, axis=1)
    combined_x = np.concatenate([LG_range, ss], axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    prob_cai, y_cai = compute_logistic_regression(cai, yy, "CAI", axs=axs)
    prob_LG, y_LG = compute_logistic_regression(LG_range, yy, "LG", axs=axs)
    prob_combined, y_combined = compute_logistic_regression(combined_x, yy, "Combined", axs=axs)
    set_AUC_curve_layout(axs, "Features              AUC")

    compute_calibration_curve(prob_cai, y_cai, "CAI", axs[2])
    compute_calibration_curve(prob_LG, y_LG, "LG", axs[2])
    compute_calibration_curve(prob_combined, y_combined, "Combined", axs[2])
    set_calibration_curve_layout([axs[2]])

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cpap_prediction", "CPAP_ROC_PR_calibration.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
