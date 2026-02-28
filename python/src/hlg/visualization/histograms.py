"""
LG histogram visualisation for CPAP success vs. failure comparison.

This module produces side-by-side bar charts showing the distribution
of per-segment loop gain (LG) scores for two patient cohorts: those
who responded successfully to CPAP therapy (green) and those who
failed CPAP (red).

The histogram uses 0.1-wide bins from 0.0 to 1.0 (covering the
clinically relevant LG range).  Each bar represents the mean
percentage of 8-minute segments whose average LG falls in that bin,
averaged across all patients in the cohort.

This figure supports the hypothesis that CPAP-failure patients have
a rightward-shifted LG distribution (more segments with high LG)
compared to CPAP-success patients -- consistent with the idea that
high loop gain predisposes to persistent central apneas under CPAP
(treatment-emergent central sleep apnea).

Source: ``EM_output_histograms.py`` -> ``total_histogram_plot``
"""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np


def total_histogram_plot(
    bars1: list[np.ndarray],
    bars2: list[np.ndarray],
) -> None:
    """Plot a paired histogram of LG distributions for two cohorts.

    Each patient's LG distribution has already been summarised as a
    vector of bin percentages (one value per 0.1-wide LG bin, produced
    by ``compute_histogram`` and ``histogram_bins_to_bars`` in the EM
    histogram pipeline).  This function averages those vectors across
    patients within each cohort and displays them as side-by-side bars.

    The bar positions are centred at bin midpoints (0.05, 0.15, …, 0.95)
    with a slight horizontal offset (+/-0.023) to separate the two cohorts
    visually.  Each bar is outlined in black for clarity against the
    coloured fill.

    Args:
        bars1: List of per-patient bar-height arrays for the "success"
            cohort.  Each element is a 1-D numpy array of length 10
            (one percentage per LG bin from 0.0-0.1 to 0.9-1.0).
        bars2: Same structure for the "failure" cohort.

    Side Effects:
        Creates a matplotlib figure with a single ``Axes``.  The figure
        is **not** saved or shown -- the caller is responsible for
        ``plt.savefig()`` or ``plt.show()``.
    """
    # Compute the cohort-averaged bar heights.
    # Each row in bars1/bars2 is a patient; we take the column-wise mean.
    mean1: np.ndarray = np.mean(bars1, 0)
    mean2: np.ndarray = np.mean(bars2, 0)

    # Bin midpoints: 0.05, 0.15, ..., 0.95 (centre of each 0.1-wide bin).
    ranges: np.ndarray = np.arange(0.05, 1, 0.1)

    # --- Create figure ---
    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Success cohort (green bars, offset left).
    ax.bar(
        [r - 0.023 for r in ranges],
        mean1,
        color="g",
        edgecolor="k",
        width=0.0455,
        label=f"success (N={len(bars1)})",
    )

    # Failure cohort (red bars, offset right).
    ax.bar(
        [r + 0.023 for r in ranges],
        mean2,
        color="r",
        edgecolor="k",
        width=0.0455,
        label=f"failure (N={len(bars2)})",
    )

    # --- Layout ---
    ax.set_ylabel("%", fontsize=10, fontweight="bold")
    ax.set_xlabel("LG score", fontsize=10, fontweight="bold")
    # x-axis spans [0, 1] (the full LG range of interest).
    ax.set_xlim(0, 1.01)
    # y-axis spans [0, 100] (percentage of segments).
    ax.set_ylim(0, 101)

    plt.title(
        "Mean Histogram: CPAP success vs failure\nAvg. LG estimation within 8 min segments.",
        fontweight="bold",
    )
    plt.legend()
