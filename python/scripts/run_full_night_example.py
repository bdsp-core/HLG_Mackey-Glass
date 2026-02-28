#!/usr/bin/env python
"""
Generate a simplified full-night overview figure from a single HF5 recording.

This script produces the multi-row full-night figure directly from the SS
pipeline HF5 output, without requiring EM model results. It shows:

  - Abdominal RIP trace colour-coded by sleep stage (black=NREM, blue=REM, red=Wake)
  - Respiratory event bars (apneas in blue, hypopneas in magenta)
  - Tagged breathing oscillation markers
  - Self-similarity regions
  - Clinical summary in the header (duration, CAI, CAHI, SS%)

The LG estimation hooks are omitted since they require the EM output CSV.

Usage:
    python -m scripts.run_full_night_example
"""

import os
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np

from hlg.config import config
from hlg.io.readers import load_sim_output
from hlg.reporting import create_report
from hlg.visualization.full_night import add_LG_hooks

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def _has_em_data(em_csv_path: str | None) -> bool:
    """Check if an EM output CSV exists and contains LG columns."""
    if em_csv_path is None or not os.path.exists(em_csv_path):
        return False
    import pandas as pd

    cols = pd.read_csv(em_csv_path, nrows=0).columns
    return "LG_nrem" in cols or "LG_nrem_corrected" in cols


def plot_full_night_from_hf5(hf5_path: str, out_path: str, em_csv_path: str | None = None) -> None:
    """Generate the full-night overview figure from a single HF5 file.

    If ``em_csv_path`` is provided and contains EM model output (LG
    estimates per segment), LG hook annotations are added to the figure.
    Otherwise, the figure is generated without hooks.
    """

    # Load recording
    print(f"Loading {os.path.basename(hf5_path)}...")
    data, hdr = load_sim_output(hf5_path)

    # Rename columns to match the reporting convention
    data = data.rename(columns={"self similarity": "T_sim", "sleep_stages": "stage"})

    # Trim trailing NaN / excessive wake
    stages = data["stage"].values
    finite_mask = np.where(np.isfinite(stages))[0]
    if len(finite_mask) == 0:
        print("  No valid sleep staging found, skipping.")
        return
    end = finite_mask[-1]
    data = data.iloc[: end + 1].reset_index(drop=True)

    # Generate summary report
    _, summary_report = create_report(data, hdr)

    # Extract signal arrays
    signal = data.abd.values.astype(float)
    sleep_stages = data.stage.values.astype(float)
    y_algo = data.flow_reductions.values.astype(float)
    tagged_breaths = data.tagged.values.astype(float)
    ss_conv_score = data.ss_conv_score.values.astype(float)
    selfsim = data.T_sim.values.astype(float)

    # Row layout: each row = 1 hour at 10 Hz
    fs = hdr["newFs"]
    block = 60 * 60 * fs
    row_ids = [np.arange(i * block, min((i + 1) * block, len(signal))) for i in range(len(signal) // block + 1)]
    row_ids.reverse()
    nrow = len(row_ids)

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    row_height = 16

    # Build stage-separated signal arrays
    sleep = np.array(signal)
    sleep[np.isnan(sleep_stages)] = np.nan
    sleep[sleep_stages == 5] = np.nan

    wake = np.zeros(signal.shape)
    wake[np.isnan(sleep_stages)] += signal[np.isnan(sleep_stages)]
    wake[sleep_stages == 5] += signal[sleep_stages == 5]
    wake[wake == 0] = np.nan

    rem = np.array(signal)
    rem[sleep_stages != 4] = np.nan

    # ── Plot signals (one trace per row) ──────────────────────────────
    for ri in range(nrow):
        ax.plot(sleep[row_ids[ri]] + ri * row_height, c="k", lw=0.3, alpha=0.75)
        ax.plot(wake[row_ids[ri]] + ri * row_height, c="r", lw=0.3, alpha=0.5)
        ax.plot(rem[row_ids[ri]] + ri * row_height, c="b", lw=0.3, alpha=0.5)

    # ── Plot labels ───────────────────────────────────────────────────
    for yi in range(3):
        if yi == 0:
            labels = y_algo
            label_color = [None, "b", "b", "b", "m"]
        elif yi == 1:
            labels = tagged_breaths
            label_color = [None, "k", "r"]
        else:
            labels = selfsim
            label_color = [None, "b"]

        for ri in range(nrow):
            loc = 0
            for val, group in groupby(labels[row_ids[ri]]):
                len_j = len(list(group))
                if np.isfinite(val) and int(val) < len(label_color) and label_color[int(val)] is not None:
                    if yi == 0:
                        # Respiratory event bars
                        shift = 3.5 if val == 1 else 4
                        ax.plot(
                            [loc, loc + len_j],
                            [ri * row_height - shift] * 2,
                            c=label_color[int(val)],
                            lw=1.5,
                            alpha=1,
                        )
                    elif yi == 1:
                        # Tagged breathing oscillation markers
                        c_score = np.round(ss_conv_score[row_ids[ri]][loc], 2)
                        if np.isfinite(c_score) and c_score >= hdr.get("SS_threshold", 0.5):
                            tag_char = "o" if val == 1 else "'"
                            ax.text(
                                loc,
                                ri * row_height - 5,
                                tag_char,
                                c="b",
                                ha="center",
                                va="center",
                                fontsize=6,
                            )
                loc += len_j

    # ── LG hooks (only if EM data is available) ─────────────────────
    em_data_loaded = False
    if _has_em_data(em_csv_path):
        import pandas as pd
        from hlg.em.postprocessing import post_process_EM_output
        from hlg.ss.scoring import convert_ss_seg_scores_into_arrays

        print("  EM data found — adding LG hooks...")
        em_data = pd.read_csv(em_csv_path)
        em_data = convert_ss_seg_scores_into_arrays(em_data)
        em_data = post_process_EM_output(em_data)

        # Trim to match HF5 data length
        min_len = min(len(em_data), len(data))
        em_data = em_data.iloc[:min_len].reset_index(drop=True)

        add_LG_hooks(em_data, data, hdr, row_ids, nrow, row_height, fs, ax)
        em_data_loaded = True

    # ── Layout ────────────────────────────────────────────────────────
    ax.set_xlim([0, max(len(x) for x in row_ids)])
    ax.axis("off")

    # Summary report header
    len_x = len(row_ids[-1])
    fz = 11
    offset = row_height * (nrow - 1) + 17
    dx = len_x // 10
    for i, key in enumerate(summary_report.keys()):
        tag = key.replace("detected ", "") + ":\n" + str(summary_report[key].values[0])
        ax.text(i * dx, offset, tag, fontsize=7, ha="left", va="bottom")

    # Line legend (bottom)
    y_legend = -10
    line_types = ["NREM", "REM", "Wake"]
    line_colors = ["k", "b", "r"]
    for i, (color, e_type) in enumerate(zip(line_colors, line_types)):
        x = 60 * fs + 200 * fs * i
        ax.plot([x, x + 50 * fs], [y_legend] * 2, c=color, lw=0.8)
        ax.text(x + 25 * fs, y_legend - 3, e_type, fontsize=fz, c=color, ha="center", va="top")

    # Event legend
    event_types = ["Apnea", "Hypopnea"]
    label_colors_legend = ["b", "m"]
    for i, (color, e_type) in enumerate(zip(label_colors_legend, event_types)):
        x = 200 * fs * (len(line_types) + 0.5) + 300 * fs * (i + 1)
        ax.plot([x, x + 100 * fs], [y_legend] * 2, c=color, lw=2)
        ax.text(x + 50 * fs, y_legend - 3, e_type, fontsize=fz, ha="center", va="top")

    # Estimated LG legend (only when hooks are drawn)
    if em_data_loaded:
        lg_tag = "Estimated LG"
        lg_dur = 8 * 60 * fs
        lg_left = len_x - 4.75 * dx
        lg_right = lg_left + lg_dur
        ax.text(lg_left + lg_dur / 2, y_legend - 3, lg_tag, color="k", fontsize=fz - 1, ha="center", va="top")
        ax.plot([lg_left, lg_right], [y_legend] * 2, color="k", lw=0.5)
        ax.plot([lg_left] * 2, [y_legend - 1, y_legend], color="k", lw=0.5)
        ax.plot([lg_right] * 2, [y_legend - 1, y_legend], color="k", lw=0.5)

    # SS marker legend
    tag = "Detected SS\nbreathing oscillation"
    ss_x = len_x - 60 * fs * 2.5 - 2 * dx if not em_data_loaded else len_x - 60 * fs * 2.5 - 4 * dx
    ax.text(ss_x, y_legend - 3, tag, color="k", fontsize=fz - 1, ha="center", va="top")
    ax.text(ss_x, y_legend, "o", c="b", fontsize=fz, ha="center", va="bottom")

    # Duration scale bar
    duration = 5
    ax.plot([len_x - 60 * fs * duration, len_x], [y_legend] * 2, color="k", lw=1)
    ax.plot([len_x - 60 * fs * duration] * 2, [y_legend - 0.5, y_legend + 0.5], color="k", lw=1)
    ax.plot([len_x] * 2, [y_legend - 0.5, y_legend + 0.5], color="k", lw=1)
    ax.text(
        len_x - 60 * fs * (duration / 2),
        y_legend + 1,
        f"{duration} min",
        color="k",
        fontsize=fz,
        ha="center",
        va="bottom",
    )
    ax.text(
        len_x - 60 * fs * (duration / 2),
        y_legend - 1,
        "(abd RIP)",
        color="k",
        fontsize=8,
        ha="center",
        va="top",
    )

    # Save
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def _find_em_csv(hf5_file: str, em_csv_dir: str | None = None) -> str | None:
    """Search for a matching EM output CSV for a given HF5 recording.

    Looks in ``em_csv_dir`` (if provided) for CSV files whose name
    contains the patient hash from the HF5 filename. Returns the path
    if found, None otherwise.
    """
    if em_csv_dir is None or not os.path.isdir(em_csv_dir):
        return None
    patient_hash = hf5_file.split("_")[0]
    import glob

    matches = glob.glob(os.path.join(em_csv_dir, f"*{patient_hash}*"))
    if matches:
        return matches[0]
    # Also check for Study N.csv naming via CSV metadata
    return None


def main():
    hf5_dir = config.hf5_dir
    hf5_files = sorted([f for f in os.listdir(hf5_dir) if f.endswith(".hf5")])

    if not hf5_files:
        print(f"No HF5 files found in {hf5_dir}")
        return

    # Optional: directory containing EM output CSVs. If present, hooks
    # will be added to the full-night figures automatically.
    em_csv_dir = os.path.join(os.path.dirname(config.hf5_dir), "em_csvs")
    if not os.path.isdir(em_csv_dir):
        em_csv_dir = None
        print("No em_csvs/ directory found — figures will be generated without LG hooks.")
        print(f"  (To add hooks, place EM output CSVs in {os.path.join(os.path.dirname(config.hf5_dir), 'em_csvs')}/)")

    # Generate full-night figure for each example recording
    for hf5_file in hf5_files:
        hf5_path = os.path.join(hf5_dir, hf5_file)
        patient_id = hf5_file.split("_")[0][:8]
        em_csv = _find_em_csv(hf5_file, em_csv_dir)
        out_path = os.path.join(FIGURES_DIR, "full_night", f"full_night_{patient_id}.pdf")
        plot_full_night_from_hf5(hf5_path, out_path, em_csv_path=em_csv)


if __name__ == "__main__":
    main()
