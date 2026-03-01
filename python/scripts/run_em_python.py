#!/usr/bin/env python
"""
Run the EM algorithm on sleep study CSVs — pure Python replacement for MATLAB main.m.

Processes one or more study CSVs through the Mackey-Glass EM parameter
estimation pipeline, producing output CSVs with estimated LG, gamma, tau,
alpha, and reconstructed ventilation waveforms.

Usage:
    # Single study:
    python -m scripts.run_em_python --input data/em_example/Study_example.csv

    # All studies in a folder:
    python -m scripts.run_em_python --input-dir data/em_input/ --output-dir data/em_output/
"""

import argparse
import glob
import os
import time

import matplotlib

matplotlib.use("Agg")

from hlg.em.run_em import process_study


def main():
    parser = argparse.ArgumentParser(description="Run EM algorithm on study CSVs.")
    parser.add_argument("--input", type=str, help="Path to a single study CSV.")
    parser.add_argument("--input-dir", type=str, help="Directory of study CSVs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to same dir as input.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="non-smooth",
        choices=["smooth", "non-smooth"],
        help="Drive signal version.",
    )
    args = parser.parse_args()

    # Collect input files
    if args.input:
        input_files = [args.input]
    elif args.input_dir:
        input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    else:
        # Default: process the example file
        script_dir = os.path.dirname(__file__)
        example = os.path.join(script_dir, "..", "data", "em_example", "Study_example.csv")
        if os.path.exists(example):
            input_files = [example]
            print(f"No input specified — using example: {example}")
        else:
            parser.error("Provide --input or --input-dir")

    output_dir = args.output_dir

    for i, csv_path in enumerate(input_files):
        print(f"\n[{i + 1}/{len(input_files)}] Processing: {os.path.basename(csv_path)}")
        t0 = time.time()

        result_df = process_study(csv_path, version=args.version)

        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, os.path.basename(csv_path))
        else:
            base, ext = os.path.splitext(csv_path)
            out_path = base + "_em_output" + ext

        result_df.to_csv(out_path, index=False)
        elapsed = time.time() - t0
        print(f"  Saved: {out_path} ({elapsed:.1f}s)")

        # Print summary of estimated parameters
        for tag in ["nrem", "rem"]:
            lg_col = f"LG_{tag}"
            g_col = f"G_{tag}"
            d_col = f"D_{tag}"
            if lg_col in result_df.columns:
                valid = result_df[lg_col].dropna()
                if len(valid) > 0:
                    print(
                        f"  {tag.upper()}: {len(valid)} segments, "
                        f"median LG={valid.median():.2f}, "
                        f"median gamma={result_df[g_col].dropna().median():.2f}, "
                        f"median tau={result_df[d_col].dropna().median():.1f}s"
                    )


if __name__ == "__main__":
    main()
