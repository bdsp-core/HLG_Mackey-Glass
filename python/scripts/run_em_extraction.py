#!/usr/bin/env python
"""
Run the EM output extraction pipeline.

Processes EM model CSV output files in parallel, computing per-segment
loop gain (LG), gain (gamma), and delay (tau) parameters. Results are
aggregated into cohort-level CSV files.

Usage:
    python -m scripts.run_em_extraction --dataset mgh --version SS_range
"""

import glob
import os

from hlg.config import config
from hlg.em.extraction import extract_EM_output


def main():
    dataset = "mgh"
    versions = ["SS_range"]
    ut_smooth = "non-smooth"

    for version in versions:
        hf5_folder = config.hf5_dir or "SS paper files/"
        bar_folder = f"./bars/{dataset}_{version}/"
        os.makedirs(bar_folder, exist_ok=True)

        input_folder = f"EM_input_csv_files/{ut_smooth}/{dataset.upper()}_{version}/"
        input_files = glob.glob(input_folder + "*.csv")

        csv_file = os.path.join(config.csv_dir, f"{dataset}_table1 100_{version}_cases.csv")
        if "CPAP" in version:
            csv_file = csv_file.replace("100_", "200_")

        interm_folder = f"./interm_Results/{ut_smooth}/group_analysis/{dataset}_{version}/"

        print(f"\n>> {version} <<")
        extract_EM_output(input_files, interm_folder, hf5_folder, version, dataset, csv_file, bar_folder)


if __name__ == "__main__":
    main()
