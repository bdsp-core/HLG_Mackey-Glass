#!/usr/bin/env python
"""
Master script: generate all publication figures.

Runs every figure-generating script in sequence and saves all output
to the ``figures/`` directory. Each script creates its own subdirectory:

    figures/
    ├── cohort_boxplots/       LG / gamma / tau boxplots across cohorts
    ├── cpap_prediction/       ROC, PR, and calibration curves
    ├── ss_relationship/       SS-vs-LG scatter with polynomial regression
    ├── altitude/              Altitude spaghetti plots + histogram grid
    ├── stable_ss/             Per-recording SS plots + length histograms
    ├── em_segments/           Per-segment multi-panel EM figures
    └── full_night/            Full-night overview PDFs

Prerequisites:
    - Intermediate result CSVs in ``interm_Results/`` (produced by the
      EM extraction pipeline).
    - HDF5 recordings in the configured ``hf5_dir``.
    - CSV metadata tables in ``csv_files/``.

Usage:
    python -m scripts.generate_all_figures          # run all
    python -m scripts.generate_all_figures --only cohort cpap   # subset
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

# Each entry maps a short name to the script module path.
FIGURE_SCRIPTS: dict[str, str] = {
    "paper": "scripts.run_paper_figures",
    "cohort": "scripts.run_group_analysis",
    "cpap": "scripts.run_cpap_analysis",
    "ss_relationship": "scripts.run_ss_relationship",
    "altitude": "scripts.run_altitude_analysis",
    "stable_ss": "scripts.run_stable_ss",
}


def run_script(name: str, module_path: str) -> bool:
    """Import and run a single figure script, returning True on success."""
    print(f"\n{'=' * 60}")
    print(f"  Generating: {name}")
    print(f"{'=' * 60}")
    try:
        mod = importlib.import_module(module_path)
        t0 = time.time()
        mod.main()
        elapsed = time.time() - t0
        print(f"  Done ({elapsed:.1f}s)")
        return True
    except FileNotFoundError as e:
        print(f"  SKIPPED -- missing data: {e}")
        return False
    except Exception as e:
        print(f"  FAILED -- {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate all HLG figures.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(FIGURE_SCRIPTS.keys()),
        help="Generate only the specified figure set(s).",
    )
    args = parser.parse_args()

    targets = args.only if args.only else list(FIGURE_SCRIPTS.keys())

    print("HLG Figure Generator")
    print(f"Targets: {', '.join(targets)}")

    results: dict[str, bool] = {}
    for name in targets:
        results[name] = run_script(name, FIGURE_SCRIPTS[name])

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED/SKIPPED"
        print(f"  {name:20s} {status}")

    n_fail = sum(not v for v in results.values())
    if n_fail:
        print(f"\n{n_fail} figure set(s) failed or were skipped (likely missing data).")
        sys.exit(1)
    else:
        print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
