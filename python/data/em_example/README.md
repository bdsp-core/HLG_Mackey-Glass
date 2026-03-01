# Example Data for Demo Scripts

This directory should contain CSV files for running the demo scripts.

## Required Files

| File | Description | Used by |
|------|-------------|---------|
| `Study_example.csv` | Input recording (ventilation, SpO2, sleep stage, etc.) | `run_full_pipeline_demo.py` |
| `Study_example_em_output.csv` | EM algorithm output (LG, gamma, tau per segment) | `run_full_pipeline_demo.py` |
| `Study_4e504ee4.csv` | High-LG patient recording | `run_figure1_demo.py`, `run_full_night_example.py` |
| `Study_4e504ee4_em_output.csv` | EM output for high-LG patient | `run_figure1_demo.py`, `run_full_night_example.py` |

## Obtaining Data

Data files are not included in the repository due to size and privacy restrictions.

Data are available at [bdsp.io](https://bdsp.io/) upon request to the corresponding author.

Once obtained, place the CSV files in this directory (`python/data/em_example/`).
