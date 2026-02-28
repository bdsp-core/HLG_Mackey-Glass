# MATLAB -- EM Parameter Estimation

This directory contains the Expectation-Maximization (EM) algorithm for estimating ventilatory control parameters from respiratory inductance plethysmography (RIP) signals.

## Directory Structure

```
matlab/
  em/              Core EM algorithm and Mackey-Glass model
  utils/           Helper functions (path setup, arousal events, etc.)
  scripts/         Entry-point scripts for batch processing
```

## Core Algorithm (`em/`)

| File | Description |
|------|-------------|
| `fcn_em_algorithm.m` | Core EM algorithm -- iteratively estimates hidden parameters by alternating E-step (state estimation via recursive Bayesian filter) and M-step (parameter optimization via log-likelihood maximization) |
| `fcn_em_algorithm_real_data.m` | Wrapper that applies the EM algorithm to real clinical RIP data with physiological constraints |
| `fcn_state_space_loop.m` | Mackey-Glass state-space model: simulates the closed-loop ventilatory control system |
| `fcn_apply_mg.m` | Applies Mackey-Glass model with specific parameters and computes the fit error (RMSE) |
| `fcn_get_loop_gain.m` | Computes loop gain from estimated parameters as VR/VD ratio |

## Utilities (`utils/`)

| File | Description |
|------|-------------|
| `fcn_arousal_event.m` | Generates arousal waveforms for the ventilatory model |
| `fcn_get_unit_function.m` | Unit step function helper |
| `fcn_get_xss_a.m` | Computes steady-state CO2 concentration for a given parameter set |
| `fcn_set_out_path.m` | Sets up output directory paths |
| `fcn_adjust_path.m` | Cross-platform path separator adjustment |
| `fcn_get_dbx_pfx.m` | Legacy Dropbox path resolver (update for your environment) |
| `filter_study.m` | Locates and filters specific study files |

## Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `main.m` | Main entry point -- batch processes all studies in configured cohorts |
| `main_run.m` | Processes a single study file (called by `main.m`) |
| `recompute_lg.m` | Re-computes loop gain from existing EM output |
| `recompute_lg_run.m` | Per-study helper for LG recomputation |
| `test_real_data.m` | Test script for running on individual recordings |

## Usage

1. Open MATLAB and navigate to `matlab/scripts/`
2. Edit `main.m` to configure:
   - `cohort` -- list of cohort names (e.g. `{'MGH_high_CAI_V2'; 'MGH_SS_OSA_V2'}`)
   - `run` -- `"parallel"` or `"series"`
   - `version` -- `"smooth"` or `"non-smooth"`
   - Data paths (update `fcn_get_dbx_pfx.m` in `utils/` for your environment)
3. Run `main.m`

## Input Format

CSV files with columns:
- `Ventilation_ABD` -- abdominal ventilation signal
- `d_i_ABD` / `d_i_ABD_smooth` -- inspiratory drive
- `arousal_locs` -- binary arousal indicators
- `nrem_starts`, `nrem_ends` -- NREM segment boundaries
- `rem_starts`, `rem_ends` -- REM segment boundaries

## Output

The EM algorithm appends columns to the input CSV:
- `Alpha`, `gamma`, `tau` -- estimated control parameters
- `LG_rem`, `LG_nrem` -- loop gain per sleep stage
- `Vo_est1`, `Vo_est2` -- modeled ventilation
- `Vo_est_scaled1`, `Vo_est_scaled2` -- scaled estimates
- `Arousal1`, `Arousal2` -- estimated arousal components
- `rmse_Vo` -- root mean square error of the fit

## Requirements

- MATLAB R2019b or later
- Parallel Computing Toolbox (optional, for `run = "parallel"`)
