# Respiratory Dynamics Analysis Toolkit

## Overview
This MATLAB codebase implements algorithms for analyzing respiratory dynamics during sleep, with a focus on modeling ventilatory control using the Mackey-Glass equations and Expectation-Maximization (EM) algorithm. The package is designed to process respiratory data to identify and quantify physiological parameters related to sleep apnea and breathing control.

## License
This code is licensed under CC BY-NC 4.0 (Attribution-NonCommercial 4.0). Commercial use is prohibited. See the LICENSE file for full details.

## Key Features
- Modeling of respiratory dynamics using Mackey-Glass equations
- Parameter estimation via EM algorithm
- Analysis of arousal events during sleep
- Loop gain estimation for respiratory control stability assessment
- Separate analysis for REM and NREM sleep stages
- Support for parallel processing to speed up analysis

## Main Components

### Core Functions
- **fcnStateSpace_Loop_TN.m**: Implements the Mackey-Glass model for ventilation
- **fcnEMAlgorithm_TN_v5.m**: Core EM algorithm for parameter estimation
- **fcnEMAlgorithm_TN_realData.m**: Wrapper for applying EM to real respiratory data
- **fcnApplyMG.m**: Applies Mackey-Glass model with specific parameters and computes error
- **fcnArousalEvent.m**: Detects and quantifies arousal events
- **fcnGetLoopGain2.m**: Calculates loop gain (measure of control system stability)

### Utility Functions
- **fcnAdjustPath.m**: Adjusts file paths for cross-platform compatibility
- **fcnGetDbxPfx.m**: Returns the Dropbox prefix path based on the operating system
- **fcnGet_unitFunction.m**: Helper function for step responses
- **fcnGet_xss_a.m**: Calculates steady-state value for a given parameter set
- **fcnSetOutPath.m**: Sets up output paths for results
- **filter_study.m**: Locates and filters specific study files

### Runner Scripts
- **main.m**: Main entry point for batch processing studies
- **mainRun.m**: Processes a single study file
- **recompute_LG.m**: Script for recalculating loop gain values
- **recompute_LG_Run.m**: Helper function for recomputing loop gain for a specific study

## Usage
1. Configure the desired cohort(s) in `main.m`
2. Adjust parameters as needed:
   - `run`: Set to "parallel" for parallel processing or "series" for sequential
   - `version`: Set to "smooth" or "non-smooth" for different preprocessing approaches
3. Execute `main.m` to process all studies in the specified cohort(s)

Example:
```matlab
% In main.m
cohort = {'MGH_high_CAI_V2'; 'MGH_SS_OSA_V2'};
run = "parallel";
version = "non-smooth";
% Then run the script
```

## Data Format
Input data should be in CSV format with the following key columns:
- Ventilation_ABD: Abdominal ventilation signal
- d_i_ABD: Inspiratory drive (non-smooth)
- d_i_ABD_smooth: Inspiratory drive (smoothed)
- arousal_locs: Binary indicators of arousal locations
- nrem_starts, nrem_ends: Indices marking NREM sleep segments
- rem_starts, rem_ends: Indices marking REM sleep segments

## Output
The code generates CSV files with the original data plus additional columns:
- Parameter estimates (Alpha, gamma, tau)
- Loop gain estimates (LG_rem, LG_nrem)
- Estimated ventilation signals (Vo_est1, Vo_est2)
- Scaled ventilation estimates (Vo_est_scaled1, Vo_est_scaled2)
- Arousal estimations (Arousal1, Arousal2)
- Error metrics (rmse_Vo)

## Requirements
- MATLAB (developed and tested on R2019b or later)
- Parallel Computing Toolbox (optional, for parallel processing)

## Authors
This code was developed for research purposes in sleep respiratory analysis.

## References
The implementation is based on the Mackey-Glass equations for modeling physiological control systems, and uses EM algorithm for parameter estimation in the presence of unobserved variables.
