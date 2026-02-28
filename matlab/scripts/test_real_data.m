% test_real_data.m
% Test script for running EM algorithm on real patient data cohorts.
%
% Usage:
%   Run from MATLAB with scripts directory as current folder. Configure
%   cohort, version, and study_range at top of script.
%
% Dependencies:
%   Requires addpath('../em') and addpath('../utils') for function access.
%

addpath('../em');
addpath('../utils');

clear
close all hidden
clc
rng('default')
dbx_pfx = fcn_get_dbx_pfx();

% set running parameters
run             = "series";     % "series"  "parallel"
version         = "smooth";     % "smooth"  "non-smooth"
study_range     = 1:100;
% cohort        = {'MGH_HLG_OSA_V1'};
cohort = {'MGH_high_CAI_V1';'MGH_HLG_OSA_V1';'MGH_NREM_OSA_V1';'MGH_REM_OSA_V1';'MGH_SS_cases_V1', 'REDEKER_Heart_Failure_V1'};

% run over all cohorts
for co = 1:length(cohort)
    % set input folder
    tag = cohort{co};
    folder_path = fcn_adjust_path(dbx_pfx + "a_People_BIDMC\ThijsNassi\ThijsTemp\EM_input_csv_files\" + tag + "\");
    folder = dir(folder_path);

    % adjust study range per cohort
    cohort_study_range = study_range;
    if tag=="REDEKER_Heart_Failure_V1"
        cohort_study_range = cohort_study_range(cohort_study_range<=65);
    end

    % init parameters
    gamma_init  = 0.5;      % "Controller gain"         [0.1, 0.5, 0.9]
    tau_init    = 15;       % Chemosensitivity delay    [10, 30, 50 sec]
    L_init      = 0.05;     % CO2 production rate       [0.01 - 0.1]
    w_init      = 5;        % Arousal Width             [3 - 10sec]

    % set output folder (adjust path as needed for your setup)
    output_folder = fcn_adjust_path(dbx_pfx + "a_People_BIDMC\ThijsNassi\ThijsTemp\EM_output\" + version + "\" + tag + "\");

    % run over all studies
    if run=="parallel"
        delete(gcp('nocreate'));    % Close the parallel pool
        parpool;                    % Start the parallel pool
    end

    for study_num = cohort_study_range
        out_path = fcn_set_out_path(output_folder, study_num, false);
        if isempty(out_path)
            continue
        end
        T = main_run(study_num, tag, folder_path, run, version);
        writetable(T, out_path);
        disp(['--> Study ' num2str(study_num)  ' is finished'])
    end
end
