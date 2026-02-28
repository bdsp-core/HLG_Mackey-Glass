clear
close all hidden
clc
rng('default')
dbx_pfx = fcnGetDbxPfx();

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
    folder_path = fcnAdjustPath(dbx_pfx + "a_People_BIDMC\ThijsNassi\ThijsTemp\EM_input_csv_files\" + tag + "\");
    folder = dir(folder_path);

    % adjust study range for REDEKER cohort
    if tag=="REDEKER_Heart_Failure_V1"
        study_range = study_range(study_range<=65);
    end

    % init parameters
    gamma_init  = 0.5;      % "Controller gain"         [0.1, 0.5, 0.9]
    tau_init    = 15;       % Chemosensitivity delay    [10, 30, 50 sec]
    L_init      = 0.05;     % CO2 production rate       [0.01 - 0.1]
    w_init      = 5;        % Arousal Width             [3 - 10sec]

    % run over all studies
    if run=="parallel"
        delete(gcp('nocreate'));    % Close the parallel pool
        parpool;                    % Start the parallel pool
    end
    
        
        % save EM output
        writetable(T, out_path);
        disp(['--> Study ' num2str(study_range(num))  ' is finished'])
    end
end