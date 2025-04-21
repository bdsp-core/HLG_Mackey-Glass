clear
close all hidden
clc
rng('default')
dbx_pfx = fcnGetDbxPfx();   % Set DropBox Prefix path

% set running parameters
run             = "series";       % "series"  "parallel"              
version         = "non-smooth";     % "smooth"  "non-smooth"
study_range     = 1:100;
% cohort        = {'MGH_high_CAI_V1'};
cohort = {'MGH_high_CAI_V1'; 'MGH_HLG_OSA_V1'; 'MGH_NREM_OSA_V1'; 'MGH_REM_OSA_V1'; 'MGH_SS_V1'; 'REDEKER_Heart_Failure_V1'; 'RT_Altitude_V1'};

% run over all cohorts
for co = 1:length(cohort)
    % set input folder
    dataset         = cohort{co};
    input_folder    = fcnAdjustPath(dbx_pfx + "a_People_BIDMC\ThijsNassi\ThijsTemp\Final Code Thijs\" + version + "\" + dataset + "\");
    output_folder   = fcnAdjustPath(dbx_pfx + "a_People_BIDMC\ThijsNassi\ThijsTemp\Final Code Thijs\ExperimentsLG" + version + "\" + dataset);

    % adjust study range for REDEKER cohort
    if dataset=="REDEKER_Heart_Failure_V1"
        study_range = study_range(study_range<=65);
    elseif dataset=="RT_Altitude_V1"
        study_range = study_range(study_range<=32);
    end

    % run over all studies
    for study_num = study_range
        % set ouput_folder 
        out_path = fcnSetOutPath(output_folder, study_num, false);
        if isempty(out_path)
            continue
        end

        % run EM on study
        Table = recompute_LG_Run(study_num, dataset, input_folder, run, version);

        % save EM output
        writetable(Table, out_path);
        disp(['--> Study ' num2str(study_num)  ' is finished'])
    end
end