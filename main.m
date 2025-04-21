clear
close all hidden
clc
rng('default')
delete(gcp('nocreate'));    % Close the parallel pool
dbx_pfx = fcnGetDbxPfx();   % Set DropBox Prefix path

% set running parameters
run             = "parallel";       % "series"  "parallel"              
version         = "non-smooth";     % "smooth"  "non-smooth"
% cohort        = {'RT_Altitude_V2'};
% cohort = {'RT_Altitude_V2'; 'MGH_high_CAI_V2'; 'MGH_SS_OSA_V2'; 'MGH_NREM_OSA_V2'; 'MGH_REM_OSA_V2'; 'MGH_SS_range_V2'; 'REDEKER_Heart_Failure_V2'};
cohort = {'BDSP_CPAP_failure_V7', 'BDSP_CPAP_success_V7'};

% run over all cohorts
for co = 1:length(cohort)
    % set input folder
    dataset         = cohort{co};
    input_folder    = fcnAdjustPath(dbx_pfx + "Thijs Nassi\ThijsNassi\ThijsTemp\EM_input_csv_files\" + dataset + "\");
    output_folder   = fcnAdjustPath(dbx_pfx + "\Thijs Nassi\ThijsNassi\ThijsTemp\Final Code Revision\" + version + "\" + dataset);

    % adjust study range for REDEKER cohort
    study_range = 1:200;
    if dataset=="REDEKER_Heart_Failure_V1"
        study_range = study_range(study_range<=65);
    elseif contains(dataset, "Altitude")
        study_range = study_range(study_range<=32);
    elseif contains(dataset, "SINGLE")
        study_range = study_range(study_range<=1);
    end

    % run over all studies
    if run=="parallel"
        % Start the parallel pool
        if isempty(gcp('nocreate'))
            parpool; 
        end
        parfor study_num = study_range
            % set ouput_folder 
            out_path = fcnSetOutPath(output_folder, study_num, false);
            if isempty(out_path)
                continue
            end
            
            % run EM on study
            Table = mainRun(study_num, dataset, input_folder, run, version);

            % save EM output
            writetable(Table, out_path);
            disp(['--> Study ' num2str(study_num)  ' is finished'])
        end
    elseif run=="series"
        for study_num = study_range
            % set ouput_folder 
            out_path = fcnSetOutPath(output_folder, study_num, false);
            if isempty(out_path)
                continue
            end

            % run EM on study
            Table = mainRun(study_num, dataset, input_folder, run, version);

            % save EM output
            writetable(Table, out_path);
            disp(['--> Study ' num2str(study_num)  ' is finished'])
        end
    end
end