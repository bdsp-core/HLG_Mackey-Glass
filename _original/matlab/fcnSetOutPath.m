function out_path = fcnSetOutPath(output_folder, study_num, overwrite)
    % check if outputfolder exists
    if ~exist(output_folder, 'dir')
       mkdir(output_folder)
    end

    % set out path
    out_path = fcnAdjustPath(output_folder + "\Study " + study_num + ".csv");
    
    % skip if already processed
    if exist(out_path, 'file') && overwrite==false
        disp(['*Study ' num2str(study_num)  ' already processed'])
        out_path = '';
    end
end