function out_path = fcn_set_out_path(output_folder, study_num, overwrite)
%FCN_SET_OUT_PATH Sets up output path for study results CSV file.
%
% Description:
%   Creates the output folder if needed, constructs the output CSV path for
%   the given study number, and optionally skips if already processed.
%
% Args:
%   output_folder (string): Base directory for output files.
%   study_num (double|string): Study identifier.
%   overwrite (logical): If false, returns empty string when file exists.
%
% Returns:
%   out_path (string): Full path to output CSV, or '' if skip.
%
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    out_path = fcn_adjust_path(output_folder + "\Study " + study_num + ".csv");

    if exist(out_path, 'file') && overwrite == false
        disp(['*Study ' num2str(study_num) ' already processed']);
        out_path = '';
    end
end
