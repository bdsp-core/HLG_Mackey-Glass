function path = filter_study(num, input_folder_path)
%FILTER_STUDY Locates a study CSV file by study number within an input folder.
%
% Description:
%   Searches for a CSV file matching the study number (with flexible naming
%   patterns: "Study N.csv", "StudyN.csv", "Study N .csv").
%
% Args:
%   num (double|string): Study identifier.
%   input_folder_path (string): Directory to search for study files.
%
% Returns:
%   path (string): Filename of matching study file, or unset if not found.
%
    files = dir(input_folder_path);
    check = 0;
    num = num2str(num);
    for i = 1:length(files)
        file = files(i).name;
        if contains(file, "Study" + num + ".csv") || ...
           contains(file, "Study " + num + ".csv") || ...
           contains(file, "Study " + num + " .csv")
            path = file;
            check = 1;
            break;
        end
    end
    if check == 0
        disp("No .csv file found for study: " + num);
    end
end
