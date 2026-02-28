function [path] = filter_study(num, input_folder_path)
    % filter study
    files = dir(input_folder_path);
    check = 0;
    num = num2str(num);
    for i = 1:length(files)
        file = files(i).name;
        if contains(file, "Study" + num + ".csv") || contains(file, "Study " + num + ".csv") || contains(file, "Study " + num + " .csv")
            path = file;
            check = 1;
            break
        end
    end
    if check == 0
        disp("No .csv file found for study: " + num2str(num))
    end
end