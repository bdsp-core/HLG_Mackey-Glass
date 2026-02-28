function adjusted_path = fcnAdjustPath(path)
    if ~contains(pwd, "C:\")
        adjusted_path = replace(path, '\', '/');
    else
        adjusted_path = path;
    end
end