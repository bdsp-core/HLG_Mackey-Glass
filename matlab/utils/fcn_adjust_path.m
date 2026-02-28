function adjusted_path = fcn_adjust_path(path)
%FCN_ADJUST_PATH Adjusts file path separators for cross-platform compatibility.
%
% Description:
%   Converts backslashes to forward slashes when not running on Windows
%   (detected by absence of "C:\" in current directory).
%
% Args:
%   path (string): Input path (may contain '\' or '/').
%
% Returns:
%   adjusted_path (string): Path with appropriate separators for current OS.
%
    if ~contains(pwd, "C:\")
        adjusted_path = replace(path, '\', '/');
    else
        adjusted_path = path;
    end
end
