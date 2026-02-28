function dbx_pfx = fcn_get_dbx_pfx()
%FCN_GET_DBX_PFX Returns the Dropbox root path based on current working directory.
%
% Description:
%   LEGACY PATH RESOLVER: Returns hardcoded Dropbox paths depending on
%   whether the current directory suggests Windows or Linux. Users should
%   update these paths for their own environment (e.g., via config file or
%   environment variable).
%
% Args:
%   (none)
%
% Returns:
%   dbx_pfx (string): Dropbox root path ending with '\' or '/'.
%
    if contains(pwd, "C:\")
        dbx_pfx = "C:\Users\Nassi DELL\KoGES Scoring Dropbox\";
    else
        dbx_pfx = "/media/cdac/hdd/cdac Dropbox/";
    end
end
