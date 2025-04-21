function dbx_pfx = fcnGetDbxPfx()
    if contains(pwd, "C:\")
        dbx_pfx = "C:\Users\Nassi DELL\KoGES Scoring Dropbox\";
        % dbx_pfx = "C:\Users\Nassi DELL\cdac Dropbox\";
    else
        dbx_pfx = "/media/cdac/hdd/cdac Dropbox/";
    end
end