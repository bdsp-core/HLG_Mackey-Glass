function T = recompute_LG_Run(study_num, dataset, input_folder, run, version)
    % extract patient data
    [path]  = filter_study(study_num, input_folder);
    T       = readtable(input_folder + path, 'VariableNamingRule', 'preserve');

    % set all segment indices
    nrem_starts = T.nrem_starts(~isnan(T.nrem_starts));
    nrem_ends   = T.nrem_ends(~isnan(T.nrem_ends));
    rem_starts  = T.rem_starts(~isnan(T.rem_starts));
    rem_ends    = T.rem_ends(~isnan(T.rem_ends));
    assert(length(nrem_starts)==length(nrem_ends) && length(rem_starts)==length(rem_ends), 'uneven segment indices!' )

    % Add 6 columns to save the estimation results
    T.LG_min_rem    = NaN(height(T),1); columnIndex_LG_min_rem          = strcmp(T.Properties.VariableNames, 'LG_min_rem');
    T.LG_min_nrem   = NaN(height(T),1); columnIndex_LG_min_nrem         = strcmp(T.Properties.VariableNames, 'LG_min_nrem');

    % run over NREM then REM
    for rr = 1:2
        % select starts/ends of all 8-minute segment
        if rr==1
            starts = nrem_starts;
            ends = nrem_ends;
            tag = 'nrem';
        elseif rr==2
            starts = rem_starts;
            ends = rem_ends;
            tag = 'rem';
        end

        % show progress
        if run=="series"
            wb = waitbar(0,'', 'Name',[dataset ' - ' tag], WindowStyle='docked');
        end

        % run over each 8-minute segment
        for s = 1:length(starts)
            % show progress bar
            if run=="series"
                msg = ['Study ' num2str(study_num)  ', segment '  num2str(s) '/' num2str(length(starts)) '..'];
                waitbar((s-1)/length(starts), wb, msg)
            end

            % cut segment from Table
            starting = starts(s);
            ending = ends(s)-1;
            if starting == 0
                starting = 1;
                ending = ending + 1;
            end
            T_seg = T(starting:min(ending,height(T)), :);

            % apply EM algo on segment (T_seg)
            LG_est = T_seg{1, "LG_"+tag};
            G_est = T_seg{1, "G_"+tag};

            % observation process
            if version=="smooth"  
                u = T.d_i_ABD_smooth;
            else
                u = T.d_i_ABD;
            end
        
            % compute largest u decrease
            u_min = min(T.d_i_ABD_smooth(u<1));

            % compute LG based on gamma and avg. apnea height
            LG_min = fcnGetLoopGain2(0.05, G_est, u_min);
            
            % insert estimated parameters into Table
            if rr==1
                T(starting:min(ending,height(T)),columnIndex_LG_min_nrem)       = table(LG_min);
            elseif rr==2
                T(starting:min(ending,height(T)),columnIndex_LG_min_rem)        = table(LG_min);
            end
        end
        % close waitbar
        if run=="series"
            close(wb)
        end
    end
end