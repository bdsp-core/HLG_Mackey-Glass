function T = mainRun(study_num, dataset, input_folder, run, version)
    % extract patient data
    [path]  = filter_study(study_num, input_folder);
    T       = readtable(input_folder + path, 'VariableNamingRule', 'preserve');
    Fs      = T.Fs(1);
    w       = 5*Fs;     
    L       = 0.05;

    % init parameters
    gamma_init  = 0.5;      % "Controller gain"         [0.1, 0.5, 0.9]
    tau_init    = 15*Fs;    % Chemosensitivity delay    [10, 30, 50 sec]
    % w_init      = 5;      % Arousal Width             [3 - 10sec]
    % L_init      = 0.05;   % CO2 production rate       [0.01 - 0.1]

    % set all segment indices
    nrem_starts = T.nrem_starts(~isnan(T.nrem_starts));
    nrem_ends   = T.nrem_ends(~isnan(T.nrem_ends));
    rem_starts  = T.rem_starts(~isnan(T.rem_starts));
    rem_ends    = T.rem_ends(~isnan(T.rem_ends));
    assert(length(nrem_starts)==length(nrem_ends) && length(rem_starts)==length(rem_ends), 'uneven segment indices!' )

    %% Code pipeline
    % Add 6 columns to save the estimation results
    T.D_rem         = NaN(height(T),1); columnIndex_D_rem           = strcmp(T.Properties.VariableNames, 'D_rem');
    T.L_rem         = NaN(height(T),1); columnIndex_L_rem           = strcmp(T.Properties.VariableNames, 'L_rem');
    T.Alpha_rem     = NaN(height(T),1); columnIndex_Alpha_rem       = strcmp(T.Properties.VariableNames, 'Alpha_rem');
    T.LG_rem        = NaN(height(T),1); columnIndex_LG_rem          = strcmp(T.Properties.VariableNames, 'LG_rem');
    T.G_rem         = NaN(height(T),1); columnIndex_G_rem           = strcmp(T.Properties.VariableNames, 'G_rem');
    T.D_nrem        = NaN(height(T),1); columnIndex_D_nrem          = strcmp(T.Properties.VariableNames, 'D_nrem');
    T.L_nrem        = NaN(height(T),1); columnIndex_L_nrem          = strcmp(T.Properties.VariableNames, 'L_nrem');
    T.Alpha_nrem    = NaN(height(T),1); columnIndex_Alpha_nrem      = strcmp(T.Properties.VariableNames, 'Alpha_nrem');
    T.LG_nrem       = NaN(height(T),1); columnIndex_LG_nrem         = strcmp(T.Properties.VariableNames, 'LG_nrem');
    T.G_nrem        = NaN(height(T),1); columnIndex_G_nrem          = strcmp(T.Properties.VariableNames, 'G_nrem');
    T.rmse_Vo       = NaN(height(T),1); columnIndex_rmse_Vo         = strcmp(T.Properties.VariableNames, 'rmse_Vo');
    T.Vmax          = NaN(height(T),1); columnIndex_Vmax            = strcmp(T.Properties.VariableNames, 'Vmax');
    % double these arrays for overlapping windows
    T.Vo_est1       = NaN(height(T),1); columnIndex_Vo_est1         = strcmp(T.Properties.VariableNames, 'Vo_est1');
    T.Vo_est2       = NaN(height(T),1); columnIndex_Vo_est2         = strcmp(T.Properties.VariableNames, 'Vo_est2');
    T.Vo_est_scaled1 = NaN(height(T),1); columnIndex_Vo_est_scaled1 = strcmp(T.Properties.VariableNames, 'Vo_est_scaled1');
    T.Vo_est_scaled2 = NaN(height(T),1); columnIndex_Vo_est_scaled2 = strcmp(T.Properties.VariableNames, 'Vo_est_scaled2');
    T.Arousal1       = NaN(height(T),1); columnIndex_Arousal1       = strcmp(T.Properties.VariableNames, 'Arousal1');
    T.Arousal2       = NaN(height(T),1); columnIndex_Arousal2       = strcmp(T.Properties.VariableNames, 'Arousal2');

    % run over NREM then REM
    for rr = 1:2
        % select starts/ends of all 8-minute segment
        if rr==1
            starts = nrem_starts;
            ends = nrem_ends;
            tag = 'NREM';
        elseif rr==2
            starts = rem_starts;
            ends = rem_ends;
            tag = 'REM';
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
            ending = min(ending,height(T));
            T_seg = T(starting:ending, :);

            % apply EM algo on segment (T_seg)
            [upAlpha, upgamma, uptau, Vo_est, h, u_avg] = fcnEMAlgorithm_TN_realData(T_seg, w, L, gamma_init, tau_init, version);

            % set EM estimations
            G_est       = upgamma(end); 
            D_est       = uptau(end)/Fs; 
            L_est       = 0.05;      
            Alpha_est   = upAlpha(end);

            % compute LG based on gamma and avg. apnea height
            % disp(['Gamma: ' num2str(G_est)  ' Tau: ' num2str(D_est) ' U_avg: ' num2str(u_avg)])
            LG_est = fcnGetLoopGain2(L_est, G_est, u_avg);
            % disp([' LG: ' num2str(LG_est)])
            
            % set Arousal Events
            t_ar = find(T_seg.arousal_locs);
            t = 1:height(T_seg);
            Arousal = zeros(height(T_seg),1);
            for idx = 1:length(t_ar)
                square = fcnGet_unitFunction(t,t_ar(idx)-w/2) - fcnGet_unitFunction(t - w/2,t_ar(idx));
                Arousal = h(idx)*square' + Arousal;
            end

            % scale Ventilation
            temp = T_seg.Ventilation_ABD(Arousal==0) ./Vo_est(Arousal==0) ;
            temp = temp( ~any( isnan( temp ) | isinf( temp ), 2 ),: );
            temp(temp>5) = [];
            Scale = (mean(temp));
            Vd = Vo_est - Arousal;
            Vo_est_scaled = Vd*Scale + Arousal;

            % insert Ventilation/Arousal into Table
            if all(isnan(T{starting:ending, columnIndex_Vo_est1}))
                T(starting:ending, columnIndex_Vo_est1)        = table(Vo_est);
                T(starting:ending, columnIndex_Vo_est_scaled1) = table(Vo_est_scaled);
                T(starting:ending, columnIndex_Arousal1)       = table(Arousal);
            else
                assert(all(isnan(T{starting:ending, columnIndex_Vo_est2})))
                T(starting:ending, columnIndex_Vo_est2)        = table(Vo_est);
                T(starting:ending, columnIndex_Vo_est_scaled2) = table(Vo_est_scaled);
                T(starting:ending, columnIndex_Arousal2)       = table(Arousal);
            end
            
            % compute Ventilation RMSE
            rmse_dec = rms(T_seg.Ventilation_ABD - Vo_est_scaled);
            T(s, columnIndex_rmse_Vo)   = table(rmse_dec);
            T(s, columnIndex_Vmax)      = table(Scale);
            
            % insert estimated parameters into Table
            if rr==1
                T(s, columnIndex_D_nrem)        = table(D_est);
                T(s, columnIndex_L_nrem)        = table(L_est);
                T(s, columnIndex_Alpha_nrem)    = table(Alpha_est);
                T(s, columnIndex_G_nrem)        = table(G_est);
                T(s, columnIndex_LG_nrem)       = table(LG_est);
            elseif rr==2
                T(s, columnIndex_D_rem)     = table(D_est);
                T(s, columnIndex_L_rem)     = table(L_est);
                T(s, columnIndex_Alpha_rem) = table(Alpha_est);
                T(s, columnIndex_G_rem)     = table(G_est);
                T(s, columnIndex_LG_rem)    = table(LG_est);
            end
        end
        % close waitbar
        if run=="series"
            close(wb)
        end
    end
end