function [upAlpha,upgamma,uptau,h] = fcnEMAlgorithm_TN_v5(K, L, V_max, gamma, tau, s, V_o, u, dit, Iter, w)
    rng('default')
    
    %% EM Update Loop
    % Init Model Estimation Array 
    upAlpha = zeros(1,Iter);
    upgamma = zeros(1,Iter);
    uptau   = zeros(1,Iter);        
    
    %% Estimate Arousal Events
    [h, Arousal] = fcnArousalEvent(dit, K, V_o, w);
    V_o_es = fcnStateSpace_Loop_TN(length(V_o), L, V_max, gamma, tau, s, u, Arousal);
    % Arousal Events update
    Arousal_err = zeros(K,1); 
    Arousal_err(Arousal~=0) = V_o_es(Arousal~=0) - V_o(Arousal~=0);
    [h_diff, Arousal_dif] = fcnArousalEvent(dit, K, Arousal_err, w);
    Arousal = Arousal - Arousal_dif;
    temp = h - h_diff;
    [~, ind] = find(temp<0);
    if isempty(ind)
        h = temp;
    else
        temp(ind) = 0.0;
        h = temp;
    end

    %% set parameter range
    % Tau
    Fs = 10;
    eps_tau     = Fs;
    tau_min     = 5*Fs;
    tau_max     = 50*Fs;
    tau_range   = tau_min:eps_tau:tau_max;
    % Gamma
    eps_gamma   = 0.01; 
    gamma_min   = 0.1;
    gamma_max   = 2.0; 
    gamma_range = gamma_min:eps_gamma:gamma_max;
    % Alpha
    alpha_range = 0:0.25:1;
    
    %% Main Loop
    for iter=1:Iter    
        % preallocate parameters for range of Alpha
        idx_l = 1;
        RMSE        = zeros(1,length(alpha_range));
        tau_temp    = zeros(1,length(alpha_range));
        gamma_temp  = zeros(1,length(alpha_range));
    
        % run over all Alpha options
        for idx_alpha = alpha_range
            % adjust U(t) based on Alpha
            D = u + idx_alpha*(1-u);
    
            % find best Gamma
            idx = 1;
            gamma_RMSE = zeros(1,length(gamma_range));
            for idx_g = gamma_range
                % apply Mackey-Glass equations
                gamma_RMSE(idx) = fcnApplyMG(V_o, V_max, idx_g, tau, s, D, Arousal, u);
                idx = idx+1;
            end
            [~, idx] = min(gamma_RMSE);
            gamma_temp(idx_l) = gamma_range(idx);
            
            % find best Tau
            idx = 1;
            tau_RMSE = zeros(1,length(tau_range));
            for idx_t = tau_range
                % apply Mackey-Glass equations
                tau_RMSE(idx) = fcnApplyMG(V_o, V_max, gamma, idx_t, s, D, Arousal, u);
                idx = idx+1;
            end
            [~, idx] = min(tau_RMSE);
            tau_temp(idx_l) = tau_range(idx);
    
            % compute Ventilation based on best Gamma/Tau
            RMSE(idx_l) = fcnApplyMG(V_o, V_max, gamma_temp(idx_l), tau_temp(idx_l), s, D, Arousal, u);
            idx_l = idx_l+1;
        end
    
        % select best parameters across different Alpha
        [~, idx]    = min(RMSE);
        Alpha       = alpha_range(idx);
        gamma       = gamma_temp(idx);
        tau         = tau_temp(idx);
        
        % save estimated parameters
        upAlpha(iter) = Alpha;
        uptau(iter)   = tau;
        upgamma(iter) = gamma;
    end
    % do this <Iter> times..
end