function [upAlpha, upgamma, uptau, V_o_est, h, u_min] = fcnEMAlgorithm_TN_realData(T, w, L, gamma_init, tau_init, version)
    rng('default')
    
    % set fixed parameters
    V_max   = 1; % fixed --> will be estimated at the end
    K       = height(T);
    s       = 10^-8;
    
    % observation process
    V_o = T.Ventilation_ABD;
    if version=="smooth"  
        u = T.d_i_ABD_smooth;
    else
        u = T.d_i_ABD;
    end

    % compute average u decrease
    % u_avg = mean(u(u<1));
    u_min = min(u);
    
    % set arousal locations
    dit = T.arousal_locs;
    
    % EM Algorithm
    Iter = 5;
    [upAlpha, upgamma, uptau, h] = fcnEMAlgorithm_TN_v5(K, L, V_max, gamma_init, tau_init, s, V_o, u, dit, Iter, w);
    
    % Arousal Events Estimation
    t_ar = find(T.arousal_locs);
    t = 1:height(T);
    Arousal_est = zeros(height(T),1);
    for id = 1:length(t_ar)
        square = fcnGet_unitFunction(t,t_ar(id)-w/2) - fcnGet_unitFunction(t - w/2,t_ar(id));
        Arousal_est = h(id)*square'+Arousal_est;
    end
    
    % Decoder
    Alpha = upAlpha(end);
    gamma = upgamma(end);
    tau   = uptau(end);
    D = u + Alpha*(1-u);
    V_o_est = fcnStateSpace_Loop_TN(K, L, V_max, gamma, tau, s, D, Arousal_est);
end