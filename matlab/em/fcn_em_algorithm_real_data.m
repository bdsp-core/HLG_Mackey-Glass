function [upAlpha, upgamma, uptau, V_o_est, h, u_min] = fcn_em_algorithm_real_data(T, w, L, gamma_init, tau_init, version)
%FCN_EM_ALGORITHM_REAL_DATA Apply EM algorithm to real respiratory data.
%
%   [upAlpha, upgamma, uptau, V_o_est, h, u_min] = fcn_em_algorithm_real_data(
%       T, w, L, gamma_init, tau_init, version)
%
%   Description:
%       Wrapper for applying the EM algorithm to real respiratory study
%       data. Extracts ventilation and drive signals, runs parameter
%       estimation, and reconstructs estimated ventilation with arousal
%       contributions.
%
%   Args:
%       T          (table): Study data with Ventilation_ABD, d_i_ABD,
%                          d_i_ABD_smooth, arousal_locs columns
%       w          (double): Arousal event window width (samples)
%       L          (double): Loop gain / steady-state parameter
%       gamma_init (double): Initial gamma for EM
%       tau_init   (double): Initial tau for EM
%       version    (char/string): "smooth" or "non-smooth" for drive signal
%
%   Returns:
%       upAlpha  (double array): Estimated Alpha per iteration
%       upgamma  (double array): Estimated gamma per iteration
%       uptau    (double array): Estimated tau per iteration
%       V_o_est  (double array): Estimated ventilation signal
%       h        (double array): Arousal event magnitudes
%       u_min    (double): Minimum drive value (u decrease)
%
%   Reference:
%       Implementation based on Mackey-Glass equations for modeling
%       physiological control systems with EM algorithm for parameter
%       estimation (see README.md).
%

    rng('default')

    % Set fixed parameters
    V_max = 1;  % Fixed --> will be estimated at the end
    K = height(T);
    s = 10^-8;

    % Observation process
    V_o = T.Ventilation_ABD;
    if version == "smooth"
        u = T.d_i_ABD_smooth;
    else
        u = T.d_i_ABD;
    end

    % Compute average u decrease
    u_min = min(u);

    % Set arousal locations
    dit = T.arousal_locs;

    % EM Algorithm
    Iter = 5;
    [upAlpha, upgamma, uptau, h] = fcn_em_algorithm(K, L, V_max, gamma_init, tau_init, s, V_o, u, dit, Iter, w);

    % Arousal Events Estimation
    t_ar = find(T.arousal_locs);
    t = 1:height(T);
    Arousal_est = zeros(height(T), 1);
    for id = 1:length(t_ar)
        square = fcn_get_unit_function(t, t_ar(id) - w/2) - fcn_get_unit_function(t - w/2, t_ar(id));
        Arousal_est = h(id) * square' + Arousal_est;
    end

    % Decoder
    Alpha = upAlpha(end);
    gamma = upgamma(end);
    tau = uptau(end);
    D = u + Alpha * (1 - u);
    V_o_est = fcn_state_space_loop(K, L, V_max, gamma, tau, s, D, Arousal_est);
end
