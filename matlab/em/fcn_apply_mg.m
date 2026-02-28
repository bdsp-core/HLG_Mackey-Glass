function RMSE = fcn_apply_mg(V_o, V_max, gamma, tau, s, D, Arousal, u)
%FCN_APPLY_MG Apply Mackey-Glass model and compute RMSE.
%
%   RMSE = fcn_apply_mg(V_o, V_max, gamma, tau, s, D, Arousal, u)
%
%   Description:
%       Applies the Mackey-Glass ventilation model with given parameters,
%       scales the estimated ventilation to observed values (excluding
%       arousals), and returns the root-mean-square error.
%
%   Args:
%       V_o     (double array): Observed ventilation signal
%       V_max   (double): Maximum ventilation parameter
%       gamma   (double): Nonlinearity exponent in Mackey-Glass equation
%       tau     (double): Delay parameter (samples)
%       s       (double): Noise scale parameter
%       D       (double array): Drive signal (adjusted for alpha)
%       Arousal (double array): Arousal event contributions
%       u       (double array): Unit mask for valid samples (1=include, 0=exclude)
%
%   Returns:
%       RMSE (double): Root-mean-square error between scaled estimate and V_o
%
%   Reference:
%       Implementation based on Mackey-Glass equations for modeling
%       physiological control systems with EM algorithm for parameter
%       estimation (see README.md).
%

    % Compute ventilation
    V_o_es = fcn_state_space_loop(length(V_o), 0.05, V_max, gamma, tau, s, D, Arousal);

    % Scale ventilation
    temp_ = V_o ./ V_o_es;
    temp_ = temp_(~any(isnan(temp_) | isinf(temp_), 2), :);
    temp_(temp_ > 5) = [];
    Scale = mean(temp_);

    % Exclude arousals
    V_o_es_scaled = V_o_es;
    V_o_es_scaled(Arousal == 0) = V_o_es(Arousal == 0) * Scale;

    % Compute RMSE
    RMSE = norm(V_o_es_scaled(u == 1) - V_o(u == 1));
end
