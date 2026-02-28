function V_o_es = fcn_state_space_loop(K, L, V_max, gamma, tau, s, u, Arousal)
%FCN_STATE_SPACE_LOOP Run Mackey-Glass state-space ventilation model.
%
%   V_o_es = fcn_state_space_loop(K, L, V_max, gamma, tau, s, u, Arousal)
%
%   Description:
%       Implements the discrete-time Mackey-Glass equations for ventilation
%       estimation. Computes state x and ventilation V_o over K samples
%       with drive u and arousal contributions.
%
%   Args:
%       K       (int): Number of samples
%       L       (double): Loop gain / steady-state parameter
%       V_max   (double): Maximum ventilation parameter
%       gamma   (double): Nonlinearity exponent in Mackey-Glass equation
%       tau     (double): Delay parameter (samples)
%       s       (double): Noise scale parameter
%       u       (double array): Drive signal (inspiratory drive, possibly
%                               adjusted for alpha)
%       Arousal (double array): Arousal event contributions
%
%   Returns:
%       V_o_es (double array): Estimated ventilation signal (K x 1)
%
%   Reference:
%       Implementation based on Mackey-Glass equations for modeling
%       physiological control systems (see README.md).
%

    %% Run the Filtering Part
    x = zeros(K, 1);
    V_o_es = zeros(K, 1);
    V_o_es(1) = 0;
    eps = 10^-2;

    % Add noise
    nss2 = s * randn(K, 1) * sqrt(s);
    nss3 = s * randn(K, 1) * sqrt(s);

    % Compute Mackey-Glass equations
    for k = 1:K
        if k <= tau
            if k == 1
                x(k) = 0;
            else
                x(k) = L + nss2(k);
            end
        else
            x(k) = x(k-1) * (1 - V_o_es(k-tau)) + L + nss2(k);
            if x(k) < 0
                x(k) = eps;
            end
        end

        % Compute Vo(t)
        V_o_es(k) = (V_max * (x(k)^gamma) / (1 + x(k)^gamma)) * u(k) + Arousal(k) + nss3(k);

        % Set lower bound of ventilation to zero
        if V_o_es(k) < 0
            V_o_es(k) = 0;
        end
    end
end
