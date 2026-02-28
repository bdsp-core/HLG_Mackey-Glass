function LG = fcn_get_loop_gain(L, g, d)
%FCN_GET_LOOP_GAIN Compute loop gain from Mackey-Glass steady-state.
%
%   LG = fcn_get_loop_gain(L, g, d)
%
%   Description:
%       Calculates loop gain (LG), a measure of respiratory control system
%       stability, from the Mackey-Glass ventilation model. Compares
%       steady-state ventilation under normal conditions vs. obstruction
%       and release.
%
%   Args:
%       L (double): Loop gain / steady-state parameter
%       g (double): Gamma (nonlinearity exponent)
%       d (double): Obstruction factor (drive during obstruction, 0-1)
%
%   Returns:
%       LG (double): Loop gain = dvr/dvd, ratio of ventilation change
%                    on release to change during obstruction
%
%   Reference:
%       Implementation based on Mackey-Glass equations for modeling
%       physiological control systems (see README.md).
%

    % Ventilation at steady state
    xss = fcn_get_xss_a(L, g, 1);     % Steady state value of x
    vss = xss.^g ./ (1 + xss.^g);     % Ventilation at steady state

    % Ventilation at steady state during obstruction
    xd = fcn_get_xss_a(L, g, d);      % Steady state value of x
    vd = d * xd.^g ./ (1 + xd.^g);   % Ventilation at steady state
    dvd = abs(vss - vd);

    % Ventilation immediately after release of obstruction
    vr = xd.^g ./ (1 + xd.^g);       % Ventilation at steady state
    dvr = abs(vss - vr);

    LG = dvr / dvd;
end
