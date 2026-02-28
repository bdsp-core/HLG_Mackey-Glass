function [h, Arousal] = fcn_arousal_event(dit, K, Vref, w)
%FCN_AROUSAL_EVENT Computes arousal events from derivative indicator times.
%
% Description:
%   Detects and quantifies arousal events by constructing square-wave
%   responses at each derivative indicator time and scaling by a reference
%   vector.
%
% Args:
%   dit (double array): Derivative indicator times (nonzero indices).
%   K (double): Number of time steps.
%   Vref (double array): Reference vector for scaling arousal magnitude.
%   w (double): Half-window width for square-wave construction.
%
% Returns:
%   h (double array): Arousal magnitudes at each event time.
%   Arousal (double array): Cumulative arousal profile (Kx1).
%
    t_ar = find(dit);
    t = 1:K;
    h = zeros(1, length(t_ar));
    Arousal = zeros(K, 1);
    for idx = 1:length(t_ar)
        square = fcn_get_unit_function(t, t_ar(idx) - w/2) - fcn_get_unit_function(t - w/2, t_ar(idx));
        h(idx) = max(square' .* Vref);
        Arousal = h(idx) * square' + Arousal;
    end
end
