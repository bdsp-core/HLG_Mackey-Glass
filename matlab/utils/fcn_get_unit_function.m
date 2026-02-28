function y = fcn_get_unit_function(t, t0)
%FCN_GET_UNIT_FUNCTION Unit step function for step response construction.
%
% Description:
%   Returns 1 for t >= t0 and 0 for t < t0. Used as a building block for
%   square-wave responses in arousal event computation.
%
% Args:
%   t (double array): Time vector.
%   t0 (double): Step onset time.
%
% Returns:
%   y (logical/double array): Unit step output (1 where t >= t0, 0 elsewhere).
%
    y = (t >= t0);
end
