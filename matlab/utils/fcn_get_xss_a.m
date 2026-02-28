function xss = fcn_get_xss_a(L, g, d)
%FCN_GET_XSS_A Computes steady-state value of x for given parameter set.
%
% Description:
%   Finds the steady-state solution of the Mackey-Glass-like equation
%   f(x) = d*x^(g+1)/(1+x^g) = L by minimizing (f(x)-L)^2 over x.
%
% Args:
%   L (double): Target level (loop gain parameter).
%   g (double): Exponent parameter (gamma).
%   d (double): Scaling parameter (delta).
%
% Returns:
%   xss (double): Steady-state value of x where f(xss) ≈ L.
%
    x = linspace(0, 50, 10000);
    f = d * x.^(g+1) ./ (1 + x.^g);
    e = (f - L).^2;

    [~, jj] = min(e);
    xss = x(jj);
end
