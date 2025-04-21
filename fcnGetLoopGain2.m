function LG = fcnGetLoopGain2(L, g, d)

% ventilation at steady state
xss = fcnGet_xss_a(L, g, 1);    % steady state value of x
vss = xss.^g./(1+xss.^g);       % ventilation - at steady state

% ventilation at steady state during obstruction
xd = fcnGet_xss_a(L, g, d);     % steady state value of x
vd = d*xd.^g./(1+xd.^g);        % ventilation - at steady state
dvd = abs(vss-vd);

% ventilation immediately after release of obstruction
vr = xd.^g./(1+xd.^g);          % ventilation - at steady state
dvr = abs(vss-vr);

LG = dvr/dvd; 
