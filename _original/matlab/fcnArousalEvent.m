function [h,Arousal] = fcnArousalEvent(dit,K,Vref,w)
% Arousal Events
t_ar = find(dit);
t = 1:K;
h = zeros(1,length(t_ar));
Arousal = zeros(K,1);
for idx = 1:length(t_ar)
    square = fcnGet_unitFunction(t,t_ar(idx)-w/2) - fcnGet_unitFunction(t - w/2,t_ar(idx));
    h(idx) = max(square'.*Vref);
    Arousal = h(idx)*square'+Arousal;
end
