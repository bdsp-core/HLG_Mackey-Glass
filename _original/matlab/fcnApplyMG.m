function RMSE = fcnApplyMG(V_o, V_max, gamma, tau, s, D, Arousal, u)
    % compute Ventilation
    V_o_es = fcnStateSpace_Loop_TN(length(V_o), 0.05, V_max, gamma, tau, s, D, Arousal);

    % scale Ventilation
    temp_ = V_o./V_o_es;
    temp_ = temp_( ~any( isnan( temp_ ) | isinf( temp_ ), 2 ),: );
    temp_(temp_>5) = [];
    Scale = (mean(temp_));

    % exclude Arousals
    V_o_es_scaled = V_o_es;
    V_o_es_scaled(Arousal==0) = V_o_es(Arousal==0)*Scale;

    % compute RMSE
    RMSE = norm(V_o_es_scaled(u==1) - V_o(u==1));
end

