function V_o_es = fcnStateSpace_Loop_TN(K, L, V_max, gamma, tau, s, u, Arousal)
    %% Run the Filtering Part
    x           = zeros(K,1);
    V_o_es      = zeros(K,1);
    V_o_es(1)   = 0;
    eps         = 10^-2;

    % add noise
    nss2 = s * randn(K,1) * sqrt(s);
    nss3 = s * randn(K,1) * sqrt(s);

    % compute Mackey-Glass equations
    for k=1:K
        if k<=tau
            if k == 1
                x(k) = 0;
            else
                x(k) = L+nss2(k);
            end
        else
            x(k) = x(k-1) * (1-V_o_es(k-tau)) + L + nss2(k);
            if x(k)<0
                x(k) = eps;
            end
        end
        % compute  Vo(t)
        V_o_es(k) = (V_max*(x(k)^gamma)/(1+x(k)^gamma)) * u(k) + Arousal(k) + nss3(k);
        
        % set lower bound of ventilation to zero
        if V_o_es(k) < 0
            V_o_es(k) = 0;
        end
    end
end

