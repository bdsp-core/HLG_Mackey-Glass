function xss = fcnGet_xss_a(L,g,d)

x = linspace(0,50,10000); 
f = d*x.^(g+1)./(1+x.^g);
e = (f-L).^2; 

[~,jj] = min(e);
xss = x(jj); emin = e(jj); 

% figure(1); clf; 
% plot(x,e);
% xlabel('x');
% ylabel('e(x)');
% 
% hold on; plot(xss,emin,'*');
% drawnow