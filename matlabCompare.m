a = 1;
b = 100;

rosen = @(x) (a - x(1))^2 + b*(x(2)-x(1)^2)^2

x0 = [1.5,1.5];

options = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10);

fminsearch(rosen,x0,options);