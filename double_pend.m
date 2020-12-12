function y=double_pend(t,x)
m=2;
l=1;
g=9.81;
k=5;

M=m*g*l/(2*m*l*l);
K=k/(m*l*l);
A=[1, 0, 0, 0;  0, 4/3, 0, cos(x(3)-x(1))/4;...
    0, 0, 1, 0; 0, cos(x(3)-x(1))/4, 0, 1/3;];

ikincisatir=3*M*sin(x(1))+x(4)*x(4)*sin(x(3)-x(1))*0.25+2*K*x(1)-2*K*x(3);
dorduncusatir=M*sin(x(3))-(x(2)*x(2)*sin(x(3)-x(1))*0.25)-K*(x(3)-x(1));


b=[x(2); ikincisatir;  x(4);dorduncusatir ;];


y=A\b;

