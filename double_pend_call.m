[t,y]=ode45('double_pend',[0,10],[-0.366,0,0.994,0]);
plot(t,y(:,1))