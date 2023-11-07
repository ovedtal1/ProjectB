function [x_est] = SoftThresh(y,lambda)
x_est = y;
x_est(abs(y)<lambda) = 0;
x_est(y<-lambda) = y(y<-lambda) + lambda;
x_est(y>lambda) = y(y>lambda) - lambda;

end