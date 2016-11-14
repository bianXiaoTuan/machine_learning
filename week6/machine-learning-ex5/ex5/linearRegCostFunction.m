function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%disp(size(X));    % 12 * 2
%disp(size(y));    % 12 * 1
%disp(size(theta));    % 2 * 1
%disp(lambda);    % 1

h = X * theta;
j = (h - y) .^ 2;
bias = theta(2:end) .^ 2;

J = sum(j) / (2 * m) + sum(bias) * lambda / (2 * m);

theta(1) = 0;
grad = X' * (h - y) / m + theta * lambda / m;

% =========================================================================

grad = grad(:);

end
