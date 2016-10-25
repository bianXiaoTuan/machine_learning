function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Debug J
%{
disp(size(theta));
disp(size(X));
disp(size(y));

a = X * theta;
disp(size(a));

A = -1 * y .* log(sigmoid(a));
disp(size(A));

B = (1 - y) .* log(1 - sigmoid(a));
disp(size(B));

C = A - B;
disp(size(C));

J = sum(C) / m;
disp(J);
%}

% J
v = sigmoid(X * theta);
J = sum(-1 * y .* log(v) - (1 - y) .* log(1 - v)) / m;

% Debug Gradient
%{
disp(size(X));

a = sigmoid(X * theta);
disp(size(a));

b = a - y;
disp(size(b));

B = b';
disp(size(B));


c = B * X;
disp(size(c));

grad = c' / m;
disp(grad);
%}

% Gradient
grad = ((v - y)' * X)' / m;


% =============================================================

end
