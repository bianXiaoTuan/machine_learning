function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% disp(size(X));    % 5 * 3
% disp(size(Theta));    % 4 * 3
% disp(size(Y));    % 5 * 4
% disp(size(R));    % 5 * 4
% disp(num_movies);    % 5
% disp(num_users);    % 4
% disp(num_features);    % 3
% disp(lambda);    % 0
% disp(size(X_grad));    % 5 * 3
% disp(size(Theta_grad));    % 4 * 3

J = sum(sum(((X * Theta' - Y) .* R) .^ 2)) / 2; 
Theta_bias = lambda * sum(sum(Theta .^ 2)) / 2;
X_bias = lambda * sum(sum(X .^ 2)) / 2;

J = J + Theta_bias + X_bias;

j = X * Theta' - Y;    % 5 * 4

% X_grad
for m = 1:num_movies    % 5
	for k = 1:num_features    % 3
		X_grad(m, k) = 0;
		for u = 1:num_users    % 4
			if R(m, u) == 1
			    X_grad(m, k) = X_grad(m, k) + j(m, u) * Theta(u, k); 
	        end 
		end
		X_grad(m, k) = X_grad(m, k) + lambda * X(m, k);
	end
end	

% Theta_grad
for u = 1:num_users    % 4
	for k = 1:num_features    % 3
		Theta_grad(u, k) = 0;
		for m = 1:num_movies    % 5
			if R(m, u) == 1
			    Theta_grad(u, k) = Theta_grad(u, k) + j(m, u) * X(m, k);
	        end 
		end
		Theta_grad(u, k) = Theta_grad(u, k) + lambda * Theta(u, k);
	end
end	

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
