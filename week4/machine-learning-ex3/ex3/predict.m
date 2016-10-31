function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%disp(size(Theta1));  % 25 * 401
%disp(size(Theta2));  % 10 * 26

%disp(num_labels);  % 10
%disp(size(p));  % 5000 * 1

X = [ones(m, 1) X];
%disp(size(X));  % 5000 * 401

a = X * Theta1';
%disp(size(a));  % 5000 * 25

b = sigmoid(a);
%disp(size(b));  % 5000 * 25

b = [ones(size(b, 1), 1) b];
%disp(size(b));  % 5000 * 26

c = b * Theta2';
%disp(size(c));  % 5000 * 10

[max_v, index] = max(c');
%disp(size(max_v)); % 1 * 5000
%disp(size(index)); % 1 * 5000

p = index';

% =========================================================================
end
