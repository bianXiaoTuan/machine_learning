function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%disp(size(X));
%disp(size(y));
%disp(size(Xval));
%disp(size(yval));

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

%C_vec = [1.0]';
%sigma_vec = [0.1]';

%disp(C_vec);
%disp(sigma_vec);

m_c = size(C_vec,1 );
m_sigma = size(sigma_vec,1 );
max_accuracy = 0.0;


for i = 1:m_c
    C_tmp = C_vec(i);
    for j = 1:m_sigma
        sigma_tmp = sigma_vec(j);
        
        model = svmTrain(X, y, C_tmp, @(x1, x2) gaussianKernel(x1, x2, sigma_tmp));
        
        predictions = svmPredict(model, Xval);
        
        accuracy = mean(double(predictions == yval));
        if accuracy >= max_accuracy
            max_accuracy = accuracy;
            C = C_tmp;
            sigma = sigma_tmp;
        end
    end
end

fprintf('×¼È·ÂÊ: %f\n', accuracy);
fprintf('C: %f\n', C);
fprintf('sigma: %f\n', sigma);

% =========================================================================

end
