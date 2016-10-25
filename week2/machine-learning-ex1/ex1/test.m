% TEST

%% ======================= TEST 1: gradientDescent =======================
X = [1 2; 1 3; 1 4];
y = [4; 5; 6];
theta = [0; 0];
alpha = 0.01;

a = sum((X * theta - y) .* X(:, 1));
b = sum((X * theta - y) .* X(:, 2));

c = [a; b];
disp(c);