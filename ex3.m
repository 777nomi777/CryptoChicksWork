clear ; close all; clc


fprintf('Loading and Visualizing Data ...\n')


data = load('headaches.txt');
X = data(:,1:11); y = data(:, 12);
y
fprintf('Program paused. Press enter to continue.\n');
pause;


% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

[m, n] = size(X);
m
n


% Initialize fitting parameters


theta_t=zeros(n+1,1);

X_t =[ones(4,1) X];
y_t = ([1;0;1;0] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')
num_labels=4;
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

all_theta
%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, [0,0,1,0,0,1,1,0,1,1,1]);
size(all_theta)
pred
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
