% Main
% Reset
clear; close all; clc;

% Read Data
pkg load io
data_train = xlsread('data_groundnut.xlsx');

X        = data_train;
X(:,end) = double(X(:,end) > 80000);
y        = X(:,end);
X        = X(:,1:(end-1));

% Re-classified
ind1    = find(y == 0);
y(ind1) = 2;

% Training Validation
rate_train = 0.75;
rate_test = 0.25;

Xtrain = X(1:round(rate_train*size(X,1)),:);
ytrain = y(1:round(rate_train*size(y,1)));
[m n] = size(Xtrain);

Xtest = X(round(rate_train*size(X,1))+1:end,:);
ytest = y(round(rate_train*size(X,1))+1:end);

% Neural Network parameters
n_inlayers  = n;
n_hidden    = 10;
n_outlayers = 2;

% Initialize parameters
epsilon  = 0.12;
theta1   = randomize(n_inlayers, n_hidden, epsilon); 
theta2   = randomize(n_hidden, n_outlayers, epsilon); 
nnparams = [theta1(:) ; theta2(:)]; 

% Test cost and gradient
lambda      = 0;
[cost grad] = nncostfunction(nnparams, n_inlayers, n_hidden, n_outlayers, ...
                             Xtrain, ytrain, lambda);

[numgrad grad] = checkNNGradients ;  

% Train neural network
costfunction = @(t) nncostfunction(t, n_inlayers, n_hidden, n_outlayers, ...
                                    Xtrain, ytrain, lambda);
options                  = optimset('MaxIter',300);
[nn_weights, cost_train] = fmincg(costfunction, nnparams, options); 

theta1nn = reshape(nn_weights(1:(n_hidden*(n_inlayers+1))), n_hidden, (n_inlayers+1));
theta2nn = reshape(nn_weights(((n_hidden*(n_inlayers+1))+1):end), n_outlayers, (n_hidden+1));   

% Fit
p1   = predict(theta1nn, theta2nn, Xtrain);
acc1 = mean(double(p1 == ytrain))*100; 

% Accuracy
p2   = predict(theta1nn, theta2nn, Xtest);
acc  = mean(double(p2 == ytest))*100;                       

