% Filename: randomforest.m
% Author: Siwei Xu
% Date: 09/17/2020
%
% Description: Helper function to use matlab random forest for prediction

function [m, n] = randomforest(xpath, ypath, testxpath, plot)

% read matrices from csv files
X = readmatrix(xpath);
y = readmatrix(ypath);
testX = readmatrix(testxpath);

% train the tree and test on test set
baggedEnsemble = fitcensemble(X, y, 'Method', 'Bag', 'Learners', 'tree', 'NumLearningCycles', 250, 'ScoreTransform', 'logit');
[y_pred, y_score] = baggedEnsemble.predict(testX);
p = str2double(plot);

% write to files
writematrix(y_pred, 'matlab_y_pred.csv');
writematrix(y_score, 'matlab_y_score.csv');

m = X;
n = y;