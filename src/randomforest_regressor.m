function [m, n] = randomforest_regressor(xpath, ypath, testxpath, plot)

X = readmatrix(xpath);
y = readmatrix(ypath);
testX = readmatrix(testxpath);
%X(:, 1) = [];
%testX(:, 1) = [];
%y(:, 1) = [];
baggedEnsemble = fitrensemble(X, y, 'Method', 'Bag', 'Learners', 'tree', 'NumLearningCycles', 250);
y_pred = baggedEnsemble.predict(testX);
p = str2double(plot);
if p > 0.5
    h = figure
    impOOB = predictorImportance(baggedEnsemble);
    bar(h);
    title('Feature Importance Plot in Random Forest');
    xlabel('Features');
    ylabel('Importance');
    savefig(h, 'feature_importance_plot.fig');
end

writematrix(y_pred, 'matlab_y_pred.csv');

m = X;
n = y;