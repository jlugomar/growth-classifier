function impOOB = feature_importance(xpath, ypath)

f1 = ["GA", "Site A", "Birth PMA", "Birthweight", "36 Weeks PMA", "Mode of Delivery", "Multiple Gestation", "Gender", "Birth Z-Score", "Site B", "Site C", "Weight\_day\_1", "PMA\_day\_1", "Maternal Age", "medication\_week0\_has\_medication", "medication\_week0\_has\_antibiotic", "medication\_week0\_has\_others"];
f15 = ["GA", "Site A", "Birth PMA", "Birthweight", "36 Weeks PMA", "Mode of Delivery", "Multiple Gestation", "Gender", "Birth Z-Score", "Site B", "Site C", "Weight\_day\_1", "Weight\_day\_15", "PMA\_day\_1", "PMA\_day\_15", "Maternal Age", "medication\_week0\_has\_medication", "medication\_week0\_has\_antibiotic", "medication\_week0\_has\_others", "medication\_week1\_has\_medication", "medication\_week1\_has\_antibiotic", "medication\_week1\_has\_others", "medication\_week2\_has\_medication", "medication\_week2\_has\_antibiotic", "medication\_week2\_has\_others", "feeding\_week0\_number\_days\_breastmilk", "feeding\_week0\_number\_days\_donated", "feeding\_week0\_number\_days\_formula", "feeding\_week0\_qty\_breastmilk", "feeding\_week0\_qty\_donated", "feeding\_week0\_qty\_formula", "feeding\_week1\_number\_days\_breastmilk", "feeding\_week1\_number\_days\_donated", "feeding\_week1\_number\_days\_formula", "feeding\_week1\_qty\_breastmilk", "feeding\_week1\_qty\_donated", "feeding\_week1\_qty\_formula"];
f29 = ["GA", "Site A", "Birth PMA", "Birthweight", "36 Weeks PMA", "Mode of Delivery", "Multiple Gestation", "Gender", "Birth Z-Score", "Site B", "Site C", "Weight\_day\_1", "Weight\_day\_15", "Weight\_day\_29", "PMA\_day\_1", "PMA\_day\_15", "PMA\_day\_29", "Maternal Age", "medication\_week0\_has\_medication", "medication\_week0\_has\_antibiotic", "medication\_week0\_has\_others", "medication\_week1\_has\_medication", "medication\_week1\_has\_antibiotic", "medication\_week1\_has\_others", "medication\_week2\_has\_medication", "medication\_week2\_has\_antibiotic", "medication\_week2\_has\_others", "medication\_week3\_has\_medication", "medication\_week3\_has\_antibiotic", "medication\_week3\_has\_others", "medication\_week4\_has\_medication", "medication\_week4\_has\_antibiotic", "medication\_week4\_has\_others", "feeding\_week0\_number\_days\_breastmilk", "feeding\_week0\_number\_days\_donated", "feeding\_week0\_number\_days\_formula", "feeding\_week0\_qty\_breastmilk", "feeding\_week0\_qty\_donated", "feeding\_week0\_qty\_formula", "feeding\_week1\_number\_days\_breastmilk", "feeding\_week1\_number\_days\_donated", "feeding\_week1\_number\_days\_formula", "feeding\_week1\_qty\_breastmilk", "feeding\_week1\_qty\_donated", "feeding\_week1\_qty\_formula", "feeding\_week2\_number\_days\_breastmilk", "feeding\_week2\_number\_days\_donated", "feeding\_week2\_number\_days\_formula", "feeding\_week2\_qty\_breastmilk", "feeding\_week2\_qty\_donated", "feeding\_week2\_qty\_formula", "feeding\_week3\_number\_days\_breastmilk", "feeding\_week3\_number\_days\_donated", "feeding\_week3\_number\_days\_formula", "feeding\_week3\_qty\_breastmilk", "feeding\_week3\_qty\_donated", "feeding\_week3\_qty\_formula"];
f43 = ["GA", "Site A", "Birth PMA", "Birthweight", "36 Weeks PMA", "Mode of Delivery", "Multiple Gestation", "Gender", "Birth Z-Score", "Site B", "Site C", "Weight\_day\_1", "Weight\_day\_15", "Weight\_day\_29", "PMA\_day\_1", "PMA\_day\_15", "PMA\_day\_29", "Maternal Age", "medication\_week0\_has\_medication", "medication\_week0\_has\_antibiotic", "medication\_week0\_has\_others", "medication\_week1\_has\_medication", "medication\_week1\_has\_antibiotic", "medication\_week1\_has\_others", "medication\_week2\_has\_medication", "medication\_week2\_has\_antibiotic", "medication\_week2\_has\_others", "medication\_week3\_has\_medication", "medication\_week3\_has\_antibiotic", "medication\_week3\_has\_others", "medication\_week4\_has\_medication", "medication\_week4\_has\_antibiotic", "medication\_week4\_has\_others", "medication\_week5\_has\_medication", "medication\_week5\_has\_antibiotic", "medication\_week5\_has\_others", "medication\_week6\_has\_medication", "medication\_week6\_has\_antibiotic", "medication\_week6\_has\_others", "feeding\_week0\_number\_days\_breastmilk", "feeding\_week0\_number\_days\_donated", "feeding\_week0\_number\_days\_formula", "feeding\_week0\_qty\_breastmilk", "feeding\_week0\_qty\_donated", "feeding\_week0\_qty\_formula", "feeding\_week1\_number\_days\_breastmilk", "feeding\_week1\_number\_days\_donated", "feeding\_week1\_number\_days\_formula", "feeding\_week1\_qty\_breastmilk", "feeding\_week1\_qty\_donated", "feeding\_week1\_qty\_formula", "feeding\_week2\_number\_days\_breastmilk", "feeding\_week2\_number\_days\_donated", "feeding\_week2\_number\_days\_formula", "feeding\_week2\_qty\_breastmilk", "feeding\_week2\_qty\_donated", "feeding\_week2\_qty\_formula", "feeding\_week3\_number\_days\_breastmilk", "feeding\_week3\_number\_days\_donated", "feeding\_week3\_number\_days\_formula", "feeding\_week3\_qty\_breastmilk", "feeding\_week3\_qty\_donated", "feeding\_week3\_qty\_formula", "feeding\_week4\_number\_days\_breastmilk", "feeding\_week4\_number\_days\_donated", "feeding\_week4\_number\_days\_formula", "feeding\_week4\_qty\_breastmilk", "feeding\_week4\_qty\_donated", "feeding\_week4\_qty\_formula", "feeding\_week5\_number\_days\_breastmilk", "feeding\_week5\_number\_days\_donated", "feeding\_week5\_number\_days\_formula", "feeding\_week5\_qty\_breastmilk", "feeding\_week5\_qty\_donated", "feeding\_week5\_qty\_formula"];
f57 = ["GA", "Site A", "Birth PMA", "Birthweight", "36 Weeks PMA", "Mode of Delivery", "Multiple Gestation", "Gender", "Birth Z-Score", "Site B", "Site C", "Weight\_day\_1", "Weight\_day\_15", "Weight\_day\_29", "PMA\_day\_1", "PMA\_day\_15", "PMA\_day\_29", "Maternal Age", "medication\_week0\_has\_medication", "medication\_week0\_has\_antibiotic", "medication\_week0\_has\_others", "medication\_week1\_has\_medication", "medication\_week1\_has\_antibiotic", "medication\_week1\_has\_others", "medication\_week2\_has\_medication", "medication\_week2\_has\_antibiotic", "medication\_week2\_has\_others", "medication\_week3\_has\_medication", "medication\_week3\_has\_antibiotic", "medication\_week3\_has\_others", "medication\_week4\_has\_medication", "medication\_week4\_has\_antibiotic", "medication\_week4\_has\_others", "medication\_week5\_has\_medication", "medication\_week5\_has\_antibiotic", "medication\_week5\_has\_others", "medication\_week6\_has\_medication", "medication\_week6\_has\_antibiotic", "medication\_week6\_has\_others", "medication\_week7\_has\_medication", "medication\_week7\_has\_antibiotic", "medication\_week7\_has\_others", "medication\_week8\_has\_medication", "medication\_week8\_has\_antibiotic", "medication\_week8\_has\_others", "feeding\_week0\_number\_days\_breastmilk", "feeding\_week0\_number\_days\_donated", "feeding\_week0\_number\_days\_formula", "feeding\_week0\_qty\_breastmilk", "feeding\_week0\_qty\_donated", "feeding\_week0\_qty\_formula", "feeding\_week1\_number\_days\_breastmilk", "feeding\_week1\_number\_days\_donated", "feeding\_week1\_number\_days\_formula", "feeding\_week1\_qty\_breastmilk", "feeding\_week1\_qty\_donated", "feeding\_week1\_qty\_formula", "feeding\_week2\_number\_days\_breastmilk", "feeding\_week2\_number\_days\_donated", "feeding\_week2\_number\_days\_formula", "feeding\_week2\_qty\_breastmilk", "feeding\_week2\_qty\_donated", "feeding\_week2\_qty\_formula", "feeding\_week3\_number\_days\_breastmilk", "feeding\_week3\_number\_days\_donated", "feeding\_week3\_number\_days\_formula", "feeding\_week3\_qty\_breastmilk", "feeding\_week3\_qty\_donated", "feeding\_week3\_qty\_formula", "feeding\_week4\_number\_days\_breastmilk", "feeding\_week4\_number\_days\_donated", "feeding\_week4\_number\_days\_formula", "feeding\_week4\_qty\_breastmilk", "feeding\_week4\_qty\_donated", "feeding\_week4\_qty\_formula", "feeding\_week5\_number\_days\_breastmilk", "feeding\_week5\_number\_days\_donated", "feeding\_week5\_number\_days\_formula", "feeding\_week5\_qty\_breastmilk", "feeding\_week5\_qty\_donated", "feeding\_week5\_qty\_formula", "feeding\_week6\_number\_days\_breastmilk", "feeding\_week6\_number\_days\_donated", "feeding\_week6\_number\_days\_formula", "feeding\_week6\_qty\_breastmilk", "feeding\_week6\_qty\_donated", "feeding\_week6\_qty\_formula", "feeding\_week7\_number\_days\_breastmilk", "feeding\_week7\_number\_days\_donated", "feeding\_week7\_number\_days\_formula", "feeding\_week7\_qty\_breastmilk", "feeding\_week7\_qty\_donated", "feeding\_week7\_qty\_formula"];

X = readmatrix(xpath);
y = readmatrix(ypath);

baggedEnsemble = fitcensemble(X, y, 'Learners', 'tree', 'NumLearningCycles', 250, 'ScoreTransform', 'logit');

curr = f1;

%h = figure
impOOB = predictorImportance(baggedEnsemble);
%bar(impOOB);
%xticklabels(curr);
%xticks(1:(length(curr)));
%xlim([0 (length(curr)+1)]);
%xtickangle(45);
%xlabel('Features');
%ylabel('Importance');
%title('Feature Importance with Raw Dataset on Day 1-1');