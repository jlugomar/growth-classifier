# Development of computational models from clinical, medication, feeding and probiotics data

## Introduction

Preterm is defined as babies born before 37 weeks of pregnancy are completed.

In general, preterm babies require long stay and complex care in NICU. Also, clinical care 
is a significant challenge for healthcare workers in terms of feeding and nutrition. 

## Goals

The project objective is mainly to improve growth and health outcomes of preterm infants. 
This includes 3 steps:

* Prediction of specific outcomes 
    - Prediction of growth risk (development status, growth failure, etc)
    - Prediction of continuous values (body weight at 36 weeks post menstrual age, etc)

* Prediction of intervention outcomes
    - When to start enteral feeding?
    - When to give probiotics?

* Identify informative features

## Feature selection and engineering

Given information about across: patient, maternal, longitudinal, medication, feeding, and 
probiotics, we process the raw features into the following categories:

* Patient tab
    - Gestational age
    - Birth post menstrual age (PMA)
    - Birth weight
    - Birth z-score
    - Mode of delivery
    - Multiple gestation?
    - Gender

* Maternal tab
    - Maternal age

* Longitudinal tab
    - Classification: 
        - Biweekly bodyweight from day 1 until day 29
    - Regression:
        - Day 1 and day 7 bodyweight

* Medication tab
    - Had any medication? (boolean)
        - Week 1, week 2, ..., week 8
    - How many Ampicillin/Gentamicin? (integer)
        - Week 1, week 2, ..., week 8
    - How many other antibiotics? (integer)
        - Week 1, week 2, ..., week 8

* Feeding tab
    - Number of days per type (discrete 0-7) [breastmilk, donated milk, formula milk]
        - Week 1, week 2, ..., week 8
    - Quantity per type (float) [breastmilk, donated milk, formula milk]
        - Week 1, week 2, ..., week 8

* Probiotics tab
    - Had probiotics? (boolean)
        - Week 1, week 2, ..., week 8
    - How many Infloran? (integer)
        - Week 1, week 2, ..., week 8
    - How many LB2? (integer)
        - Week 1, week 2, ..., week 8

All features are then processed into feature sets which is a function of longitudinal 
periods. That is, a feature set with longitudinal period of day 1-15 will only use day 1 
and day 15 weight from longitudinal tab and will only use week 1 and week 2 features from 
medication, feeding, and probiotics tab. 

### Feature selection for minimal set of features

Given the full set of features mentioned above, the entire dataset is imputed, and the 
features are ranked based on the absolute value of the principal component vectors 
normalized by their explained variance. 

Then, the dataset with only the top feature is sent to the logistic regression classifier 
that reports classification accuracy. After that, lower-ranked features are being added 
each time and get a classification accuracy report. 

Finally, after using all features possible, the cutoff is selected on the first point where 
the classification accuracy reaches the top 5% among all the accuracies. 

## Label creation

There are two types of labels created: typical development and growth failure.

A non-typical developed infant satisfies any of the following conditions:

* [(discharge Z-score) - (birth Z-score)] < -1.2
* diagnosed as NEC
* diagnosed as Sepsis

All other infants then have label typical development.

---

An infant having growth failure satisfies the following condition:

* [(discharge Z-score) - (birth Z-score)] < -1.2

All other infants then have label non-growth failure

## Classification task

### Model used

There are two models used: 

* Random Forest
    - Number of trees: 250
    - Maximum depth: squared root of number of trees

* Logistic Regression
    - Maximum iteration: 600
    - C for regularization: 2
    - Regularization: L2
    - Classifier solver: L-bfgs 

### Data imputation

Since only random forest could use the partially-filled raw dataset for classification, 
we utilize K-nearest neighbors imputation with `k = 5` to fill the dataset. This k-NN 
imputer is modified from the k-NN imputer of scikit-learn with the following new behavior:

* If the feature is discrete, uses majority voting as the result (ties broken by distance
 of neighbors)
* If the feature is continuous, use the arithmetic average of all neighbors

### Model evaluation

The model is trained using 5-fold cross-validation, which means: 

1. The dataset is split into 5 equally-sized folds
2. Data balancing is performed for each fold. That means the data in that fold are balanced such that the number of positive and negative samples are the same.
3. All models are trained on 4 of the 5 folds and tested on the other fold
4. Repeat step 2-3 such that all folds are tested

With the results, we evaluate based on the following metrics:

* Confusion matrix
* Classification accuracy
* Receiver Operating Characteristic curve (ROC curve)
* Precision-Recall curve (PR curve)
* Area under ROC curve
* Area under PR curve

### Usage of code

##### Prerequisite

To run the classification code, a current MATLAB installation is needed. 
Also, the "MATLAB engine for python" need to be installed as well. 

All other dependencies could be found in the anaconda environment dump `misc/matlab.yml`.

##### Split datasheet

From the original Astarte datasheet, store each tab separately as csv files

##### Create label file

Use `src/create_labels.py` to generate label csv files. Refer to file header of the each 
python script for detailed usage. 

##### Run experiment

First, replace the filepath of feature files and label files on the top part 
of `src/classification.py`. Then, specify the experiment to run (i.e. what features to contain, 
what longitudinal period(s) to choose) on the bottom of the same file. More detailed explanation 
of the parameters are available in the function header of `run_experiment()` in that file. 

