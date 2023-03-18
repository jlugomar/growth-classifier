# Filename: classification.py
# Author: Siwei Xu
# Date: 10/08/2020
#
# Usage: python classification.py
#
# Description: experiment runner code for classification tasks
# NOTE: Must install matlab for python plugin before running this code
#       and need to be run inside the src folder
#
#       Also, the clinical data need to be saved in csv format and
#       change the filepath in line 53-60

import matlab.engine
import numpy as np 
import pandas as pd 
import os
import time
from subprocess import call
import pickle
import matplotlib.pyplot as plt 

from sklearn.preprocessing import normalize
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

from generate_features import *
from util import *

# minimal set of features 
MINIMAL_FEATURES = dict()
MINIMAL_FEATURES['gf'] = dict()
MINIMAL_FEATURES['gf'][1] = [['Birthweight', 'Birth PMA', 'GA', 'Birth Z-Score', 'Gender', 'Multiple Gestation', 'Mode of Delivery', 'Maternal Age'], ['PMA_day_1', 'Weight_day_1'], [], []]
MINIMAL_FEATURES['gf'][15] = [['Birthweight', 'Birth PMA', 'GA'], ['PMA_day_1', 'PMA_day_15', 'Weight_day_1', 'Weight_day_15'], ['medication_week1_medication_other_antibiotics', 'medication_week2_medication_other_antibiotics'], ['feeding_week1_number_days_donated', 'feeding_week1_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week1_qty_formula', 'feeding_week2_number_days_breastmilk', 'feeding_week2_qty_breastmilk', 'feeding_week2_qty_donated', 'feeding_week2_qty_formula']]
MINIMAL_FEATURES['gf'][29] = [['Birthweight', 'Birth PMA', 'GA'], ['PMA_day_1', 'PMA_day_15', 'PMA_day_29', 'Weight_day_1', 'Weight_day_15', 'Weight_day_29'], [], ['feeding_week3_qty_breastmilk', 'feeding_week1_qty_breastmilk', 'feeding_week3_qty_donated', 'feeding_week1_qty_donated', 'feeding_week3_qty_formula', 'feeding_week1_qty_formula', 'feeding_week4_number_days_breastmilk', 'feeding_week2_qty_breastmilk', 'feeding_week4_number_days_formula', 'feeding_week2_qty_formula', 'feeding_week2_qty_donated', 'feeding_week4_qty_breastmilk', 'feeding_week3_number_days_formula', 'feeding_week4_qty_donated', 'feeding_week4_qty_formula']]
MINIMAL_FEATURES['td'] = dict()
MINIMAL_FEATURES['td'][1] = [['Birthweight', 'Birth PMA', 'GA', 'Birth Z-Score', 'Gender', 'Multiple Gestation', 'Mode of Delivery', 'Maternal Age'], ['PMA_day_1', 'Weight_day_1'], [], []]
MINIMAL_FEATURES['td'][15] = [['Birthweight', 'Birth PMA', 'GA'], ['PMA_day_1', 'PMA_day_15', 'Weight_day_1', 'Weight_day_15'], ['medication_week2_medication_other_antibiotics'], ['feeding_week1_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week1_qty_formula', 'feeding_week2_number_days_breastmilk', 'feeding_week2_qty_breastmilk', 'feeding_week2_qty_donated', 'feeding_week2_qty_formula']]
MINIMAL_FEATURES['td'][29] = [['Birthweight', 'Birth PMA', 'GA'], ['PMA_day_1', 'PMA_day_15', 'PMA_day_29', 'Weight_day_1', 'Weight_day_15', 'Weight_day_29'], ['medication_week3_medication_other_antibiotics', 'medication_week2_medication_other_antibiotics'], ['feeding_week3_number_days_formula', 'feeding_week1_qty_breastmilk', 'feeding_week3_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week3_qty_donated', 'feeding_week1_qty_formula', 'feeding_week3_qty_formula', 'feeding_week2_number_days_breastmilk', 'feeding_week4_number_days_breastmilk', 'feeding_week2_number_days_formula', 'feeding_week4_number_days_formula', 'feeding_week2_qty_breastmilk', 'feeding_week4_qty_breastmilk', 'feeding_week2_qty_donated', 'feeding_week4_qty_donated', 'feeding_week2_qty_formula', 'feeding_week4_qty_formula', 'feeding_week3_number_days_breastmilk']]


# define prefix, by default is the data folder
PREFIX = os.path.join('..', 'data')

# define all csv file paths
patient_fpath = os.path.join(PREFIX, '2020_07_9_patients.csv')
maternal_fpath = os.path.join(PREFIX, '2020_07_9_maternal_data.csv')
longitudinal_fpath = os.path.join(PREFIX, '2020_07_9_longitudinal.csv')
label_fpath = os.path.join(PREFIX, '2020_07_9_labels_flipped.csv')
medication_fpath = os.path.join(PREFIX, '2020_07_9_medication.csv')
feeding_fpath = os.path.join(PREFIX, '2020_07_9_feeding.csv')
has_probiotics_fpath = os.path.join(PREFIX, '2020_08_12_has_probiotics.csv')
probiotics_fpath = os.path.join(PREFIX, '2020_07_9_probiotics.csv')

# initialize matlab engine
eng = matlab.engine.start_matlab()
pwd = os.getcwd()

def run_experiment(feature_list = [], 
                   longitudinal_period = 1, 
                   use_feeding = True, 
                   use_medication = True, 
                   use_site_split = False, 
                   use_simple_cut = False, 
                   filter_sampling = False, 
                   use_probiotics = None, 
                   probiotics_fpath = None, 
                   exp_prefix_time = None, 
                   save_model = False, 
                   is_growth_failure = False):
    """
    Run the experiment for all predictors at a specific

    Parameters:
    feature_list         (list): The subset of features to use. Default to 
                                 empty (use all features)
    longitudinal_period   (int): end date of the longitudinal period in use
    use_feeding          (bool): Use feeding data. Default to true
    use_medication       (bool): Use medication data. Default to true
    use_site_split       (bool): Use site-based cross-validation instead of 
                                 the 5-fold cross-validation. Default to false
    use_simple_cut       (bool): Use 80/20 simple cut or now. Default to false
    filter_sampling      (bool): Use only sequenced patients. Default to false
    use_probiotics        (str): None for no probiotics; probiotics list 
                                 filepath for probiotics features. Default to 
                                 None.
    probiotics_fpath      (str): Filepath for actual probiotics data csv file.
                                 Default to None. 
    exp_prefix_time(int or str): prefix to the folder for better organization. 
                                 Default to the current timestamp. 
    save_model           (bool): Save the logisticRegression model or not. 
                                 Default to false.
    is_growth_failure    (bool): Is the task for TD/NTD or GF/NGF. Only used
                                 to generate labels for confusion matrices.
    """

    # experiment start printout
    print(f'Running experiment on lon_period={longitudinal_period}, use_feeding={use_feeding}, use_medication={use_medication}, use_site_split={use_site_split}, filter_sampling={filter_sampling}')

    # generate experiment folder and prefix
    if exp_prefix_time is None:
        exp_prefix = os.path.join('..', 'results', f'results_{time.time()}', f'feeding_{use_feeding}_medication_{use_medication}_site_{use_site_split}_filter_{filter_sampling}')
    else:
        exp_prefix = os.path.join('..', 'results', f'results_{exp_prefix_time}', f'feeding_{use_feeding}_medication_{use_medication}_site_{use_site_split}_filter_{filter_sampling}')
    os.makedirs(exp_prefix, exist_ok = True)
    
    # get preprocessed features
    features = preprocess_features(patient_fpath, 
                                   maternal_fpath, 
                                   longitudinal_fpath, 
                                   medication_fpath, 
                                   feeding_fpath, 
                                   label_fpath, 
                                   use_dc = False, 
                                   longitudinal_period = longitudinal_period, 
                                   use_feeding = use_feeding, 
                                   use_medication = use_medication, 
                                   use_probiotics = use_probiotics, 
                                   probiotics_fpath = probiotics_fpath, 
                                   use_day_features = False)

    # remove the 36 weeks pma features and fetch the labels
    features = features.drop(labels = '36 Weeks PMA', axis = 1)
    labels = features.merge(pd.read_csv(label_fpath), on='Astarte ID', how='inner')

    # use sequenced patients only if needed
    if filter_sampling:
        labels = filter_unsampled(labels)

    # filter features only to those patients
    labels = labels[['Astarte ID', 'TypDev']]
    features = features[features['Astarte ID'].isin(labels['Astarte ID'].tolist())]

    # make features and labels the same order and remove ID for imputation
    features = features.sort_values(by = 'Astarte ID').drop(labels = ['Astarte ID'], axis = 1)
    labels = labels.sort_values(by = 'Astarte ID')

    # save Astarte ID and labels as numpy array
    ids = labels['Astarte ID'].to_numpy().astype(np.int)
    labels = labels.drop('Astarte ID', axis = 1).to_numpy()

    if len(feature_list) > 0:
        features = features[feature_list]

    # save raw features as csv files
    save_raw_features(features, exp_prefix, longitudinal_period = longitudinal_period)

    # save feature list
    training_feature_list = features.columns.tolist()

    # impute all data
    random_miss, imputed_features = impute_features(features.to_numpy(), feature_list = training_feature_list)

    # save imputed features/labels in matlab matrix format for feature importance analysis
    X_header = ','.join([f'feature_{fidx}' for fidx in range(features.shape[1])])
    y_header = 'label'
    #np.savetxt(os.path.join(exp_prefix, f'matlab_train_x_imputed_{longitudinal_period}.csv'), 
    #           imputed_features, delimiter = ',', header = X_header, fmt = '%.5f')
    #np.savetxt(os.path.join(exp_prefix, f'matlab_train_y_imputed_{longitudinal_period}.csv'), 
    #           labels.astype(np.int), delimiter = ',', header = y_header, fmt = '%d')

    # save imputed features as csv files
    imputed_dict = {}
    imputed_dict['Astarte ID'] = ids.tolist()
    columns = features.columns.tolist()
    assert len(columns) == imputed_features.shape[1]
    for i in range(len(columns)):
        imputed_dict[columns[i]] = imputed_features[:, i]
    pd.DataFrame(imputed_dict).to_csv(os.path.join(exp_prefix, f'imputed_features_{longitudinal_period}.csv'))

    features = features.to_numpy()

    # save raw features in matlab matrix format for feature importance analysis
    #np.savetxt(os.path.join(exp_prefix, f'matlab_train_x_{longitudinal_period}.csv'), features, delimiter = ',', header = X_header, fmt = '%.5f')

    print(f'  Dataset size: {features.shape[0]}')

    # save logistic regression model as pickle file
    if save_model:
        print('  Saving model...')
        clf = LogisticRegression(max_iter = 600, C = 2, penalty = 'l2', solver = 'lbfgs', class_weight = 'balanced')
        clf.fit(imputed_features, labels)
        pickle.dump(clf, open(os.path.join(exp_prefix, f'logistic_model_{longitudinal_period}.pickle'), 'wb'))

    
    # 5-fold cross validation or site-specific split
    if use_simple_cut == False:
        if use_site_split:
            split = site_cross_validation(ids, patient_fpath)
        else:
            k_fold = KFold(n_splits = 5, shuffle = False)
            split = k_fold.split(features)
    

    total_num = features.shape[0]

    if use_simple_cut:
        train_idx = np.random.permutation(list(range(total_num)))
        train_num = int(total_num * 0.8)
        split = [(train_idx[:train_num], train_idx[train_num:])]

    # initialize lists and matrices
    logistic_mat = np.zeros((2, 2))
    randomforest_mat = np.zeros((2, 2))
    randomforest_raw_mat = np.zeros((2, 2))

    logistic_probas = []
    randomforest_probas = []
    randomforest_raw_probas = []

    labels_probas = []

    randomforest_preds = []
    randomforest_raw_preds = []
    logistic_preds = []

    idss = []

    coef = None

    # iterate through splits
    for fold_idx, (train_idx, test_idx) in enumerate(split):
        
        # obtain training and testing features and labels
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]
        imputer, train_features_imputed = impute_features(train_features, feature_list = training_feature_list)
        test_features_imputed = imputer.transform(test_features)

        # initialize classifiers
        logistic = LogisticRegression(max_iter = 600, 
                                      C = 2, 
                                      penalty = 'l2', 
                                      solver = 'lbfgs', 
                                      class_weight = 'balanced')
        
        # fit on all classifiers
        randomforest_raw_pred, randomforest_raw_proba = run_classifier(train_features, train_labels, test_features, eng)
        randomforest_pred, randomforest_proba = run_classifier(train_features_imputed, train_labels, test_features_imputed, eng)
        logistic_pred, logistic_proba = run_classifier(train_features_imputed, train_labels, test_features_imputed, logistic)

        # save logistic weight if needed
        if coef is None:
            coef = np.array(logistic.coef_)
        else:
            coef = np.add(coef, logistic.coef_)

        # record predictions and class probabilities
        randomforest_raw_preds.append(randomforest_raw_pred)
        randomforest_raw_probas.append(randomforest_raw_proba)

        randomforest_preds.append(randomforest_pred)
        randomforest_probas.append(randomforest_proba)

        logistic_preds.append(logistic_pred)
        logistic_probas.append(logistic_proba)

        # record confusion matrices
        logistic_mat = np.add(logistic_mat, confusion_matrix(test_labels, logistic_pred))
        randomforest_mat = np.add(randomforest_mat, confusion_matrix(test_labels, randomforest_pred))
        randomforest_raw_mat = np.add(randomforest_raw_mat, confusion_matrix(test_labels, randomforest_raw_pred))

        # record actual labels and ids for plotting purposes
        labels_probas.append(test_labels)
        idss.append(ids[test_idx])
        
        # generate split-specific confusion matrices and ROC 
        randomforest_tmp = confusion_matrix(test_labels, randomforest_pred)
        randomforest_raw_tmp = confusion_matrix(test_labels, randomforest_raw_pred)
        logistic_tmp = confusion_matrix(test_labels, logistic_pred)
        
        logistic_proba_tmp = logistic_proba[:, 1]
        randomforest_proba_tmp = randomforest_proba[:, 1]
        randomforest_raw_proba_tmp = randomforest_raw_proba[:, 1]

        randomforest_fpr, randomforest_tpr, _ = roc_curve(test_labels, randomforest_proba_tmp)
        randomforest_auc = auc(randomforest_fpr, randomforest_tpr)
        randomforest_raw_fpr, randomforest_raw_tpr, _ = roc_curve(test_labels, randomforest_raw_proba_tmp)
        randomforest_raw_auc = auc(randomforest_raw_fpr, randomforest_raw_tpr)
        logistic_fpr, logistic_tpr, _ = roc_curve(test_labels, logistic_proba_tmp)
        logistic_auc = auc(logistic_fpr, logistic_tpr)

        # print split-specific confusion matrices and ROC
        print(f'  Fold {fold_idx} with size {train_features.shape[0]}')
        print(f'  Random Forest Acc: {((randomforest_raw_tmp[0, 0] + randomforest_raw_tmp[1, 1]) / randomforest_raw_tmp.sum()):.3f}, AUROC: {randomforest_raw_auc:.3f}')
        print(f'  Random Forest w/ Imputed Acc: {((randomforest_tmp[0, 0] + randomforest_tmp[1, 1]) / randomforest_tmp.sum()):.3f}, AUROC: {randomforest_auc:.3f}')
        print(f'  Logistic Acc: {((logistic_tmp[0, 0] + logistic_tmp[1, 1]) / logistic_tmp.sum()):.3f}, AUROC: {logistic_auc:.3f}')

    # save logistic regression weight if needed
    if coef is not None:
        np.savetxt(os.path.join(exp_prefix, f'logistic_coef_day_{longitudinal_period}.csv'), coef, delimiter = ',')

    # save confusion matrices
    save_confusion_matrix(randomforest_mat, title = f'Random Forest on Imputed Dataset with Day 1-{longitudinal_period} Data, acc={((randomforest_mat[0, 0] + randomforest_mat[1, 1]) / np.sum(randomforest_mat)):.3f}', fpath = os.path.join(exp_prefix, f'cm_randomforest_impute_{longitudinal_period}.png'), is_growth_failure = is_growth_failure)
    save_confusion_matrix(randomforest_raw_mat, title = f'Random Forest on Raw Dataset with Day 1-{longitudinal_period} Data, acc={((randomforest_raw_mat[0, 0] + randomforest_raw_mat[1, 1]) / np.sum(randomforest_raw_mat)):.3f}', fpath = os.path.join(exp_prefix, f'cm_randomforest_raw_{longitudinal_period}.png'), is_growth_failure = is_growth_failure)
    save_confusion_matrix(logistic_mat, title = f'Logistic on Imputed Dataset with Day 1-{longitudinal_period} Data, acc={((logistic_mat[0, 0] + logistic_mat[1, 1]) / np.sum(logistic_mat)):.3f}', fpath = os.path.join(exp_prefix, f'cm_logistic_{longitudinal_period}.png'), is_growth_failure = is_growth_failure)

    idss = np.concatenate(idss, axis = 0)
    idss.dtype = int

    # save prediction and class probabilities
    np.savetxt(os.path.join(exp_prefix, f'random_forest_{longitudinal_period}.csv'), np.concatenate(randomforest_raw_probas, axis = 0), delimiter = ',')
    np.savetxt(os.path.join(exp_prefix, f'random_forest_impute_{longitudinal_period}.csv'), np.concatenate(randomforest_probas, axis = 0), delimiter = ',')
    np.savetxt(os.path.join(exp_prefix, f'logistic_impute_{longitudinal_period}.csv'), np.concatenate(logistic_probas, axis = 0), delimiter = ',')
    np.savetxt(os.path.join(exp_prefix, f'random_forest_pred_{longitudinal_period}.csv'), np.concatenate(randomforest_raw_preds, axis = 0), delimiter = ',')
    np.savetxt(os.path.join(exp_prefix, f'random_forest_impute_pred_{longitudinal_period}.csv'), np.concatenate(randomforest_preds, axis = 0), delimiter = ',')
    np.savetxt(os.path.join(exp_prefix, f'logistic_pred_impute_{longitudinal_period}.csv'), np.concatenate(logistic_preds, axis = 0), delimiter = ',')

    # concatenate statstics from all splits 
    labels_probas = np.concatenate(labels_probas, axis = 0)
    randomforest_probas = np.concatenate(randomforest_probas, axis = 0)[:, 1]
    randomforest_raw_probas = np.concatenate(randomforest_raw_probas, axis = 0)[:, 1]
    logistic_probas = np.concatenate(logistic_probas, axis = 0)[:, 1]

    np.savetxt(os.path.join(exp_prefix, f'labels_{longitudinal_period}.csv'), labels_probas, delimiter = ',')

    # calculate ROC curve and AUROC
    plt.clf()
    randomforest_fpr, randomforest_tpr, _ = roc_curve(labels_probas, randomforest_probas)
    randomforest_auc = auc(randomforest_fpr, randomforest_tpr)
    randomforest_raw_fpr, randomforest_raw_tpr, _ = roc_curve(labels_probas, randomforest_raw_probas)
    randomforest_raw_auc = auc(randomforest_raw_fpr, randomforest_raw_tpr)
    logistic_fpr, logistic_tpr, _ = roc_curve(labels_probas, logistic_probas)
    logistic_auc = auc(logistic_fpr, logistic_tpr)

    # plot ROC curve
    plt.plot(randomforest_fpr, randomforest_tpr, label = f'RandomForest w/ Imputed (auc={randomforest_auc:.3f})')
    plt.plot(randomforest_raw_fpr, randomforest_raw_tpr, label = f'RandomForest w/o Imputed (auc={randomforest_raw_auc:.3f})')
    plt.plot(logistic_fpr, logistic_tpr, label = f'LogisticRegression w/ Imputed (auc={logistic_auc:.3f})')
    plt.title(f'ROC Curves for All Classifiers with Day 1-{longitudinal_period} Features')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'lower right')
    plt.savefig(os.path.join(exp_prefix, f'roc_curve_1_{longitudinal_period}.png'), dpi = 500)

    # calculate precision-recall values
    plt.clf()
    randomforest_fpr, randomforest_tpr, threshold = precision_recall_curve(labels_probas, randomforest_probas)
    randomforest_auc = area_under_curve(randomforest_tpr, randomforest_fpr)
    randomforest_raw_fpr, randomforest_raw_tpr, _ = precision_recall_curve(labels_probas, randomforest_raw_probas)
    randomforest_raw_auc = area_under_curve(randomforest_raw_tpr, randomforest_raw_fpr)
    logistic_fpr, logistic_tpr, _ = precision_recall_curve(labels_probas, logistic_probas)
    logistic_auc = area_under_curve(logistic_tpr, logistic_fpr)

    # plot the PR curve
    plt.scatter(randomforest_tpr, randomforest_fpr, label = f'RandomForest w/ Imputed (auc={randomforest_auc:.3f})', s = 1)
    plt.scatter(randomforest_raw_tpr, randomforest_raw_fpr, label = f'RandomForest w/o Imputed (auc={randomforest_raw_auc:.3f})', s = 1)
    plt.scatter(logistic_tpr, logistic_fpr, label = f'LogisticRegression w/ Imputed (auc={logistic_auc:.3f})', s = 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'Precision-Recall Curves for All Classifiers with Day 1-{longitudinal_period} Features')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc = 'lower left')
    plt.savefig(os.path.join(exp_prefix, f'recall_precision_curve_1_{longitudinal_period}.png'), dpi = 500)

    # print statistics
    print(f'Day 1-{longitudinal_period}')
    print(f'Random Forest on Imputed Dataset acc={((randomforest_mat[0, 0] + randomforest_mat[1, 1]) / len(labels)):.3f}')
    print(f'Random Forest on Raw Dataset acc={((randomforest_raw_mat[0, 0] + randomforest_raw_mat[1, 1]) / len(labels)):.3f}')
    print(f'Logistic on Imputed Dataset acc={((logistic_mat[0, 0] + logistic_mat[1, 1]) / len(labels)):.3f}')
    print(f'RandomForest w/ Imputed auc={randomforest_auc:.3f}')
    print(f'RandomForest w/o Imputed auc={randomforest_raw_auc:.3f}')
    print(f'LogisticRegression w/ Imputed auc={logistic_auc:.3f}')

# running the experiments
if __name__ == '__main__':

    # define longitudinal periods to test
    lon_list = [1, 15, 29, 43, 57]

    # set default filename prefix
    curr_time = time.strftime('%m_%d_%Y', time.gmtime())

    # Note: to run individual experiment, only keep the corresponding 
    #       run_experiment calls

    # run for normal labels (TD/NTD)
    curr_time_ = str(curr_time) + f'_td_with_probiotics'
    label_fpath = os.path.join(PREFIX, '2020_07_9_labels_flipped.csv')
          
    for use_feeding in [True, False]:
        for use_medication in [True, False]:
            for lon in lon_list:
                run_experiment(is_growth_failure = False, 
                               save_model = True, 
                               use_probiotics = has_probiotics_fpath, 
                               probiotics_fpath = probiotics_fpath, 
                               longitudinal_period = lon, 
                               use_feeding = use_feeding, 
                               use_medication = use_medication, 
                               use_site_split = False, 
                               filter_sampling = False, 
                               exp_prefix_time = curr_time_)
    
    # run for growth failure    
    curr_time_ = str(curr_time) + f'_gf_with_probiotics'
    label_fpath = os.path.join(PREFIX, '2020_07_9_gf_labels_flipped.csv')

    for use_feeding in [True, False]:
        for use_medication in [True, False]:
            for lon in lon_list:
                run_experiment(is_growth_failure = True, 
                               save_model = True, 
                               use_probiotics = has_probiotics_fpath, 
                               probiotics_fpath = probiotics_fpath, 
                               longitudinal_period = lon, 
                               use_feeding = use_feeding, 
                               use_medication = use_medication, 
                               use_site_split = False, 
                               filter_sampling = False, 
                               exp_prefix_time = curr_time_)

    # run for normal labels (TD/NTD) WITHOUT probiotics features
    curr_time_ = str(curr_time) + f'_td_without_probiotics'
    label_fpath = os.path.join(PREFIX, '2020_07_9_labels_flipped.csv')
          
    for use_feeding in [True, False]:
        for use_medication in [True, False]:
            for lon in lon_list:
                run_experiment(is_growth_failure = False, 
                               save_model = True, 
                               use_probiotics = None, 
                               probiotics_fpath = None, 
                               longitudinal_period = lon, 
                               use_feeding = use_feeding, 
                               use_medication = use_medication, 
                               use_site_split = False, 
                               filter_sampling = False, 
                               exp_prefix_time = curr_time_)
    
    # run for growth failure WITHOUT probiotics features
    curr_time_ = str(curr_time) + f'_gf_without_probiotics'
    label_fpath = os.path.join(PREFIX, '2020_07_9_gf_labels_flipped.csv')

    for use_feeding in [True, False]:
        for use_medication in [True, False]:
            for lon in lon_list:
                run_experiment(is_growth_failure = True, 
                               save_model = True, 
                               use_probiotics = None, 
                               probiotics_fpath = None, 
                               longitudinal_period = lon, 
                               use_feeding = use_feeding, 
                               use_medication = use_medication, 
                               use_site_split = False, 
                               filter_sampling = False, 
                               exp_prefix_time = curr_time_) 