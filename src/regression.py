# Filename: regression.py
# Author: Siwei Xu
# Date: 08/24/2020
#
# Usage: python regression.py
#
# Description: experiment runner code for regression tasks
# NOTE: must install matlab for python plugin before running this code
# and need to be run inside the src folder (currently)
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
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsRegressor

import torch
import torch.nn as nn

from generate_features import *
#from model import NNRegressor
from util import *

# define prefix, by default is the data folder
PREFIX = os.path.join('..', 'data')

patient_fpath = os.path.join(PREFIX, '2020_09_19_patients.csv')
maternal_fpath = os.path.join(PREFIX, '2020_07_9_maternal_data.csv')
longitudinal_fpath = os.path.join(PREFIX, '2020_07_9_longitudinal.csv')
label_fpath = os.path.join(PREFIX, '2020_09_19_regression_weight.csv')
medication_fpath = os.path.join(PREFIX, '2020_07_9_medication.csv')
feeding_fpath = os.path.join(PREFIX, '2020_07_9_feeding.csv')
has_probiotics_fpath = os.path.join(PREFIX, '2020_08_12_has_probiotics.csv')
probiotics_fpath = os.path.join(PREFIX, '2020_07_9_probiotics.csv')
b_fpath = os.path.join(PREFIX, 'boys_weight_parameters.tsv')
g_fpath = os.path.join(PREFIX, 'girls_weight_parameters.tsv')

eng = matlab.engine.start_matlab()
pwd = os.getcwd()

def run_experiment(use_dc = False, 
                   longitudinal_period = 1, 
                   use_feeding = True, 
                   use_medication = True, 
                   use_site_split = False, 
                   filter_sampling = False, 
                   use_probiotics = None, 
                   probiotics_fpath = None, 
                   exp_prefix_time = None, 
                   save_model = False):
    """
    Run the experiment

    Parameters:
    use_dc               (bool): Use d/c related features. Default to false
    longitudinal_period  (int): end date of the longitudinal period in use
    use_feeding          (bool): Use feeding data. Default to true
    use_medication       (bool): Use medication data. Default to true
    use_site_split       (bool): Use site-based cross-validation instead of 
                                 the 5-fold cross-validation. Default to false
    filter_sampling      (bool): Use only sequenced patients. Default to false
    use_probiotics        (str): None for no probiotics; probiotics list 
                                 filepath for probiotics features. Default to 
                                 None. 
    exp_prefix_time(int or str): prefix to the folder for better organization. 
                                 Default to the current timestamp. 
    save_model           (bool): Save the logisticRegression model or not. 
                                 Default to false.
    """
    print(f'Running REGRESSION experiment on lon_period={longitudinal_period}, use_feeding={use_feeding}, use_medication={use_medication}, use_site_split={use_site_split}, filter_sampling={filter_sampling}')

    # generate experiment folder
    if exp_prefix_time is None:
        exp_prefix = os.path.join('..', 'results', f'regression_results_{time.time()}', f'feeding_{use_feeding}_medication_{use_medication}_site_{use_site_split}_filter_{filter_sampling}')
    else:
        exp_prefix = os.path.join('..', 'results', f'regression_results_{exp_prefix_time}', f'feeding_{use_feeding}_medication_{use_medication}_site_{use_site_split}_filter_{filter_sampling}')
    os.makedirs(exp_prefix, exist_ok = True)
    
    # get features
    features = preprocess_features(patient_fpath, 
                                   maternal_fpath, 
                                   longitudinal_fpath, 
                                   medication_fpath, 
                                   feeding_fpath, 
                                   label_fpath, 
                                   use_dc = use_dc, 
                                   longitudinal_period = longitudinal_period, 
                                   use_feeding = use_feeding, 
                                   use_medication = use_medication, 
                                   use_probiotics = use_probiotics, 
                                   probiotics_fpath = probiotics_fpath, 
                                   is_regression = True)

    # remove the 36 weeks pma features

    labels = features.merge(pd.read_csv(label_fpath), on='Astarte ID', how='inner')

    features = features.drop(labels = '36 Weeks PMA', axis = 1)

    # use sequenced patients if needed
    if filter_sampling:
        labels = filter_unsampled(labels)

    # filter features only to those patients
    labels = labels[['Astarte ID', 'TypDev', 'Gender', '36 Weeks PMA']]
    features = features[features['Astarte ID'].isin(labels['Astarte ID'].tolist())]

    # make features and labels the same order
    features = features.sort_values(by = 'Astarte ID').drop(labels = ['Astarte ID'], axis = 1)
    labels = labels.sort_values(by = 'Astarte ID')

    if 'weight' in label_fpath:
        genders = np.array(labels['Gender'].tolist())
        #pma_days = np.array([round(x) for x in labels['36 Weeks PMA'].tolist()])
        pma_days = np.array([252 for x in labels['36 Weeks PMA'].tolist()])
        zscores = to_zscore(labels['TypDev'].tolist(), genders, pma_days, b_fpath, g_fpath)

    # save Astarte ID and labels as numpy array
    ids = labels['Astarte ID'].to_numpy().astype(np.int)
    labels = labels.drop(labels = ['Astarte ID', 'Gender', '36 Weeks PMA'], axis = 1).to_numpy()

    # save raw and imputed features as csv files
    save_raw_features(features, exp_prefix, longitudinal_period = longitudinal_period)

    # impute all data
    random_miss, imputed_features = impute_features(features.to_numpy())

    # save in matlab matrix format for feature importance analysis
    X_header = ','.join([f'feature_{fidx}' for fidx in range(features.shape[1])])
    y_header = 'label'
    np.savetxt(os.path.join(exp_prefix, f'matlab_train_x_imputed_{longitudinal_period}.csv'), normalize(imputed_features, norm = 'l2'), delimiter = ',', header = X_header, fmt = '%.5f')
    np.savetxt(os.path.join(exp_prefix, f'matlab_train_y_imputed_{longitudinal_period}.csv'), labels, delimiter = ',', header = y_header, fmt = '%.5f')

    # save imputed features
    imputed_dict = {}
    imputed_dict['Astarte ID'] = ids.tolist()
    columns = features.columns.tolist()
    assert len(columns) == imputed_features.shape[1]
    for i in range(len(columns)):
        imputed_dict[columns[i]] = imputed_features[:, i]
    pd.DataFrame(imputed_dict).to_csv(os.path.join(exp_prefix, f'imputed_features_{longitudinal_period}.csv'))

    # save features to numpy array
    features = features.to_numpy()

    np.savetxt(os.path.join(exp_prefix, f'matlab_train_x_{longitudinal_period}.csv'), features, delimiter = ',', header = X_header, fmt = '%.5f')

    print(f'Dataset size: {features.shape[0]}')

    if save_model:
        print('Saving model...')
        if 'weight' in label_fpath:
            lasso = Lasso(max_iter = 100000, alpha = 5)
            elasticnet = ElasticNet(max_iter = 100000, alpha = 5, l1_ratio = 0.3)
        else:
            lasso = Lasso(max_iter = 100000, alpha = 0.5, tol = 1E-4)
            elasticnet = ElasticNet(max_iter = 100000, alpha = 0.5, l1_ratio = 0.3, tol = 1E-4)
        lasso.fit(imputed_features, labels)
        elasticnet.fit(imputed_features, labels)
        pickle.dump(lasso, open(os.path.join(exp_prefix, f'lasso_model_{longitudinal_period}.pickle'), 'wb'))
        pickle.dump(elasticnet, open(os.path.join(exp_prefix, f'elasticnet_model_{longitudinal_period}.pickle'), 'wb'))

    # 5-fold cross validation
    if use_site_split:
        split = site_cross_validation(ids, patient_fpath)
    else:
        k_fold = KFold(n_splits = 5)
        split = k_fold.split(features)

    total_num = features.shape[0]

    labels_probas = []

    lasso_preds = []
    elasticnet_preds = []
    nn_preds = []
    knn_preds = []

    lasso_r2s = []
    elasticnet_r2s = []
    nn_r2s = []
    knn_r2s = []

    lasso_lowers, lasso_uppers = [], []
    lasso_pi_lowers, lasso_pi_uppers = [], []
    elasticnet_lowers, elasticnet_uppers = [], []
    elasticnet_pi_lowers, elasticnet_pi_uppers = [], []

    lasso_lowers2, lasso_uppers2 = [], []
    lasso_pi_lowers2, lasso_pi_uppers2 = [], []
    elasticnet_lowers2, elasticnet_uppers2 = [], []
    elasticnet_pi_lowers2, elasticnet_pi_uppers2 = [], []

    lasso_diffs, lasso_diffs2, lasso_pi_diffs, lasso_pi_diffs2 = [], [], [], []
    elasticnet_diffs, elasticnet_diffs2, elasticnet_pi_diffs, elasticnet_pi_diffs2 = [], [], [], []

    if 'weight' in label_fpath:
        lasso_zscore_lowers, lasso_zscore_uppers = [], []
        lasso_zscore_pi_lowers, lasso_zscore_pi_uppers = [], []
        elasticnet_zscore_lowers, elasticnet_zscore_uppers = [], []
        elasticnet_zscore_pi_lowers, elasticnet_zscore_pi_uppers = [], []

        lasso_zscore_lowers2, lasso_zscore_uppers2 = [], []
        lasso_zscore_pi_lowers2, lasso_zscore_pi_uppers2 = [], []
        elasticnet_zscore_lowers2, elasticnet_zscore_uppers2 = [], []
        elasticnet_zscore_pi_lowers2, elasticnet_zscore_pi_uppers2 = [], []

        lasso_zscore_diffs, lasso_zscore_diffs2, lasso_zscore_pi_diffs, lasso_zscore_pi_diffs2 = [], [], [], []
        elasticnet_zscore_diffs, elasticnet_zscore_diffs2, elasticnet_zscore_pi_diffs, elasticnet_zscore_pi_diffs2 = [], [], [], []

    test_labels_zscores = []
    lasso_zscore_preds = []
    elasticnet_zscore_preds = []

    idss = []

    coef = None

    for fold_idx, (train_idx, test_idx) in enumerate(split):
        
        # obtain training and testing features and labels
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]

        if 'weight' in label_fpath:
            train_genders = genders[train_idx]
            train_pma_days = pma_days[train_idx]
            test_genders = genders[test_idx]
            test_pma_days = pma_days[test_idx]

            train_labels_zscore = zscores[train_idx]
            test_labels_zscore = zscores[test_idx]

        imputer, train_features_imputed = impute_features(train_features)
        test_features_imputed = imputer.transform(test_features)

        if 'weight' in label_fpath:
            lasso = Lasso(max_iter = 100000, alpha = 5)
            elasticnet = ElasticNet(max_iter = 100000, alpha = 5, l1_ratio = 0.3)
        else:
            lasso = Lasso(max_iter = 100000, alpha = 0.5, tol = 1E-4)
            elasticnet = ElasticNet(max_iter = 100000, alpha = 0.5, l1_ratio = 0.3, tol = 1E-4)

        nn = eng#NNRegressor(train_features.shape[1], [128, 1])
        knn = KNeighborsRegressor(n_neighbors = 7, weights = 'distance', n_jobs = 8)
        
        # fit on all regressors
        test_x_new = np.array(test_features_imputed)
        test_y_new = np.array(test_labels)
        lasso_pred, lasso_r2 = run_regressor(train_features_imputed, train_labels, test_features_imputed, test_labels, lasso)
        elasticnet_pred, elasticnet_r2 = run_regressor(train_features_imputed, train_labels, test_x_new, test_y_new, elasticnet)
        nn_pred, nn_r2 = run_regressor(train_features_imputed, train_labels, test_features_imputed, test_labels, nn)
        knn_pred, knn_r2 = run_regressor(train_features_imputed, train_labels, test_features_imputed, test_labels, knn)

        if 'weight' in label_fpath:
            lasso_train_zscore = to_zscore(lasso.predict(train_features_imputed).flatten(), train_genders, train_pma_days, b_fpath, g_fpath)
            lasso_test_zscore = to_zscore(lasso_pred.flatten(), test_genders, test_pma_days, b_fpath, g_fpath)
            elasticnet_train_zscore = to_zscore(elasticnet.predict(train_features_imputed).flatten(), train_genders, train_pma_days, b_fpath, g_fpath)
            elasticnet_test_zscore = to_zscore(elasticnet_pred.flatten(), test_genders, test_pma_days, b_fpath, g_fpath)

            test_labels_zscores.append(test_labels_zscore)
            lasso_zscore_preds.append(lasso_test_zscore)
            elasticnet_zscore_preds.append(elasticnet_test_zscore)

        lasso_lower, lasso_upper, lasso_diff, lasso_pi_lower, lasso_pi_upper, lasso_pi_diff, lasso_lower2, lasso_upper2, lasso_diff2, lasso_pi_lower2, lasso_pi_upper2, lasso_pi_diff2 = get_prediction_interval(test_features_imputed, lasso_pred, train_features_imputed, train_labels.flatten(), lasso.predict(train_features_imputed).flatten(), pi=.95)
        lasso_lowers.append(np.array(lasso_lower))
        lasso_uppers.append(np.array(lasso_upper))
        lasso_pi_lowers.append(np.array(lasso_pi_lower))
        lasso_pi_uppers.append(np.array(lasso_pi_upper))

        lasso_lowers2.append(np.array(lasso_lower2))
        lasso_uppers2.append(np.array(lasso_upper2))
        lasso_pi_lowers2.append(np.array(lasso_pi_lower2))
        lasso_pi_uppers2.append(np.array(lasso_pi_upper2))

        if 'weight' in label_fpath:
            lasso_zscore_lower, lasso_zscore_upper, lasso_zscore_diff, lasso_zscore_pi_lower, lasso_zscore_pi_upper, lasso_zscore_pi_diff, lasso_zscore_lower2, lasso_zscore_upper2, lasso_zscore_diff2, lasso_zscore_pi_lower2, lasso_zscore_pi_upper2, lasso_zscore_pi_diff2 = get_prediction_interval(test_features_imputed, lasso_test_zscore, train_features_imputed, train_labels_zscore, lasso_train_zscore, pi=.95)
            lasso_zscore_lowers.append(np.array(lasso_zscore_lower))
            lasso_zscore_uppers.append(np.array(lasso_zscore_upper))
            lasso_zscore_pi_lowers.append(np.array(lasso_zscore_pi_lower))
            lasso_zscore_pi_uppers.append(np.array(lasso_zscore_pi_upper))

            lasso_zscore_lowers2.append(np.array(lasso_zscore_lower2))
            lasso_zscore_uppers2.append(np.array(lasso_zscore_upper2))
            lasso_zscore_pi_lowers2.append(np.array(lasso_zscore_pi_lower2))
            lasso_zscore_pi_uppers2.append(np.array(lasso_zscore_pi_upper2))

        lasso_diffs.append(np.array(lasso_diff))
        lasso_diffs2.append(np.array(lasso_diff2))
        lasso_pi_diffs.append(np.array(lasso_pi_diff))
        lasso_pi_diffs2.append(np.array(lasso_pi_diff2))

        if 'weight' in label_fpath:
            lasso_zscore_diffs.append(np.array(lasso_zscore_diff))
            lasso_zscore_diffs2.append(np.array(lasso_zscore_diff2))
            lasso_zscore_pi_diffs.append(np.array(lasso_zscore_pi_diff))
            lasso_zscore_pi_diffs2.append(np.array(lasso_zscore_pi_diff2))
        

        elasticnet_lower, elasticnet_upper, elasticnet_diff, elasticnet_pi_lower, elasticnet_pi_upper, elasticnet_pi_diff, elasticnet_lower2, elasticnet_upper2, elasticnet_diff2, elasticnet_pi_lower2, elasticnet_pi_upper2, elasticnet_pi_diff2 = get_prediction_interval(test_features_imputed, elasticnet_pred, train_features_imputed, train_labels.flatten(), elasticnet.predict(train_features_imputed).flatten(), pi=.95)
        elasticnet_lowers.append(np.array(elasticnet_lower))
        elasticnet_uppers.append(np.array(elasticnet_upper))
        elasticnet_pi_lowers.append(np.array(elasticnet_pi_lower))
        elasticnet_pi_uppers.append(np.array(elasticnet_pi_upper))

        elasticnet_lowers2.append(np.array(elasticnet_lower2))
        elasticnet_uppers2.append(np.array(elasticnet_upper2))
        elasticnet_pi_lowers2.append(np.array(elasticnet_pi_lower2))
        elasticnet_pi_uppers2.append(np.array(elasticnet_pi_upper2))

        if 'weight' in label_fpath:
            elasticnet_zscore_lower, elasticnet_zscore_upper, elasticnet_zscore_diff, elasticnet_zscore_pi_lower, elasticnet_zscore_pi_upper, elasticnet_zscore_pi_diff, elasticnet_zscore_lower2, elasticnet_zscore_upper2, elasticnet_zscore_diff2, elasticnet_zscore_pi_lower2, elasticnet_zscore_pi_upper2, elasticnet_zscore_pi_diff2 = get_prediction_interval(test_features_imputed, elasticnet_test_zscore, train_features_imputed, train_labels_zscore, elasticnet_train_zscore, pi=.95)
            elasticnet_zscore_lowers.append(np.array(elasticnet_zscore_lower))
            elasticnet_zscore_uppers.append(np.array(elasticnet_zscore_upper))
            elasticnet_zscore_pi_lowers.append(np.array(elasticnet_zscore_pi_lower))
            elasticnet_zscore_pi_uppers.append(np.array(elasticnet_zscore_pi_upper))

            elasticnet_zscore_lowers2.append(np.array(elasticnet_zscore_lower2))
            elasticnet_zscore_uppers2.append(np.array(elasticnet_zscore_upper2))
            elasticnet_zscore_pi_lowers2.append(np.array(elasticnet_zscore_pi_lower2))
            elasticnet_zscore_pi_uppers2.append(np.array(elasticnet_zscore_pi_upper2))

        elasticnet_diffs.append(np.array(elasticnet_diff))
        elasticnet_diffs2.append(np.array(elasticnet_diff2))
        elasticnet_pi_diffs.append(np.array(elasticnet_pi_diff))
        elasticnet_pi_diffs2.append(np.array(elasticnet_pi_diff2))

        if 'weight' in label_fpath:
            elasticnet_zscore_diffs.append(np.array(elasticnet_zscore_diff))
            elasticnet_zscore_diffs2.append(np.array(elasticnet_zscore_diff2))
            elasticnet_zscore_pi_diffs.append(np.array(elasticnet_zscore_pi_diff))
            elasticnet_zscore_pi_diffs2.append(np.array(elasticnet_zscore_pi_diff2))


        # record predictions and class probabilities
        lasso_preds.append(lasso_pred)
        elasticnet_preds.append(elasticnet_pred)
        nn_preds.append(nn_pred)
        knn_preds.append(knn_pred)

        lasso_r2s.append(lasso_r2)
        elasticnet_r2s.append(elasticnet_r2)
        nn_r2s.append(nn_r2)
        knn_r2s.append(knn_r2)

        labels_probas.append(test_labels)
        idss.append(ids[test_idx])

        print(f'Fold {fold_idx} with size {train_features.shape[0]}')

    idss = np.concatenate(idss, axis = 0)
    idss.dtype = int

    #np.savetxt(os.path.join(exp_prefix, f'lasso_pred_{longitudinal_period}.csv'), np.concatenate(lasso_preds, axis = 0), delimiter = ',')
    #np.savetxt(os.path.join(exp_prefix, f'elasticnet_pred_{longitudinal_period}.csv'), np.concatenate(elasticnet_preds, axis = 0), delimiter = ',')
    np.savetxt(os.path.join(exp_prefix, f'randomforest_pred_{longitudinal_period}.csv'), np.concatenate(nn_preds, axis = 0), delimiter = ',')
    np.savetxt(os.path.join(exp_prefix, f'knn_pred_{longitudinal_period}.csv'), np.concatenate(knn_preds, axis = 0), delimiter = ',')

    if 'weight' in label_fpath:
        lasso_table = pd.DataFrame({'Astarte ID': idss, 
                                    'True Weight': np.concatenate(labels_probas, axis = 0).flatten(), 
                                    'Predicted Weight': np.concatenate(lasso_preds, axis = 0).flatten(), 
                                    'Weight Lower CI 95%': np.concatenate(lasso_lowers, axis = 0), 
                                    'Weight Upper CI 95%': np.concatenate(lasso_uppers, axis = 0), 
                                    'Weight CI 95%': np.concatenate(lasso_diffs, axis = 0), 
                                    'Weight Lower PI 95%': np.concatenate(lasso_pi_lowers, axis = 0), 
                                    'Weight Upper PI 95%': np.concatenate(lasso_pi_uppers, axis = 0), 
                                    'Weight PI 95%': np.concatenate(lasso_pi_diffs, axis = 0), 
                                    'Weight Lower CI 99%': np.concatenate(lasso_lowers2, axis = 0), 
                                    'Weight Upper CI 99%': np.concatenate(lasso_uppers2, axis = 0), 
                                    'Weight CI 99%': np.concatenate(lasso_diffs2, axis = 0), 
                                    'Weight Lower PI 99%': np.concatenate(lasso_pi_lowers2, axis = 0), 
                                    'Weight Upper PI 99%': np.concatenate(lasso_pi_uppers2, axis = 0), 
                                    'Weight PI 99%': np.concatenate(lasso_pi_diffs2, axis = 0), 
                                    'True Derived Z-Score': np.concatenate(test_labels_zscores,axis = 0), 
                                    'Predicted Z-Score':  np.concatenate(lasso_zscore_preds, axis = 0), 
                                    'Z-Score Lower CI 95%': np.concatenate(lasso_zscore_lowers, axis = 0), 
                                    'Z-Score Upper CI 95%': np.concatenate(lasso_zscore_uppers, axis = 0), 
                                    'Z-Score CI 95%': np.concatenate(lasso_zscore_diffs, axis = 0), 
                                    'Z-Score Lower PI 95%': np.concatenate(lasso_zscore_pi_lowers, axis = 0), 
                                    'Z-Score Upper PI 95%': np.concatenate(lasso_zscore_pi_uppers, axis = 0), 
                                    'Z-Score PI 95%': np.concatenate(lasso_zscore_pi_diffs, axis = 0), 
                                    'Z-Score Lower CI 99%': np.concatenate(lasso_zscore_lowers2, axis = 0), 
                                    'Z-Score Upper CI 99%': np.concatenate(lasso_zscore_uppers2, axis = 0), 
                                    'Z-Score CI 99%': np.concatenate(lasso_zscore_diffs2, axis = 0), 
                                    'Z-Score Lower PI 99%': np.concatenate(lasso_zscore_pi_lowers2, axis = 0), 
                                    'Z-Score Upper PI 99%': np.concatenate(lasso_zscore_pi_uppers2, axis = 0), 
                                    'Z-Score PI 99%': np.concatenate(lasso_zscore_pi_diffs2, axis = 0)})

        elasticnet_table = pd.DataFrame({'Astarte ID': idss, 
                                    'True Weight': np.concatenate(labels_probas, axis = 0).flatten(), 
                                    'Predicted Weight': np.concatenate(elasticnet_preds, axis = 0).flatten(), 
                                    'Weight Lower CI 95%': np.concatenate(elasticnet_lowers, axis = 0), 
                                    'Weight Upper CI 95%': np.concatenate(elasticnet_uppers, axis = 0), 
                                    'Weight CI 95%': np.concatenate(elasticnet_diffs, axis = 0), 
                                    'Weight Lower PI 95%': np.concatenate(elasticnet_pi_lowers, axis = 0), 
                                    'Weight Upper PI 95%': np.concatenate(elasticnet_pi_uppers, axis = 0), 
                                    'Weight PI 95%': np.concatenate(elasticnet_pi_diffs, axis = 0), 
                                    'Weight Lower CI 99%': np.concatenate(elasticnet_lowers2, axis = 0), 
                                    'Weight Upper CI 99%': np.concatenate(elasticnet_uppers2, axis = 0), 
                                    'Weight CI 99%': np.concatenate(elasticnet_diffs2, axis = 0), 
                                    'Weight Lower PI 99%': np.concatenate(elasticnet_pi_lowers2, axis = 0), 
                                    'Weight Upper PI 99%': np.concatenate(elasticnet_pi_uppers2, axis = 0), 
                                    'Weight PI 99%': np.concatenate(elasticnet_pi_diffs2, axis = 0), 
                                    'True Derived Z-Score': np.concatenate(test_labels_zscores,axis = 0), 
                                    'Predicted Z-Score':  np.concatenate(elasticnet_zscore_preds, axis = 0), 
                                    'Z-Score Lower CI 95%': np.concatenate(elasticnet_zscore_lowers, axis = 0), 
                                    'Z-Score Upper CI 95%': np.concatenate(elasticnet_zscore_uppers, axis = 0), 
                                    'Z-Score CI 95%': np.concatenate(elasticnet_zscore_diffs, axis = 0), 
                                    'Z-Score Lower PI 95%': np.concatenate(elasticnet_zscore_pi_lowers, axis = 0), 
                                    'Z-Score Upper PI 95%': np.concatenate(elasticnet_zscore_pi_uppers, axis = 0), 
                                    'Z-Score PI 95%': np.concatenate(elasticnet_zscore_pi_diffs, axis = 0), 
                                    'Z-Score Lower CI 99%': np.concatenate(elasticnet_zscore_lowers2, axis = 0), 
                                    'Z-Score Upper CI 99%': np.concatenate(elasticnet_zscore_uppers2, axis = 0), 
                                    'Z-Score CI 99%': np.concatenate(elasticnet_zscore_diffs2, axis = 0), 
                                    'Z-Score Lower PI 99%': np.concatenate(elasticnet_zscore_pi_lowers2, axis = 0), 
                                    'Z-Score Upper PI 99%': np.concatenate(elasticnet_zscore_pi_uppers2, axis = 0), 
                                    'Z-Score PI 99%': np.concatenate(elasticnet_zscore_pi_diffs2, axis = 0)})
    else:
        lasso_table = pd.DataFrame({'Astarte ID': idss, 
                                    'True Z-Score': np.concatenate(labels_probas, axis = 0).flatten(), 
                                    'Predicted Z-Score': np.concatenate(lasso_preds, axis = 0).flatten(), 
                                    'Z-Score Lower CI 95%': np.concatenate(lasso_lowers, axis = 0), 
                                    'Z-Score Upper CI 95%': np.concatenate(lasso_uppers, axis = 0), 
                                    'Z-Score CI 95%': np.concatenate(lasso_diffs, axis = 0), 
                                    'Z-Score Lower PI 95%': np.concatenate(lasso_pi_lowers, axis = 0), 
                                    'Z-Score Upper PI 95%': np.concatenate(lasso_pi_uppers, axis = 0), 
                                    'Z-Score PI 95%': np.concatenate(lasso_pi_diffs, axis = 0), 
                                    'Z-Score Lower CI 99%': np.concatenate(lasso_lowers2, axis = 0), 
                                    'Z-Score Upper CI 99%': np.concatenate(lasso_uppers2, axis = 0), 
                                    'Z-Score CI 99%': np.concatenate(lasso_diffs2, axis = 0), 
                                    'Z-Score Lower PI 99%': np.concatenate(lasso_pi_lowers2, axis = 0), 
                                    'Z-Score Upper PI 99%': np.concatenate(lasso_pi_uppers2, axis = 0), 
                                    'Z-Score PI 99%': np.concatenate(lasso_pi_diffs2, axis = 0)})

        elasticnet_table = pd.DataFrame({'Astarte ID': idss, 
                                    'True Z-Score': np.concatenate(labels_probas, axis = 0).flatten(), 
                                    'Predicted Z-Score': np.concatenate(elasticnet_preds, axis = 0).flatten(), 
                                    'Z-Score Lower CI 95%': np.concatenate(elasticnet_lowers, axis = 0), 
                                    'Z-Score Upper CI 95%': np.concatenate(elasticnet_uppers, axis = 0), 
                                    'Z-Score CI 95%': np.concatenate(elasticnet_diffs, axis = 0), 
                                    'Z-Score Lower PI 95%': np.concatenate(elasticnet_pi_lowers, axis = 0), 
                                    'Z-Score Upper PI 95%': np.concatenate(elasticnet_pi_uppers, axis = 0), 
                                    'Z-Score PI 95%': np.concatenate(elasticnet_pi_diffs, axis = 0), 
                                    'Z-Score Lower CI 99%': np.concatenate(elasticnet_lowers2, axis = 0), 
                                    'Z-Score Upper CI 99%': np.concatenate(elasticnet_uppers2, axis = 0), 
                                    'Z-Score CI 99%': np.concatenate(elasticnet_diffs2, axis = 0), 
                                    'Z-Score Lower PI 99%': np.concatenate(elasticnet_pi_lowers2, axis = 0), 
                                    'Z-Score Upper PI 99%': np.concatenate(elasticnet_pi_uppers2, axis = 0), 
                                    'Z-Score PI 99%': np.concatenate(elasticnet_pi_diffs2, axis = 0)})

    lasso_table.to_csv(os.path.join(exp_prefix, f'lasso_pred_{longitudinal_period}.csv'))
    elasticnet_table.to_csv(os.path.join(exp_prefix, f'elasticnet_pred_{longitudinal_period}.csv'))

    #np.savetxt('labels.csv', np.concatenate(labels_probas, axis = 0), delimiter = ',')
    #np.savetxt('ids.csv', idss, fmt = '%d', delimiter = '\n')

    # box plot
    plt.clf()
    plt.boxplot([lasso_r2s, elasticnet_r2s, nn_r2s], labels = ['Lasso', 'ElasticNet', 'RandomForest'])
    plt.ylim(-0.2, 1)
    plt.title(f'R^2 Scores for All Regressors with Day 1-{longitudinal_period} Data')
    plt.xlabel('Regressors')
    plt.ylabel('R^2 Scores')
    plt.savefig(os.path.join(exp_prefix, f'r2_{longitudinal_period}.png'), dpi = 500)

    # actual patients plot
    plt.clf()
    plt.scatter(np.concatenate(labels_probas, axis = 0), np.concatenate(lasso_preds, axis = 0), c = 'red', marker = 'o', alpha = 0.7, s = 0.5, label = 'Lasso')
    plt.scatter(np.concatenate(labels_probas, axis = 0), np.concatenate(elasticnet_preds, axis = 0), c = 'blue', marker = 'x', alpha = 0.7, s = 0.5, label = 'ElasticNet')
    plt.scatter(np.concatenate(labels_probas, axis = 0), np.concatenate(nn_preds, axis = 0), c = 'green', marker = '*', alpha = 0.7, s = 0.5, label = 'RandomForest Regressor')
    plt.legend(loc = 'lower right')
    plt.title(f'Predicted vs. Actual Plot with Day 1-{longitudinal_period} Data')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(os.path.join(exp_prefix, f'pred_plot_{longitudinal_period}.png'), dpi = 600)

    print(f'Day 1-{longitudinal_period}')
    print(f'Lasso r^2: {average(lasso_r2s)}')
    print(f'ElasticNet r^2: {average(elasticnet_r2s)}')
    print(f'kNN r^2: {average(knn_r2s)}')
    #print(f'NeuralNet r^2: {average(nn_r2s)}')

# TODO: not modified to get up to date. 
"""
def run_experiment_on_site(use_dc = False, longitudinal_period = 57, use_feeding = True, use_medication = True, use_site_split = False, filter_sampling = False, exp_prefix_time = None):
    print(f'Running experiment on lon_period={longitudinal_period}, use_feeding={use_feeding}, use_medication={use_medication}, use_site_split={use_site_split}, filter_sampling={filter_sampling}')

    if exp_prefix_time is None:
        exp_prefix = os.path.join(f'results_{time.time()}', f'feeding_{use_feeding}_medication_{use_medication}_site_{use_site_split}_filter_{filter_sampling}')
    else:
        exp_prefix = os.path.join(f'results_{exp_prefix_time}', f'feeding_{use_feeding}_medication_{use_medication}_site_{use_site_split}_filter_{filter_sampling}')
    os.makedirs(exp_prefix, exist_ok = True)
    # get features
    features = patient_features(patient_fpath, use_dc = use_dc).merge(longitudinal_features(longitudinal_fpath, longitudinal_period = longitudinal_period).drop_duplicates(), on='Astarte ID', how='inner')
    features = merge_features([features, maternal_features(maternal_fpath)], method = 'left')
    if use_medication:
        features = merge_features([features, medication_features(medication_fpath, label_fpath, longitudinal_period = longitudinal_period)], method = 'left')
    if use_feeding:
        features = merge_features([features, feeding_features(feeding_fpath, longitudinal_period = longitudinal_period)], method = 'left')
    labels = features.merge(pd.read_csv(label_fpath), on='Astarte ID', how='inner')

    if filter_sampling:
        labels = filter_unsampled(labels)

    labels = labels[['Astarte ID', 'TypDev']]
    features = features[features['Astarte ID'].isin(labels['Astarte ID'].tolist())]

    features = features.sort_values(by = 'Astarte ID').drop(labels = ['Astarte ID'], axis = 1)
    labels = labels.sort_values(by = 'Astarte ID')

    ids = labels['Astarte ID'].to_numpy().astype(np.int)
    labels = labels.drop('Astarte ID', axis = 1).to_numpy()


    features.to_csv(os.path.join(exp_prefix, f'raw_features_{longitudinal_period}.csv'), index = False)
    random_miss = IterativeImputer()
    imputed_features = random_miss.fit_transform(features)
    imputed_dict = {}
    imputed_dict['Astarte ID'] = ids.tolist()
    columns = features.columns.tolist()
    assert len(columns) == imputed_features.shape[1]
    for i in range(len(columns)):
        imputed_dict[columns[i]] = imputed_features[:, i]
    pd.DataFrame(imputed_dict).to_csv(os.path.join(exp_prefix, f'imputed_features_{longitudinal_period}.csv'), index = False)

    features = features.to_numpy()

    print(f'Dataset size: {features.shape[0]}')

    # 5-fold cross validation
    splits = site_cross_validation(ids, patient_fpath)
    for idx, (site_feature_idx, site_label_idx) in enumerate(splits):

        site_feature = features[site_label_idx]
        site_label = labels[site_label_idx]

        k_fold = KFold(n_splits = 5)
        split = k_fold.split(site_feature)

        print(f'Site {idx}: {site_label.sum()} TypDev, {site_label.shape[0] - site_label.sum()} NonTypDev')

        X_header = ','.join([f'feature_{fidx}' for fidx in range(site_feature.shape[1])])
        y_header = 'label'
        np.savetxt(f'site_{idx}_matlab_train_x.csv', site_feature, delimiter = ',', header = X_header, fmt = '%.5f')
        np.savetxt(f'site_{idx}_matlab_train_y.csv', site_label, delimiter = ',', header = y_header, fmt = '%.5f')

        total_num = site_feature.shape[0]

        # initialize lists and matrices
        logistic_mat = np.zeros((2, 2))
        randomforest_mat = np.zeros((2, 2))
        randomforest_raw_mat = np.zeros((2, 2))

        logistic_proba = []
        randomforest_proba = []
        randomforest_raw_proba = []
        labels_proba = []
        randomforest_preds = []
        randomforest_raw_preds = []
        logistic_preds = []
        idss = []
        coef = None

        for fold_idx, (train_idx, test_idx) in enumerate(split):
            
            train_features = site_feature[train_idx]
            train_labels = site_label[train_idx]
            test_features = site_feature[test_idx]
            test_labels = site_label[test_idx]

            imputer = IterativeImputer()
            #imputer = KNNImputer(n_neighbors = 5)

            train_features_imputed = imputer.fit_transform(train_features)
            test_features_imputed = imputer.transform(test_features)

            randomforest = RandomForestClassifier(n_estimators = 200, max_features = 'sqrt')
            logistic = LogisticRegression(max_iter = 600, C = 1, penalty = 'l1', solver = 'liblinear')
            #mlp = NeuralNet(train_features.shape[1], [32, 16, 1])
            
            # fit on all classifiers
            randomforest.fit(train_features_imputed, train_labels)
            logistic.fit(train_features_imputed, train_labels)
            if coef is None:
                coef = np.array(logistic.coef_)
            else:
                coef = np.add(coef, logistic.coef_)

            randomforest_pred = randomforest.predict(test_features_imputed)
            logistic_pred = logistic.predict(test_features_imputed)

            # Connect to matlab

            # save training and testing data to folder
            X_header = ','.join([f'feature_{fidx}' for fidx in range(train_features.shape[1])])
            y_header = 'label'
            #train_labels_cpy = np.array(train_labels)
            np.savetxt('matlab_train_x.csv', train_features, delimiter = ',', header = X_header, fmt = '%.5f')
            np.savetxt('matlab_train_y.csv', train_labels.astype(np.int), delimiter = ',', header = y_header, fmt = '%d')
            np.savetxt('matlab_test_x.csv', test_features, delimiter = ',', header = X_header, fmt = '%.5f')

            randomforest_raw_pred = eng.randomforest(os.path.join(pwd, 'matlab_train_x.csv'), os.path.join(pwd, 'matlab_train_y.csv'), os.path.join(pwd, 'matlab_test_x.csv'), '0')
            pred_table=np.array(pd.read_csv('matlab_y_pred.csv', sep=',',header=None).values)
            score_table=np.array(pd.read_csv('matlab_y_score.csv', sep=',',header=None).values)
            randomforest_raw_pred = pred_table

            logistic_mat = np.add(logistic_mat, confusion_matrix(test_labels, logistic_pred))
            randomforest_mat = np.add(randomforest_mat, confusion_matrix(test_labels, randomforest_pred))
            randomforest_raw_mat = np.add(randomforest_raw_mat, confusion_matrix(test_labels, randomforest_raw_pred))

            logistic_proba.append(logistic.predict_proba(test_features_imputed))
            randomforest_proba.append(randomforest.predict_proba(test_features_imputed))
            randomforest_raw_proba.append(score_table)
            labels_proba.append(test_labels)
            idss.append(ids[test_idx])



        save_confusion_matrix(randomforest_mat, title = f'Random Forest on Imputed Dataset with Day 1-{longitudinal_period} Data, acc={((randomforest_mat[0, 0] + randomforest_mat[1, 1]) / len(site_label)):.3f}', fpath = os.path.join(exp_prefix, f'site_{idx}cm_randomforest_raw_{longitudinal_period}.png'))

        save_confusion_matrix(randomforest_raw_mat, title = f'Random Forest on Raw Dataset with Day 1-{longitudinal_period} Data, acc={((randomforest_raw_mat[0, 0] + randomforest_raw_mat[1, 1]) / len(site_label)):.3f}', fpath = os.path.join(exp_prefix, f'site_{idx}cm_randomforest_{longitudinal_period}.png'))

        save_confusion_matrix(logistic_mat, title = f'Logistic on Imputed Dataset with Day 1-{longitudinal_period} Data, acc={((logistic_mat[0, 0] + logistic_mat[1, 1]) / len(site_label)):.3f}', fpath = os.path.join(exp_prefix, f'site_{idx}cm_logistic_{longitudinal_period}.png'))

        idss = np.concatenate(idss, axis = 0)
        idss.dtype = int

        np.savetxt('labels.csv', np.concatenate(labels_proba, axis = 0), delimiter = ',')
        np.savetxt('ids.csv', idss, fmt = '%d', delimiter = '\n')

        labels_proba = np.concatenate(labels_proba, axis = 0)
        randomforest_proba = np.concatenate(randomforest_proba, axis = 0)[:, 1]
        randomforest_raw_proba = np.concatenate(randomforest_raw_proba, axis = 0)[:, 1]
        logistic_proba = np.concatenate(logistic_proba, axis = 0)[:, 1]

        plt.clf()
        randomforest_fpr, randomforest_tpr, _ = roc_curve(labels_proba, randomforest_proba)
        randomforest_auc = auc(randomforest_fpr, randomforest_tpr)
        randomforest_raw_fpr, randomforest_raw_tpr, _ = roc_curve(labels_proba, randomforest_raw_proba)
        randomforest_raw_auc = auc(randomforest_raw_fpr, randomforest_raw_tpr)
        logistic_fpr, logistic_tpr, _ = roc_curve(labels_proba, logistic_proba)
        logistic_auc = auc(logistic_fpr, logistic_tpr)

        plt.plot(randomforest_fpr, randomforest_tpr, label = f'RandomForest w/ Imputed (auc={randomforest_auc:.3f})')
        plt.plot(randomforest_raw_fpr, randomforest_raw_tpr, label = f'RandomForest w/o Imputed (auc={randomforest_raw_auc:.3f})')
        plt.plot(logistic_fpr, logistic_tpr, label = f'LogisticRegression w/ Imputed (auc={logistic_auc:.3f})')
        plt.title(f'ROC Curves for All Classifiers with Day 1-{longitudinal_period} Features')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc = 'lower right')
        plt.savefig(os.path.join(exp_prefix, f'site_{idx}_roc_curve_1_{longitudinal_period}.png'), dpi = 500)

        plt.clf()
        randomforest_fpr, randomforest_tpr, _ = precision_recall_curve(labels_proba, randomforest_proba)
        randomforest_raw_fpr, randomforest_raw_tpr, _ = precision_recall_curve(labels_proba, randomforest_raw_proba)
        logistic_fpr, logistic_tpr, _ = precision_recall_curve(labels_proba, logistic_proba)

        plt.scatter(randomforest_tpr, randomforest_fpr, label = f'RandomForest w/ Imputed', s = 1)
        plt.scatter(randomforest_raw_tpr, randomforest_raw_fpr, label = f'RandomForest w/o Imputed', s = 1)
        plt.scatter(logistic_tpr, logistic_fpr, label = f'LogisticRegression w/ Imputed', s = 1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f'Precision-Recall Curves for All Classifiers with Day 1-{longitudinal_period} Features')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc = 'lower left')
        plt.savefig(os.path.join(exp_prefix, f'site_{idx}_recall_precision_curve_1_{longitudinal_period}.png'), dpi = 500)

        print(f'Day 1-{longitudinal_period}')
        print(f'Random Forest on Imputed Dataset acc={((randomforest_mat[0, 0] + randomforest_mat[1, 1]) / len(site_label)):.3f}')
        print(f'Random Forest on Raw Dataset acc={((randomforest_raw_mat[0, 0] + randomforest_raw_mat[1, 1]) / len(site_label)):.3f}')
        print(f'Logistic on Imputed Dataset acc={((logistic_mat[0, 0] + logistic_mat[1, 1]) / len(site_label)):.3f}')
        print(f'RandomForest w/ Imputed auc={randomforest_auc:.3f}')
        print(f'RandomForest w/o Imputed auc={randomforest_raw_auc:.3f}')
        print(f'LogisticRegression w/ Imputed auc={logistic_auc:.3f}')
"""
if __name__ == '__main__':

    curr_time = time.strftime('%m_%d_%Y', time.gmtime())

    
    curr_time_ = str(curr_time) + '_zscore_35-37_sequenced'
    label_fpath = os.path.join(PREFIX, '2020_09_19_regression_z_sequenced.csv')
    run_experiment(save_model = True, use_probiotics = has_probiotics_fpath, probiotics_fpath = probiotics_fpath, use_dc = False, longitudinal_period = 7, use_feeding = True, use_medication = True, use_site_split = False, filter_sampling = False, exp_prefix_time = curr_time_)
    
    curr_time_ = str(curr_time) + '_weight_35-37_sequenced'
    label_fpath = os.path.join(PREFIX, '2020_09_19_regression_weight_sequenced.csv')
    run_experiment(save_model = True, use_probiotics = has_probiotics_fpath, probiotics_fpath = probiotics_fpath, use_dc = False, longitudinal_period = 7, use_feeding = True, use_medication = True, use_site_split = False, filter_sampling = False, exp_prefix_time = curr_time_)
