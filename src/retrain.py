# Filename: classification.py
# Author: Siwei Xu
# Date: 08/24/2020
#
# Usage: python classification.py
#
# Description: experiment runner code for classification tasks
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

import torch
import torch.nn as nn

from generate_features import *
from util import *

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
pwd = os.getcwd()

def get_feature_rank(data_fpath, longitudinal_period = 1):
    features = pd.read_csv(os.path.join(data_fpath, f'raw_features_{longitudinal_period}.csv'))

    # remove the 36 weeks pma features
    features = features.drop(labels = ['Unnamed: 0'], axis = 1)
    l = features.columns.to_list()
    features = features.to_numpy()
    labels = pd.read_csv(os.path.join(data_fpath, f'matlab_train_y_imputed_{longitudinal_period}.csv')).to_numpy()

    imputer, features_imputed = impute_features(features)

    pca = PCA(n_components = features_imputed.shape[1])

    pca.fit(features_imputed)

    pca_ratio = pca.explained_variance_ratio_[:3]
    pca_1 = np.abs(pca.components_[0])
    pca_2 = np.abs(pca.components_[1])
    pca_3 = np.abs(pca.components_[2])
    pca_1 *= pca_ratio[0]
    pca_2 *= pca_ratio[1]
    pca_3 *= pca_ratio[2]

    ls = []
    for i in range(len(l)):
        ls.append((max(pca_1[i], pca_2[i], pca_3[i]), l[i]))
    ls = list(reversed(sorted(ls)))
    _, res = zip(*ls)

    return res


def run_experiment(data_fpath, 
                   longitudinal_period = 1, 
                   exp_prefix_time = None, feature_list = []):
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
    
    # generate experiment folder
    if exp_prefix_time is None:
        exp_prefix = os.path.join('..', 'results', f'results_{time.time()}', f'retrain')
    else:
        exp_prefix = os.path.join('..', 'results', f'results_{exp_prefix_time}', f'retrain')
    os.makedirs(exp_prefix, exist_ok = True)

    features = pd.read_csv(os.path.join(data_fpath, f'raw_features_{longitudinal_period}.csv'))

    # remove the 36 weeks pma features
    features = features.drop(labels = ['Unnamed: 0'], axis = 1)[feature_list].to_numpy()
    labels = pd.read_csv(os.path.join(data_fpath, f'matlab_train_y_imputed_{longitudinal_period}.csv')).to_numpy()

    # 5-fold cross validation
    k_fold = KFold(n_splits = 5, shuffle = True)
    split = k_fold.split(features)

    total_num = features.shape[0]

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

    for fold_idx, (train_idx, test_idx) in enumerate(split):
        
        # obtain training and testing features and labels
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]

        imputer, train_features_imputed = impute_features(train_features)
        test_features_imputed = imputer.transform(test_features)

        logistic = LogisticRegression(max_iter = 600, 
                                      C = 2, 
                                      penalty = 'l2', 
                                      solver = 'lbfgs', 
                                      class_weight = 'balanced')
        #mlp = NeuralNet(train_features.shape[1], [32, 16, 1])
        
        # fit on all classifiers
        logistic_pred, logistic_proba = run_classifier(train_features_imputed, train_labels, test_features_imputed, logistic)

        if coef is None:
            coef = np.array(logistic.coef_)
        else:
            coef = np.add(coef, logistic.coef_)

        # record predictions and class probabilities

        logistic_preds.append(logistic_pred)
        logistic_probas.append(logistic_proba)

        logistic_mat = np.add(logistic_mat, confusion_matrix(test_labels, logistic_pred))

        labels_probas.append(test_labels)
        
        logistic_tmp = confusion_matrix(test_labels, logistic_pred)
        
        logistic_proba_tmp = logistic_proba[:, 1]

        logistic_fpr, logistic_tpr, _ = roc_curve(test_labels, logistic_proba_tmp)
        logistic_auc = auc(logistic_fpr, logistic_tpr)

    print(f'{feature_list[-1]},{((logistic_mat[0, 0] + logistic_mat[1, 1]) / len(labels)):.6f}')
    '''
    np.savetxt(os.path.join(exp_prefix, f'logistic_coef_day_{longitudinal_period}.csv'), coef, delimiter = ',')

    save_confusion_matrix(logistic_mat, title = f'Logistic on Imputed Dataset with Day 1-{longitudinal_period} Data, acc={((logistic_mat[0, 0] + logistic_mat[1, 1]) / len(labels)):.3f}', fpath = os.path.join(exp_prefix, f'cm_logistic_{longitudinal_period}.png'))

    np.savetxt(os.path.join(exp_prefix, f'logistic_impute_{longitudinal_period}.csv'), np.concatenate(logistic_probas, axis = 0), delimiter = ',')
    np.savetxt(os.path.join(exp_prefix, f'logistic_pred_impute_{longitudinal_period}.csv'), np.concatenate(logistic_preds, axis = 0), delimiter = ',')

    #np.savetxt('labels.csv', np.concatenate(labels_probas, axis = 0), delimiter = ',')
    #np.savetxt('ids.csv', idss, fmt = '%d', delimiter = '\n')

    labels_probas = np.concatenate(labels_probas, axis = 0)
    logistic_probas = np.concatenate(logistic_probas, axis = 0)[:, 1]

    plt.clf()
    logistic_fpr, logistic_tpr, _ = roc_curve(labels_probas, logistic_probas)
    logistic_auc = auc(logistic_fpr, logistic_tpr)

    plt.plot(logistic_fpr, logistic_tpr, label = f'LogisticRegression w/ Imputed (auc={logistic_auc:.3f})')
    plt.title(f'ROC Curves for All Classifiers with Day 1-{longitudinal_period} Features')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'lower right')
    plt.savefig(os.path.join(exp_prefix, f'roc_curve_1_{longitudinal_period}.png'), dpi = 500)

    plt.clf()
    logistic_fpr, logistic_tpr, _ = precision_recall_curve(labels_probas, logistic_probas)
    logistic_auc = area_under_curve(logistic_tpr, logistic_fpr)

    plt.scatter(logistic_tpr, logistic_fpr, label = f'LogisticRegression w/ Imputed (auc={logistic_auc:.3f})', s = 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'Precision-Recall Curves for All Classifiers with Day 1-{longitudinal_period} Features')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc = 'lower left')
    plt.savefig(os.path.join(exp_prefix, f'recall_precision_curve_1_{longitudinal_period}.png'), dpi = 500)

    print(f'Day 1-{longitudinal_period}')
    print(f'Logistic on Imputed Dataset acc={((logistic_mat[0, 0] + logistic_mat[1, 1]) / len(labels)):.3f}')
    print(f'LogisticRegression w/ Imputed auc={logistic_auc:.3f}')
    '''

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
    features = features.drop(labels = '36 Weeks PMA', axis = 1)
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
        np.savetxt(f'site_{idx}_matlab_train_x_{longitudinal_period}.csv', site_feature, delimiter = ',', header = X_header, fmt = '%.5f')
        np.savetxt(f'site_{idx}_matlab_train_y_{longitudinal_period}.csv', site_label.astype(np.int), delimiter = ',', header = y_header, fmt = '%d')

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

# running the experiments
if __name__ == '__main__':
    lon_list = [1, 15, 29, 43, 57]

    curr_time = time.strftime('%m_%d_%Y', time.gmtime())

    curr_time_ = str(curr_time) + '_minimal_normal'
    label_fpath = os.path.join(PREFIX, '2020_07_9_gf_labels_flipped.csv')

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    print('features_len,acc')
    
    #run_experiment(save_model = True, use_probiotics = has_probiotics_fpath, probiotics_fpath = probiotics_fpath, use_dc = False, longitudinal_period = 57, use_feeding = True, use_medication = True, use_site_split = False, filter_sampling = False, exp_prefix_time = curr_time_)

    
    l = dict()
    if 'gf' in label_fpath:
        l[1] = ['Birthweight', 'Weight_day_1', 'Birth PMA', 'PMA_day_1', 'GA', 'Maternal Age', 'Birth Z-Score', 'Gender', 'Multiple Gestation', 'Mode of Delivery']
        l[15] = ['Birthweight', 'Weight_day_1', 'Weight_day_15', 'feeding_week2_qty_breastmilk', 'feeding_week2_qty_formula', 'feeding_week1_qty_donated', 'feeding_week1_qty_breastmilk', 'feeding_week2_qty_donated', 'feeding_week1_qty_formula', 'Birth PMA', 'PMA_day_1', 'PMA_day_15', 'GA', 'feeding_week2_number_days_breastmilk', 'medication_week2_medication_other_antibiotics']
        l[29] = ['feeding_week4_qty_breastmilk', 'feeding_week3_qty_breastmilk', 'feeding_week2_qty_breastmilk', 'Weight_day_29', 'feeding_week4_qty_formula', 'Birthweight', 'Weight_day_15', 'Weight_day_1', 'feeding_week3_qty_formula', 'feeding_week2_qty_formula', 'feeding_week1_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week2_qty_donated', 'feeding_week1_qty_formula', 'feeding_week3_qty_donated', 'feeding_week4_qty_donated', 'Birth PMA', 'PMA_day_1', 'PMA_day_15', 'PMA_day_29', 'GA', 'feeding_week4_number_days_breastmilk', 'feeding_week4_number_days_formula', 'feeding_week3_number_days_formula', 'feeding_week3_number_days_breastmilk', 'feeding_week2_number_days_formula', 'feeding_week2_number_days_breastmilk', 'medication_week2_medication_other_antibiotics', 'medication_week3_medication_other_antibiotics']
    else:
        l[1] = ['Birthweight', 'Weight_day_1', 'Birth PMA', 'PMA_day_1', 'GA', 'Maternal Age', 'Birth Z-Score', 'Gender', 'Multiple Gestation', 'Mode of Delivery']
        l[15] = ['Birthweight', 'Weight_day_1', 'Weight_day_15', 'feeding_week2_qty_breastmilk', 'feeding_week2_qty_formula', 'feeding_week1_qty_donated', 'feeding_week1_qty_breastmilk', 'feeding_week2_qty_donated', 'feeding_week1_qty_formula', 'Birth PMA', 'PMA_day_1', 'PMA_day_15', 'GA', 'feeding_week2_number_days_breastmilk', 'medication_week2_medication_other_antibiotics', 'feeding_week1_number_days_donated', 'medication_week1_medication_other_antibiotics']
        l[29] = ['feeding_week4_qty_breastmilk', 'feeding_week3_qty_breastmilk', 'feeding_week2_qty_breastmilk', 'Weight_day_29', 'feeding_week4_qty_formula', 'Birthweight', 'Weight_day_15', 'Weight_day_1', 'feeding_week3_qty_formula', 'feeding_week2_qty_formula', 'feeding_week1_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week2_qty_donated', 'feeding_week1_qty_formula', 'feeding_week3_qty_donated', 'feeding_week4_qty_donated', 'Birth PMA', 'PMA_day_1', 'PMA_day_15', 'PMA_day_29', 'GA', 'feeding_week4_number_days_breastmilk', 'feeding_week4_number_days_formula', 'feeding_week3_number_days_formula']
    for lon in [1, 15, 29]:

        #entries = get_feature_rank(os.path.join('..', 'results', 'results_09_16_2020_normal_label_with_probio', 'feeding_True_medication_True_site_False_filter_False'), longitudinal_period = lon)
        run_experiment(feature_list = l[lon], data_fpath = os.path.join('..', 'results', 'results_09_16_2020_normal_label_with_probio', 'feeding_True_medication_True_site_False_filter_False'), longitudinal_period = lon, exp_prefix_time = curr_time_)
    '''
    curr_time_ = str(curr_time) + '_gf'
    label_fpath = os.path.join(PREFIX, '2020_07_9_gf_labels_flipped.csv')

    for use_site_split in [True, False]:
        for filter_sampling in [True, False]:
            for lon in [1, 15, 29, 43, 57]:
                run_experiment(save_model = True, use_probiotics = has_probiotics_fpath, probiotics_fpath = probiotics_fpath, use_dc = False, longitudinal_period = lon, use_feeding = True, use_medication = True, use_site_split = use_site_split, filter_sampling = filter_sampling, exp_prefix_time = curr_time_)
    '''