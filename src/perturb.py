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

import torch
import torch.nn as nn

from generate_features import *
from util import *

type_l = ['breastmilk', 'donated', 'formula']

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

def perturb(features_table, num, exp_prefix, longitudinal_period, day_1_type = None, day_2_type = None, qty_1_times = 1):
    
    assert np.logical_xor(day_1_type != None and day_2_type != None, qty_1_times != 1) 

    names = ['feeding_week1_number_days_breastmilk', 'feeding_week1_number_days_donated', 'feeding_week1_number_days_formula', 'feeding_week1_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week1_qty_formula']
    selected_table = features_table.loc[features_table['True label'] > 0.5, :]
    before_table = selected_table.copy(deep = True)
    selected_table[['Astarte ID', 'True label']].to_csv(os.path.join(exp_prefix, f'perturb_patients_selected_{longitudinal_period}.csv'))
    

    for row in selected_table.index:
        
        if day_1_type != None and day_2_type != None:
            feature_1 = type_l[day_1_type]
            feature_2 = type_l[day_2_type]

            selected_table.loc[row, 'feeding_week1_number_days_breastmilk'] = 0
            selected_table.loc[row, 'feeding_week1_number_days_donated'] = 0
            selected_table.loc[row, 'feeding_week1_number_days_formula'] = 0
            selected_table.loc[row, 'feeding_week1_qty_breastmilk'] = 0
            selected_table.loc[row, 'feeding_week1_qty_donated'] = 0
            selected_table.loc[row, 'feeding_week1_qty_formula'] = 0

            selected_table.loc[row, 'feeding_week2_number_days_breastmilk'] = 0
            selected_table.loc[row, 'feeding_week2_number_days_donated'] = 0
            selected_table.loc[row, 'feeding_week2_number_days_formula'] = 0
            selected_table.loc[row, 'feeding_week2_qty_breastmilk'] = 0
            selected_table.loc[row, 'feeding_week2_qty_donated'] = 0
            selected_table.loc[row, 'feeding_week2_qty_formula'] = 0

            selected_table.loc[row, f'feeding_week1_number_days_{feature_1}'] += 7
            selected_table.loc[row, f'feeding_week2_number_days_{feature_2}'] += 7

            prev_feed1 = 10000
            prev_feed2 = 10000

            for i in range(1, 8):
                selected_table.loc[row, f'feeding_week1_qty_{feature_1}'] += min(DEFAULT_FEEDING_QTY_D[i][day_1_type], prev_feed1 + 30)
                prev_feed1 = min(DEFAULT_FEEDING_QTY_D[i][day_1_type], prev_feed1 + 30)
            for  i in range(8, 15):
                selected_table.loc[row, f'feeding_week2_qty_{feature_2}'] += min(DEFAULT_FEEDING_QTY_D[i][day_2_type], prev_feed2 + 30)
                prev_feed2 = min(DEFAULT_FEEDING_QTY_D[i][day_2_type], prev_feed2 + 30)
        elif qty_1_times != 1:
            selected_table.loc[row, 'feeding_week1_qty_breastmilk'] = qty_1_times
            selected_table.loc[row, 'feeding_week1_qty_donated'] *= qty_1_times
            selected_table.loc[row, 'feeding_week1_qty_formula'] *= qty_1_times

    return before_table, selected_table
        
        

def run_experiment(data_fpath, 
                   longitudinal_period = 1, 
                   exp_prefix_time = None, day_1_type = None, day_2_type = None, qty_1_times = 1):
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
        if day_1_type is not None and day_2_type is not None:
            exp_prefix = os.path.join('..', 'results', f'results_{time.time()}', f'perturb_{type_l[day_1_type]}_{type_l[day_2_type]}')
        elif qty_1_times != 1:
            exp_prefix = os.path.join('..', 'results', f'results_{time.time()}', f'perturb_qty_{qty_1_times:.2f}x')
    else:
        if day_1_type is not None and day_2_type is not None:
            exp_prefix = os.path.join('..', 'results', f'results_{exp_prefix_time}', f'perturb_{type_l[day_1_type]}_{type_l[day_2_type]}')
        elif qty_1_times != 1:
            exp_prefix = os.path.join('..', 'results', f'results_{exp_prefix_time}', f'perturb_qty_{qty_1_times:.2f}x')
    os.makedirs(exp_prefix, exist_ok = True)

    features = pd.read_csv(os.path.join(data_fpath, f'raw_features_{longitudinal_period}.csv'))

    # remove the 36 weeks pma features
    features_table = features.drop(labels = ['Unnamed: 0'], axis = 1)
    ids = pd.read_csv(os.path.join(data_fpath, f'imputed_features_{longitudinal_period}.csv'))['Astarte ID'].tolist()

    features_table['Astarte ID'] = ids
    labels_table = pd.read_csv(os.path.join(data_fpath, f'matlab_train_y_imputed_{longitudinal_period}.csv'))['# label'].tolist()
    features_table['True label'] = labels_table
    features = features.drop(labels = ['Unnamed: 0'], axis = 1).to_numpy()
    labels = pd.read_csv(os.path.join(data_fpath, f'matlab_train_y_imputed_{longitudinal_period}.csv')).to_numpy()

    print(features_table)

    # 5-fold cross validation

    total_num = features.shape[0]

    # initialize lists and matrices
    logistic_mat = np.zeros((2, 2))

    logistic_probas = []

    labels_probas = []

    logistic_preds = []

    idss = []

    coef = None

    before_table, selected_table = perturb(features_table, 100, exp_prefix, longitudinal_period, day_1_type = day_1_type, day_2_type = day_2_type, qty_1_times = qty_1_times)

    before_table_matrix = before_table.drop(labels = ['Astarte ID', 'True label'], axis = 1).to_numpy()
    selected_table_matrix = selected_table.drop(labels = ['Astarte ID', 'True label'], axis = 1).to_numpy()


        
    # obtain training and testing features and labels
    train_features = features
    train_labels = labels

    imputer, train_features_imputed = impute_features(train_features)
    before_table_matrix_imputed = imputer.transform(before_table_matrix)
    selected_table_matrix_imputed = imputer.transform(selected_table_matrix)

    logistic = LogisticRegression(max_iter = 600, C = 2, penalty = 'l2', solver = 'lbfgs', class_weight = 'balanced')
    #mlp = NeuralNet(train_features.shape[1], [32, 16, 1])
    
    # fit on all classifiers
    logistic_pred, logistic_proba = run_classifier(train_features_imputed, train_labels, train_features_imputed, logistic)

    before_pred = logistic.predict(before_table_matrix_imputed)
    selected_pred = logistic.predict(selected_table_matrix_imputed)

    before_score = logistic.predict_proba(before_table_matrix_imputed)
    selected_score = logistic.predict_proba(selected_table_matrix_imputed)

    before_table['Pred label'] = before_pred.flatten().tolist()
    before_table['Pred prob_0'] = before_score[:, 0].tolist()
    before_table['Pred prob_1'] = before_score[:, 1].tolist()
    
    selected_table['Pred label'] = selected_pred.flatten().tolist()
    selected_table['Pred prob_0'] = selected_score[:, 0].tolist()
    selected_table['Pred prob_1'] = selected_score[:, 1].tolist()

    before_table[['Astarte ID', 'feeding_week1_number_days_breastmilk', 'feeding_week1_number_days_donated', 'feeding_week1_number_days_formula', 'feeding_week1_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week1_qty_formula', 'Pred label', 'Pred prob_0', 'Pred prob_1']].to_csv(os.path.join(exp_prefix, f'perturb_patients_original_{longitudinal_period}.csv'))
    selected_table[['Astarte ID', 'feeding_week1_number_days_breastmilk', 'feeding_week1_number_days_donated', 'feeding_week1_number_days_formula', 'feeding_week1_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week1_qty_formula', 'Pred label', 'Pred prob_0', 'Pred prob_1']].to_csv(os.path.join(exp_prefix, f'perturb_patients_perturbed_{longitudinal_period}.csv'))


# running the experiments
if __name__ == '__main__':
    lon_list = [1, 15, 29, 43, 57]

    curr_time = time.strftime('%m_%d_%Y', time.gmtime())

    
    curr_time_ = str(curr_time) + '_perturb_day1-15'
    label_fpath = os.path.join(PREFIX, '2020_07_9_labels_flipped.csv')
    
    #run_experiment(save_model = True, use_probiotics = has_probiotics_fpath, probiotics_fpath = probiotics_fpath, use_dc = False, longitudinal_period = 57, use_feeding = True, use_medication = True, use_site_split = False, filter_sampling = False, exp_prefix_time = curr_time_)

    for lon in [15]:
        for day_1_type in [0, 1, 2]:
            for day_2_type in [0, 1, 2]:
                run_experiment(data_fpath = os.path.join('..', 'results', 'results_09_16_2020_normal_label_with_probio', 'feeding_True_medication_True_site_False_filter_False'), longitudinal_period = lon, exp_prefix_time = curr_time_, day_1_type = day_1_type, day_2_type = day_2_type)
    '''
    for lon in [15]:
        for qty_1_times in [0.5, 1.5, 2]:
            run_experiment(data_fpath = os.path.join('..', 'results', 'results_09_16_2020_normal_label_with_probio', 'feeding_True_medication_True_site_False_filter_False'), longitudinal_period = lon, exp_prefix_time = curr_time_, qty_1_times = qty_1_times)
    '''
    curr_time_ = str(curr_time) + '_perturb_gf_day1-15'
    label_fpath = os.path.join(PREFIX, '2020_07_9_gf_labels_flipped.csv')
    
    #run_experiment(save_model = True, use_probiotics = has_probiotics_fpath, probiotics_fpath = probiotics_fpath, use_dc = False, longitudinal_period = 57, use_feeding = True, use_medication = True, use_site_split = False, filter_sampling = False, exp_prefix_time = curr_time_)

    for lon in [15]:
        for day_1_type in [0, 1, 2]:
            for day_2_type in [0, 1, 2]:
                run_experiment(data_fpath = os.path.join('..', 'results', 'results_09_16_2020_gf_with_probio', 'feeding_True_medication_True_site_False_filter_False'), longitudinal_period = lon, exp_prefix_time = curr_time_, day_1_type = day_1_type, day_2_type = day_2_type)
    '''
    for lon in [15]:
        for qty_1_times in [0.5, 1.5, 2]:
            run_experiment(data_fpath = os.path.join('..', 'results', 'results_09_16_2020_gf_with_probio', 'feeding_True_medication_True_site_False_filter_False'), longitudinal_period = lon, exp_prefix_time = curr_time_, qty_1_times = qty_1_times)
    '''
    