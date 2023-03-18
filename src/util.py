# Filename: util.py
# Author: Siwei Xu
# Date: 10/08/2020
#
# Description: utility functions for classification and regression

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from scipy.stats import t as t_dist

from generate_labels import LABELS
from generate_features import *

pwd = os.getcwd()

# function for calculating auc
def area_under_curve(xs, ys):
    s = list(sorted(zip(xs, ys)))
    res = 0.0
    for i in range(1, len(s)):
        res += (s[i - 1][1] + s[i][1]) * (s[i][0] - s[i - 1][0]) / 2
    return res


# function for saving confusion matrices
def save_confusion_matrix(cm, title = '', fpath = 'figure.png', is_growth_failure = False):
    if is_growth_failure:
        LABELS = ['Non-GF', 'GF']
    else:
        LABELS = ['TypDev', 'NonTypDev']
    cm = cm.astype(np.int)
    plt.clf()
    disp = ConfusionMatrixDisplay(cm, display_labels = LABELS)
    disp = disp.plot(include_values = True, cmap = plt.cm.Blues, values_format = 'd')
    disp.ax_.set_title(title)
    plt.savefig(fpath, dpi = 450)


# function for generating site-specific cross-validation folds
def site_cross_validation(labels, patient_fpath):
    patient_table = pd.read_csv(patient_fpath, index_col = 0)[['Sequenced', 'Site']].dropna()
    patient_table.index.astype(np.int)
    sites = pd.unique(patient_table['Site']).tolist()
    folds = {}
    for i in sites:
        folds[i] = []
    for i in range(len(labels)):
        folds[patient_table.loc[labels[i], 'Site']].append(i)
    result = []
    for i in sites:
        train_idx = []
        test_site = i
        test_idx = folds[i]
        train_sites = [x for x in sites if x != i]
        for train_site in train_sites:
            train_idx += folds[train_site]
        result.append((train_idx, test_idx))
    return result


# function for filtering out unsequenced patients
def filter_unsampled(df):
    return df[df['Sequenced'].isin([1])]


# function for preprocessing the features
def preprocess_features(patient_fpath, maternal_fpath, longitudinal_fpath, medication_fpath, feeding_fpath, label_fpath, use_dc = False, longitudinal_period = 1, use_feeding = True, use_medication = True, use_probiotics = None, probiotics_fpath = None, is_regression = False, use_day_features = False):
    features = patient_features(patient_fpath, use_dc = use_dc, is_regression = is_regression)
    features = features.merge(longitudinal_features(longitudinal_fpath, longitudinal_period = longitudinal_period, is_regression = is_regression).drop_duplicates(), on='Astarte ID', how='inner')
    features = merge_features([features, maternal_features(maternal_fpath)], method = 'left')
    if use_medication:
        features = merge_features([features, medication_features_weekly(medication_fpath, label_fpath, longitudinal_period = longitudinal_period, is_regression = is_regression, use_day_features = use_day_features)], method = 'left')
    if use_feeding:
        features = merge_features([features, feeding_features_weekly(feeding_fpath, longitudinal_period = longitudinal_period, use_day_features = use_day_features)], method = 'left')
    if use_probiotics is not None and probiotics_fpath is not None:
        features = merge_features([features, probiotics_features_weekly(probiotics_fpath, label_fpath, longitudinal_period = longitudinal_period, is_regression = is_regression, use_day_features = use_day_features)], method = 'left')
    return features


# function for initializing the imputers
def impute_features(features, feature_list = None):
    imputer = VotingKNNImputer(n_neighbors = 5, weights = 'distance', feature_list = feature_list)
    res = imputer.fit_transform(features)
    return imputer, res


# function for running a classifier
def run_classifier(train_x, train_y, test_x, clf):
    if type(clf).__name__ != 'MatlabEngine':

        # if it's sklearn classifier, just run it
        clf.fit(train_x, train_y)
        pred = clf.predict(test_x)
        proba = clf.predict_proba(test_x)
    else:

        # if it's matlab classifier, 
        X_header = ','.join([f'feature_{fidx}' for fidx in range(train_x.shape[1])])
        y_header = 'label'
        unique, counts = np.unique(train_y.flatten(), return_counts=True)

        # do random data re-sampling
        count_dict = dict(zip(unique, counts))
        neg_count = count_dict[0]
        pos_count = count_dict[1]
        if neg_count > 1 * pos_count:
            desired_num = int(1 * (neg_count - pos_count))
            idx = np.nonzero(train_y.flatten() == 1)[0]
            resampled_idx = np.random.choice(idx, size = desired_num)
            train_x = np.concatenate([train_x, train_x[resampled_idx, :]], axis = 0)
            train_y = np.concatenate([train_y, train_y[resampled_idx, :]], axis = 0)
        elif pos_count > 1 * neg_count:
            desired_num = int(1 * (pos_count - neg_count))
            idx = np.nonzero(train_y.flatten() == 0)[0]
            resampled_idx = np.random.choice(idx, size = desired_num)
            train_x = np.concatenate([train_x, train_x[resampled_idx, :]], axis = 0)
            train_y = np.concatenate([train_y, train_y[resampled_idx, :]], axis = 0)

        # save as file and pass filename to matlab function
        np.savetxt('matlab_train_x.csv', train_x, delimiter = ',', header = X_header, fmt = '%.5f')
        np.savetxt('matlab_train_y.csv', train_y.astype(np.int), delimiter = ',', header = y_header, fmt = '%d')
        np.savetxt('matlab_test_x.csv', test_x, delimiter = ',', header = X_header, fmt = '%.5f')
        randomforest_raw_pred = clf.randomforest(os.path.join(pwd, 'matlab_train_x.csv'), os.path.join(pwd, 'matlab_train_y.csv'), os.path.join(pwd, 'matlab_test_x.csv'), '0')
        pred = np.array(pd.read_csv('matlab_y_pred.csv', sep=',',header=None).values)
        proba = np.array(pd.read_csv('matlab_y_score.csv', sep=',',header=None).values)

        # remove temp files
        os.remove('matlab_train_x.csv')
        os.remove('matlab_train_y.csv')
        os.remove('matlab_test_x.csv')
        os.remove('matlab_y_pred.csv')
        os.remove('matlab_y_score.csv')
    return pred, proba


# function for running a regressor
def run_regressor(train_x, train_y, test_x, test_y, reg):
    if type(reg).__name__ != 'MatlabEngine':
        reg.fit(train_x, train_y)
        return reg.predict(test_x), reg.score(test_x, test_y)
    else:
        X_header = ','.join([f'feature_{fidx}' for fidx in range(train_x.shape[1])])
        y_header = 'label'
        np.savetxt('matlab_train_x.csv', train_x, delimiter = ',', header = X_header, fmt = '%.5f')
        np.savetxt('matlab_train_y.csv', train_y, delimiter = ',', header = y_header, fmt = '%.5f')
        np.savetxt('matlab_test_x.csv', test_x, delimiter = ',', header = X_header, fmt = '%.5f')
        randomforest_raw_pred = reg.randomforest_regressor(os.path.join(pwd, 'matlab_train_x.csv'), os.path.join(pwd, 'matlab_train_y.csv'), os.path.join(pwd, 'matlab_test_x.csv'), '0')
        pred = np.array(pd.read_csv('matlab_y_pred.csv', sep=',',header=None).values)
        os.remove('matlab_train_x.csv')
        os.remove('matlab_train_y.csv')
        os.remove('matlab_test_x.csv')
        os.remove('matlab_y_pred.csv')
        return pred, r2_score(test_y, pred)

# utility function to save raw features as a csv table
def save_raw_features(features, exp_prefix, longitudinal_period = 1):
    features.to_csv(os.path.join(exp_prefix, f'raw_features_{longitudinal_period}.csv'))


# utility function to calculate the arithmatic average
def average(lst): 
    return sum(lst) / len(lst) 


# utility function to calculate the confidence and prediction intervals
def get_prediction_interval(prediction_x, prediction, x_test, y_test, test_predictions, pi=.95):
    
    #get standard deviation of y_test
    assert y_test.shape == test_predictions.shape

    # calculate statistics
    x_bar = np.mean(x_test, axis = 0)
    s_x = np.linalg.norm(np.std(x_test, axis = 0))
    n = y_test.shape[0]
    s_res = np.std(test_predictions - y_test)

    # calculate T-scores
    t_score = t_dist.ppf(1 - (1 - 0.95) / 2, n - 1)
    t_score_90 = t_dist.ppf(1 - (1 - 0.99) / 2, n - 1)

    # initialize result arrays
    uppers = []
    lowers = []

    uppers2 = []
    lowers2 = []

    pi_uppers = []
    pi_lowers = []

    pi_uppers2 = []
    pi_lowers2 = []

    diff = []
    diff2 = []
    pi_diff = []
    pi_diff2 = []

    for i in range(prediction.shape[0]):

        # calculate standard error
        se = s_res * np.sqrt(1 / n + np.linalg.norm((prediction_x[i, :] - x_bar), ord = 1) ** 2 / ((n - 1) * (s_x ** 2)))
        se2 = s_res * np.sqrt(1 + 1 / n + np.linalg.norm((prediction_x[i, :] - x_bar), ord = 1) ** 2 / ((n - 1) * (s_x ** 2)))

        # append results
        diff.append(t_score * se)
        diff2.append(t_score_90 * se)
        pi_diff.append(t_score * se2)
        pi_diff2.append(t_score_90 * se2)

        lowers.append(prediction[i] - t_score * se) 
        uppers.append(prediction[i] + t_score * se)
        lowers2.append(prediction[i] - t_score_90 * se) 
        uppers2.append(prediction[i] + t_score_90 * se)
        pi_lowers2.append(prediction[i] - t_score_90 * se2)
        pi_uppers2.append(prediction[i] + t_score_90 * se2)
        pi_lowers.append(prediction[i] - t_score * se2)
        pi_uppers.append(prediction[i] + t_score * se2)

    return lowers, uppers, diff, pi_lowers, pi_uppers, pi_diff, lowers2, uppers2, diff2, pi_lowers2, pi_uppers2, pi_diff2


# utility function to convert weight to z-score
def to_zscore(weights, genders, pma_days, b_fpath, g_fpath):
    b_table = pd.read_csv(b_fpath, sep = '\t')
    g_table = pd.read_csv(g_fpath, sep = '\t')

    b_dict, g_dict = {}, {}

    for idx, item in b_table.iterrows():
        b_dict[round(item['time'], 1)] = (item['lambda'], item['mu'], item['sigma'])

    for idx, item in g_table.iterrows():
        g_dict[round(item['time'], 1)] = (item['lambda'], item['mu'], item['sigma'])

    res = []

    for i in range(len(weights)):
        if genders[i] > 0.5:
            lam, mu, sigma = b_dict[round(pma_days[i] / 7, 1)]
        else:
            lam, mu, sigma = g_dict[round(pma_days[i] / 7, 1)]
        res.append((np.power((weights[i]/mu),lam) - 1.0)/ (lam * sigma))

    return np.array(res)


# utility function to plot the histogram of observed typdev/nontypdev
def plot(column):
    plt.clf()
    plt.hist(typdev[column], bins = 30, color = 'cyan', label = 'TypDev', alpha = 0.7)
    plt.hist(nontypdev[column], bins = 30, color = 'orange', label = 'NonTypDev', alpha = 0.7)
    plt.title(f'Observed {column}')
    plt.xlabel(f'Observed {column}')
    plt.yticks(range(20))
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{column.replace("/", ",")}.png', dpi = 500)


# Implement KNNImputer with voting instead of average
from sklearn.neighbors._base import _get_weights, _check_weights
class VotingKNNImputer(KNNImputer):

    def __init__(self, *, missing_values=np.nan, n_neighbors=5,
                 weights="uniform", metric="nan_euclidean", copy=True,
                 add_indicator=False, feature_list = None):
        super().__init__(missing_values = missing_values, n_neighbors = n_neighbors, 
                         weights = weights, metric = metric, copy = copy, 
                         add_indicator = add_indicator)
        self.feature_list = feature_list
        self.continuous_features = ['Birthweight', 'Birth Z-Score', 'Weight_day_1', 'Weight_day_15', 'Weight_day_29', 'Maternal Age', 'feeding_week1_qty_breastmilk', 'feeding_week1_qty_donated', 'feeding_week1_qty_formula', 'feeding_week2_qty_breastmilk', 'feeding_week2_qty_donated', 'feeding_week2_qty_formula', 'feeding_week3_qty_breastmilk', 'feeding_week3_qty_donated', 'feeding_week3_qty_formula', 'feeding_week4_qty_breastmilk', 'feeding_week4_qty_donated', 'feeding_week4_qty_formula', 'feeding_week5_qty_breastmilk', 'feeding_week5_qty_donated', 'feeding_week5_qty_formula', 'feeding_week6_qty_breastmilk', 'feeding_week6_qty_donated', 'feeding_week6_qty_formula', 'feeding_week7_qty_breastmilk', 'feeding_week7_qty_donated', 'feeding_week7_qty_formula', 'feeding_week8_qty_breastmilk', 'feeding_week8_qty_donated', 'feeding_week8_qty_formula']
        
    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        """Helper function to impute a single column.

        The logic is to first check if the input value is continuous or 
        discrete based on the overall value given by the datasheet. 
        If the data is likely discrete, we do majority voting. Otherwise,
        we do average of the k nearest neighbors.

        Parameters
        ----------
        dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
            Distance matrix between the receivers and potential donors from
            training set. There must be at least one non-nan distance between
            a receiver and a potential donor.
        n_neighbors : int
            Number of neighbors to consider.
        fit_X_col : ndarray of shape (n_potential_donors,)
            Column of potential donors from training set.
        mask_fit_X_col : ndarray of shape (n_potential_donors,)
            Missing mask for fit_X_col.
        Returns
        -------
        imputed_values: ndarray of shape (n_receivers,)
            Imputed values for receiver.
        """
        # Get donors
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1,
                                     axis=1)[:, :n_neighbors]

        # Get weight matrix from from distance matrix
        donors_dist = dist_pot_donors[
            np.arange(donors_idx.shape[0])[:, None], donors_idx]

        weight_matrix = _get_weights(donors_dist, self.weights)

        # fill nans with zeros
        if weight_matrix is not None:
            weight_matrix[np.isnan(weight_matrix)] = 0.0

        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        donors = np.ma.array(donors, mask=donors_mask)

        assert donors_mask.any() == False

        donors = donors.data 
        result = np.zeros((donors.shape[0]))

        for i in range(donors.shape[0]):
            d = {}

            for j in range(donors.shape[1]):
                if donors[i, j] not in d:
                    d[donors[i, j]] = [1, weight_matrix[i, j]]
                else:
                    d[donors[i, j]] = [d[donors[i, j]][0] + 1, d[donors[i, j]][1] + weight_matrix[i, j]]

            s = sum((donors[i, :] / n_neighbors).astype(int))
            if self.feature_list is not None:
                # get average if the value is continuous
                if self.feature_list[j] in self.continuous_features:
                    result[i] = s

                # get majority voting if the value is discrete
                else:
                    max_val = 0
                    max_freq = 0
                    max_weight = 0
                    for k in d:
                        if d[k][0] > max_freq:
                            max_val = k 
                            max_freq = d[k][0]
                            max_weight = d[k][1]
                        elif d[k][0] == max_freq:
                            if 0 in d:
                                max_val = 0
                            elif d[k][1] > max_weight:
                                max_val = k
                                max_weight = d[k][1]

                    result[i] = max_val
            else:
                # get average if the value is continuous
                if abs(s - sum(donors[i, :] / 5)) < 1E-9 and (-1.0 not in d) and (0.0 not in d or d[0.0][0] < (n_neighbors / 2 - 1)):
                    result[i] = s

                # get majority voting if the value is discrete
                else:
                    max_val = 0
                    max_freq = 0
                    max_weight = 0
                    for k in d:
                        if d[k][0] > max_freq:
                            max_val = k 
                            max_freq = d[k][0]
                            max_weight = d[k][1]
                        elif d[k][0] == max_freq:
                            if 0 in d:
                                max_val = 0
                            elif d[k][1] > max_weight:
                                max_val = k
                                max_weight = d[k][1]

                    result[i] = max_val

        return result
