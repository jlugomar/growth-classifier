import shap
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, ElasticNet

from classification import patient_fpath, maternal_fpath, longitudinal_fpath, label_fpath, medication_fpath, feeding_fpath, has_probiotics_fpath, probiotics_fpath
from regression import label_fpath as reg_label_fpath

from util import *

shap.initjs()

def perturbation_analysis(clf, longitudinal_period = None, X = None):
    if type(clf).__name__ == 'str':
        clf = pickle.load(open(clf, 'rb'))
    if X == None:
        if type(clf).__name__ == 'LogisticRegression':
            is_regression = False
            if longitudinal_period == None:
                raise AttributeError('Should specify X or longitudinal_period for classification task')
        else:
            is_regression = True
            longitudinal_period = 7
        X = preprocess_features(patient_fpath, 
                                maternal_fpath, 
                                longitudinal_fpath, 
                                medication_fpath, 
                                feeding_fpath, 
                                (reg_label_fpath if is_regression else label_fpath), 
                                use_dc = False, 
                                longitudinal_period = longitudinal_period, 
                                use_feeding = True, 
                                use_medication = True, 
                                use_probiotics = has_probiotics_fpath, 
                                probiotics_fpath = probiotics_fpath, 
                                is_regression = is_regression)
        X = X.drop(labels = '36 Weeks PMA', axis = 1)
        X = X.to_numpy()
    explainer = shap.LinearExplainer(clf, X)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values, X)