# Filename: generate_features.py
# Author: Siwei Xu
# Date: 09/17/2020
#
# Description: process the features from the csv files

import pandas as pd 
import numpy as np
import re
import statistics
import math
import sys

# feature labels to be included
PATIENT_FEATURE_COLUMNS = ['Astarte ID','GA', 'Site', 'Birth PMA', 'Birthweight', 'Length of Stay', 'D/C Weight', '36 Weeks PMA', 'Mode of Delivery', 'Multiple Gestation', 'Gender', 'Birth Z-Score']
PATIENT_FEATURE_COLUMNS_NO_DC = ['Astarte ID','GA', 'Site', 'Birth PMA', 'Birthweight', '36 Weeks PMA', 'Mode of Delivery', 'Multiple Gestation', 'Gender', 'Birth Z-Score']
PATIENT_FEATURE_COLUMNS_NO_DC_REGRESSION = ['Astarte ID','GA', 'Site', 'Birth PMA', 'Birthweight', '36 Weeks PMA', 'Mode of Delivery', 'Multiple Gestation', 'Gender', 'Ethnicity', 'Number of Additional Births', 'ROM Prior to Delivery', 'Apgar 1 min', 'Apgar 5 min', 'Antenatal Steroids', 'Birth Z-Score']

MEDICATION_TYPE = {'Acyclovir': 0, 'Acyclovir (Zovirax)': 1, 'Ambisome': 2, 'Amoxicillin': 3, 'Ampicillin': 4, 'Azithromycin': 5, 'Azithromycin PO': 6, 'Cefazolin (Ancef)': 7, 'Cefazolin (ancef)': 8, 'Cefepime': 9, 'Cefepime (Maxipime)': 10, 'Cefotaxime': 11, 'Cefotaxime (Claforan)': 12, 'Cefoxitin': 13, 'Ceftazidime': 14, 'Ceftazidime (Fortaz)': 15, 'Cefuroximine': 16, 'Cephalexin': 17, 'Cephalexin (keflex)': 18, 'Clindamycin': 19, 'Clotrimazole': 20, 'Co-amoxiclav': 21, 'Co-trimoxazole': 22, 'Erythromycin PO (GI dosing)': 23, 'Exacillin (bactocill)': 24, 'Flucloxacillin': 25, 'Fluconazole': 26, 'Gentamicin': 27, 'Linezolid': 28, 'Meropenem': 29, 'Metronidazole': 30, 'Metronidazole/Flagyl': 31, 'Moxifloxacin': 32, 'Mupirocin': 33, 'Nafcillin': 34, 'Nevirapine': 35, 'Oxacillin': 36, 'Penicillin': 37, 'Penicillin G potassium': 38, 'Penicillin v potassium (VEETID)': 39, 'Tazocin': 40, 'Tobramycin': 41, 'Trimethoprim prophylaxis': 42, 'Vancomycin': 43, 'Zidovudine (RETROVIR)': 44, 'Zosyn': 45, 'amoxicillin': 46, 'gentamicin': 47, 'meropenem': 48, 'metronidazole': 49}
MEDICATION_FEATURE_STR = ['had_medication', 'medication_amp_gen', 'medication_other_antibiotics']

FEEDING_TYPE = {'DBM': 1, 'EBM': 0, 'Breastfeeding': 0, 'DM': 1, 'Form (Monogen)': 2, 'Form': 2, 'HM': 0, 'Other': 0, 'Epf': 2, 'Enfacare': 2, 'Epf High Protein': 2, 'Enfamil': 2}
FEEDING_FEATURE_STR = ['number_days_breastmilk', 'number_days_donated', 'number_days_formula', 'qty_breastmilk', 'qty_donated', 'qty_formula']

# constants for feeding quantity
USE_DELIVERY_BMI = False
DEFAULT_FEEDING_QTY_D = dict()
DEFAULT_FEEDING_QTY_D[0] = [0, 0, 0]
DEFAULT_FEEDING_QTY_D[1] = [3, 5.9, 11]
DEFAULT_FEEDING_QTY_D[2] = [6, 18.4, 29.6]
DEFAULT_FEEDING_QTY_D[3] = [15, 28, 49.75]
DEFAULT_FEEDING_QTY_D[4] = [29.45, 52.9, 81]
DEFAULT_FEEDING_QTY_D[5] = [52.7, 72.75, 83]
DEFAULT_FEEDING_QTY_D[6] = [74, 102.75, 108]
DEFAULT_FEEDING_QTY_D[7] = [86.25, 102, 87]
DEFAULT_FEEDING_QTY_D[8] = [127.75, 137, 150]
DEFAULT_FEEDING_QTY_D[9] = [150, 144, 92]
DEFAULT_FEEDING_QTY_D[10] = [150, 118, 149]
DEFAULT_FEEDING_QTY_D[11] = [150, 128, 150]
DEFAULT_FEEDING_QTY_D[12] = [150, 110, 150]
DEFAULT_FEEDING_QTY_D[13] = [150, 96, 150]
DEFAULT_FEEDING_QTY_D[14] = [150, 132, 150]
DEFAULT_FEEDING_QTY_D[15] = [150, 145, 150]
DEFAULT_FEEDING_QTY_D[-1] = [150, 145, 150]

ENTERAL_PATTERN = r'([\w\(\) ]+)=([\d\.]+)/(\d+)'
pd.set_option('chained_assignment',None)



def patient_features(patient_fpath, use_dc = False, use_site = False, is_regression = False):
    """
    Process the patient features and return a processed pandas dataframe

    Parameters:
    patients_fpath        (str): File path to the patients csv file
    use_dc               (bool): Use d/c related features. Default to false
    use_site             (bool): Use site as features. Default to false
    is_regression        (bool): Use as regression feature. Default to false

    Returns:
    Pandas DataFrame: pandas table containing patient features
    """
    patient_table = pd.read_csv(patient_fpath)
    if False:#is_regression:
        features = patient_table[PATIENT_FEATURE_COLUMNS_NO_DC_REGRESSION]
    else:
        features = patient_table[PATIENT_FEATURE_COLUMNS if use_dc else PATIENT_FEATURE_COLUMNS_NO_DC]
    '''
    if is_regression:
        features.loc[features['Ethnicity'] == 'Hispanic', 'Ethnicity'] = 1
        features.loc[features['Ethnicity'] == 'Non-Hispanic', 'Ethnicity'] = 0
    '''
    # add site feature if needed
    if use_site:
        features['Site_B'] = (features['Site'] == 'B').astype(int)
        features['Site_C'] = (features['Site'] == 'C').astype(int)
        features['Site'] = (features['Site'] == 'A').astype(int)
    else:
        features = features.drop(labels = 'Site', axis = 1)
    return features


def maternal_features(maternal_fpath):
    """
    Process the maternal features and return a processed pandas dataframe

    Parameters:
    maternal_fpath        (str): File path to the maternal csv file

    Returns:
    Pandas DataFrame: pandas table containing maternal features
    """
    maternal_table = pd.read_csv(maternal_fpath)

    # only use the maternal age for default
    feature_column = ['Astarte ID', 'Maternal Age']
    if USE_DELIVERY_BMI:
        feature_column.append('Delivery BMI (kg/m2)')
    features = maternal_table[feature_column]
    return features


def longitudinal_features(lon_fpath, longitudinal_period = 1, is_regression = False):
    """
    Process the longitudinal features and return a processed pandas dataframe

    Parameters:
    lon_fpath             (str): File path to the longitudinal csv file
    longitudinal_period  (int): end date of the longitudinal period in use
    is_regression        (bool): Use as regression feature. Default to false

    Returns:
    Pandas DataFrame: pandas table containing longitudinal features
    """
    DOL = [i for i in [1, 15, 29] if i <= longitudinal_period]
    lon_table = pd.read_csv(lon_fpath).drop_duplicates()
    raw_features = lon_table[['Astarte ID', 'DOL', 'Daily Weight (g)', 'PMA Days', 'Weekly Head Circumference', 'Weekly Length (cm)']]

    ids = []

    g = raw_features.groupby('Astarte ID')

    # only reporting day 1, day 15, and day 29 weights

    # regression model - only report day 1 and day 7
    if longitudinal_period == 7:
        weights = [[], [], [], []]
        heads = []
        lengths = []
        for i in g:
            if 1.0 in i[1]['DOL'].tolist():
                head = np.NaN
                length = np.NaN
                flag = True
                has_first = True
                for index, item in i[1].drop_duplicates().iterrows():
                    if item['DOL'] == 1.0 and has_first == True:
                        ids.append(item['Astarte ID'])
                        weights[0].append(item['Daily Weight (g)'])
                        weights[2].append(item['PMA Days'])
                        has_first = False
                    elif item['DOL'] == 7.0 and flag:
                        weights[1].append(item['Daily Weight (g)'])
                        weights[3].append(item['PMA Days'])
                        flag = False
                    if is_regression:
                        if item['DOL'] <= 7.0 and math.isnan(item['Weekly Head Circumference']) == False:
                            head = item['Weekly Head Circumference']
                        if item['DOL'] <= 7.0 and math.isnan(item['Weekly Length (cm)']) == False:
                            length = item['Weekly Length (cm)']
                heads.append(head)
                lengths.append(length)
                    
                if flag:
                    weights[1].append(np.NaN)
                    weights[3].append(np.NaN)
        if is_regression:
            features = pd.DataFrame({'Astarte ID': ids, 
                                     'Weight_day_1': weights[0], 
                                     'Weight_day_7': weights[1], 
                                     'PMA_day_1': weights[2], 
                                     'PMA_day_7': weights[3], 
                                     'First_week_head_circumference': heads, 
                                     'First_week_length': lengths})
        else:
            features = pd.DataFrame({'Astarte ID': ids, 
                                     'Weight_day_1': weights[0], 
                                     'Weight_day_7': weights[1], 
                                     'PMA_day_1': weights[2], 
                                     'PMA_day_7': weights[3]})
    elif longitudinal_period >= 29:
        weights = [[], [], [], [], [], []]
        for i in g:
            if 1.0 in i[1]['DOL'].tolist():
                flag1 = True
                flag2 = True

                # record day 1, 15, and 29 weights
                for index, item in i[1].iterrows():
                    if item['DOL'] == 1.0:
                        ids.append(item['Astarte ID'])
                        weights[0].append(item['Daily Weight (g)'])
                        weights[3].append(item['PMA Days'])
                    elif item['DOL'] == 15.0 and math.isnan(item['Daily Weight (g)']) == False:
                        weights[1].append(item['Daily Weight (g)'])
                        weights[4].append(item['PMA Days'])
                        flag1 = False
                    elif item['DOL'] == 29.0 and math.isnan(item['Daily Weight (g)']) == False:
                        weights[2].append(item['Daily Weight (g)'])
                        weights[5].append(item['PMA Days'])
                        flag2 = False

                # try to infer if not available
                if flag1:
                    d = {}
                    for index, item in i[1].iterrows():
                        if math.isnan(item['Daily Weight (g)']) == False:
                            d[int(item['DOL'])] = (item['Daily Weight (g)'], item['PMA Days'])
                    if 14 in d and 16 in d:
                        weights[1].append((d[14][0] + d[16][0]) / 2)
                        weights[4].append(d[14][1] + 1)
                        flag1 = False
                if flag1:
                    weights[1].append(np.NaN)
                    weights[4].append(np.NaN)
                if flag2:
                    d = {}
                    for index, item in i[1].iterrows():
                        if math.isnan(item['Daily Weight (g)']) == False:
                            d[int(item['DOL'])] = (item['Daily Weight (g)'], item['PMA Days'])
                    if 27 in d and 28 in d:
                        weights[2].append(d[28][0] - d[27][0] + d[28][0])
                        weights[5].append(d[28][1] + 1)
                        flag2 = False
                    elif 28 in d:
                        weights[2].append((d[28][0] - d[1][0]) / 27 + d[28][0])
                        weights[5].append(d[28][1] + 1)
                        flag2 = False
                if flag2:
                    weights[2].append(np.NaN)
                    weights[5].append(np.NaN)
        features = pd.DataFrame({'Astarte ID': ids, 
                                 'Weight_day_1': weights[0], 
                                 'Weight_day_15': weights[1], 
                                 'Weight_day_29': weights[2], 
                                 'PMA_day_1': weights[3], 
                                 'PMA_day_15': weights[4], 
                                 'PMA_day_29': weights[5]})
    elif longitudinal_period >= 15:
        weights = [[], [], [], []]
        for i in g:
            if 1.0 in i[1]['DOL'].tolist():
                flag = True
                
                # record day 1 and 15 weights
                for index, item in i[1].iterrows():
                    if item['DOL'] == 1.0:
                        ids.append(item['Astarte ID'])
                        weights[0].append(item['Daily Weight (g)'])
                        weights[2].append(item['PMA Days'])
                    elif item['DOL'] == 15.0 and math.isnan(item['Daily Weight (g)']) == False:
                        weights[1].append(item['Daily Weight (g)'])
                        weights[3].append(item['PMA Days'])
                        flag = False

                # try to infer if not available
                if flag:
                    d = {}
                    for index, item in i[1].iterrows():
                        if math.isnan(item['Daily Weight (g)']) == False:
                            d[int(item['DOL'])] = (item['Daily Weight (g)'], item['PMA Days'])
                    if 14 in d and 16 in d:
                        weights[1].append((d[14][0] + d[16][0]) / 2)
                        weights[3].append(d[14][1] + 1)
                        flag = False
                if flag:
                    weights[1].append(np.NaN)
                    weights[3].append(np.NaN)
        features = pd.DataFrame({'Astarte ID': ids, 
                                 'Weight_day_1': weights[0], 
                                 'Weight_day_15': weights[1], 
                                 'PMA_day_1': weights[2], 
                                 'PMA_day_15': weights[3]})
    else:
        weights = [[], []]
        for i in g:
            # record day 1 weight
            if 1.0 in i[1]['DOL'].tolist():
                for index, item in i[1].iterrows():
                    if item['DOL'] == 1.0:
                        ids.append(item['Astarte ID'])
                        weights[0].append(item['Daily Weight (g)'])
                        weights[1].append(item['PMA Days'])
        features = pd.DataFrame({'Astarte ID': ids, 
                                 'Weight_day_1': weights[0], 
                                 'PMA_day_1': weights[1]})
    return features


def medication_features(medication_fpath, label_fpath, longitudinal_period = 1, is_regression = False, use_day_features = False):
    """
    Process the medication features into day2-8, etc and return a processed pandas dataframe

    Parameters:
    medication_fpath      (str): File path to the medication csv file
    label_fpath           (str): File path to the label csv file
    longitudinal_period   (int): End date of the longitudinal period in use
    is_regression        (bool): Use as regression feature. Default to false
    use_day_features     (bool): Include day features or not (day 1, 15, 29, 
                                 etc). Default to false. 

    Returns:
    Pandas DataFrame: pandas table containing medication features
    """

    # read csv table from file
    max_day = int(longitudinal_period / 7)
    medication_vector_size = 3
    medication_table = pd.read_csv(medication_fpath).drop_duplicates()
    type_vector_size = len(MEDICATION_TYPE)
    label_table = pd.read_csv(label_fpath).drop_duplicates()

    # initialize arrays
    features = {}
    features_day = {}
    for index, item in label_table.iterrows():
        features[int(item['Astarte ID'])] = [0] * (medication_vector_size * (max_day))
        if longitudinal_period < 15:
            features_day[int(item['Astarte ID'])] = [0, 0, 0]
        elif longitudinal_period < 29:
            features_day[int(item['Astarte ID'])] = [0, 0, 0, 0, 0, 0]
        else:
            features_day[int(item['Astarte ID'])] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # get all the numbers from raw csv files
    g = medication_table.groupby('Astarte ID')
    for aid, i in g:
        for index, item in i.iterrows():

            # find start and end for each medication
            start = item['Start Date']
            if math.isnan(start):
                start = 1
            else:
                start = int(start)
            duration = item['Duration']
            if math.isnan(duration):
                duration = 1
            else:
                duration = int(duration)
            end = start + duration

            # check if antibiotic and if amp or gen
            is_antibiotic = int(item['Therapuetic'] == 'Antibiotic')
            is_amp_gen = int(item['Medication'].lower() == 'gentamicin' or item['Medication'].lower() == 'ampicillin')
            if aid in features:

                # populate day features
                if start <= 1 and end > 1:
                    features_day[int(aid)][0] = 1
                    if is_antibiotic:
                        if is_amp_gen:
                            features_day[int(aid)][1] += 1
                        else:
                            features_day[int(aid)][1] += 1
                if longitudinal_period >= 15 and start <= 15 and end > 15:
                    features_day[int(aid)][3] = 1
                    if is_antibiotic:
                        if is_amp_gen:
                            features_day[int(aid)][4] += 1
                        else:
                            features_day[int(aid)][5] += 1
                if longitudinal_period >= 29 and start <= 29 and end > 29:
                    features_day[int(aid)][6] = 1
                    if is_antibiotic:
                        if is_amp_gen:
                            features_day[int(aid)][7] += 1
                        else:
                            features_day[int(aid)][8] += 1

                # populate week features
                for day in range(max(2, start), min(end, longitudinal_period+1)):
                    week = int((day - 2) / 7)
                    features[int(aid)][week * medication_vector_size] = 1
                if is_antibiotic:
                    for day in range(max(2, start), min(end, longitudinal_period+1)):
                        week = int((day - 2) / 7)
                        if is_amp_gen:
                            features[int(aid)][week * medication_vector_size + 1] += 1
                        else:
                            features[int(aid)][week * medication_vector_size + 1] += 1

    # convert to pandas dataframe
    ids, features = zip(*list(features.items()))
    ids_day, features_day = zip(*list(features_day.items()))
    assert ids == ids_day
    features = np.array(features)
    features_day = np.array(features_day)
    table_dict = {}
    table_dict['Astarte ID'] = ids

    if use_day_features:
        table_dict[f'medication_day_1_{MEDICATION_FEATURE_STR[0]}'] = features_day[:, 0].tolist()
        table_dict[f'medication_day_1_{MEDICATION_FEATURE_STR[1]}'] = features_day[:, 1].tolist()
        table_dict[f'medication_day_1_{MEDICATION_FEATURE_STR[2]}'] = features_day[:, 2].tolist()
        if features_day.shape[1] >= 4:
            table_dict[f'medication_day_15_{MEDICATION_FEATURE_STR[0]}'] = features_day[:, 3].tolist()
            table_dict[f'medication_day_15_{MEDICATION_FEATURE_STR[1]}'] = features_day[:, 4].tolist()
            table_dict[f'medication_day_15_{MEDICATION_FEATURE_STR[2]}'] = features_day[:, 5].tolist()
        if features_day.shape[1] >= 6:
            table_dict[f'medication_day_29_{MEDICATION_FEATURE_STR[0]}'] = features_day[:, 6].tolist()
            table_dict[f'medication_day_29_{MEDICATION_FEATURE_STR[1]}'] = features_day[:, 7].tolist()
            table_dict[f'medication_day_29_{MEDICATION_FEATURE_STR[2]}'] = features_day[:, 8].tolist()
    
    for i in range(features.shape[1]):
        table_dict[f'medication_day_{int(i / medication_vector_size) * 7 + 2}-{min((int(i / medication_vector_size) + 1) * 7 + 1, longitudinal_period)}_{MEDICATION_FEATURE_STR[i % medication_vector_size]}'] = features[:, i].tolist()
    
    result = pd.DataFrame(table_dict)
    return result


def medication_features_weekly(medication_fpath, label_fpath, longitudinal_period = 1, is_regression = False, use_day_features = False):
    """
    Process the medication features into weeks and return a processed pandas dataframe

    Parameters:
    medication_fpath      (str): File path to the medication csv file
    label_fpath           (str): File path to the label csv file
    longitudinal_period   (int): End date of the longitudinal period in use
    is_regression        (bool): Use as regression feature. Default to false
    use_day_features     (bool): Include day features or not (day 1, 15, 29, 
                                 etc). Default to false. 

    Returns:
    Pandas DataFrame: pandas table containing medication features
    """

    # add offset for regression (assuming that it's for day 1-7)
    if is_regression:
        longitudinal_period += 1

    # read from csv files
    max_day = int(longitudinal_period / 7)
    medication_vector_size = 3
    medication_table = pd.read_csv(medication_fpath).drop_duplicates()
    type_vector_size = len(MEDICATION_TYPE)
    label_table = pd.read_csv(label_fpath).drop_duplicates()

    # initialize arrays
    features = {}
    features_day = {}
    for index, item in label_table.iterrows():
        features[int(item['Astarte ID'])] = [0] * (medication_vector_size * (max_day))
        if longitudinal_period < 15:
            features_day[int(item['Astarte ID'])] = [0, 0, 0]
        elif longitudinal_period < 29:
            features_day[int(item['Astarte ID'])] = [0, 0, 0, 0, 0, 0]
        else:
            features_day[int(item['Astarte ID'])] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # get all the numbers from raw csv files
    g = medication_table.groupby('Astarte ID')
    for aid, i in g:
        for index, item in i.iterrows():

            # fetch start and end date
            start = item['Start Date']
            if math.isnan(start):
                start = 1
            else:
                start = int(start)
            duration = item['Duration']
            if math.isnan(duration):
                duration = 1
            else:
                duration = int(duration)
            end = start + duration

            # check if antibiotics and if amp or gen
            is_antibiotic = int(item['Therapuetic'] == 'Antibiotic')
            is_amp_gen = int(item['Medication'].lower() == 'gentamicin' or item['Medication'].lower() == 'ampicillin')
            if aid in features:

                # populate day features
                if start <= 1 and end > 1:
                    features_day[int(aid)][0] = 1
                    if is_antibiotic:
                        if is_amp_gen:
                            features_day[int(aid)][1] += 1
                        else:
                            features_day[int(aid)][2] += 1
                if longitudinal_period >= 15 and start <= 15 and end > 15:
                    features_day[int(aid)][3] = 1
                    if is_antibiotic:
                        if is_amp_gen:
                            features_day[int(aid)][4] += 1
                        else:
                            features_day[int(aid)][5] += 1
                if longitudinal_period >= 29 and start <= 29 and end > 29:
                    features_day[int(aid)][6] = 1
                    if is_antibiotic:
                        if is_amp_gen:
                            features_day[int(aid)][7] += 1
                        else:
                            features_day[int(aid)][8] += 1

                # populate week features
                for day in range(start, min(end, longitudinal_period)):
                    week = int((day - 1) / 7)
                    features[int(aid)][week * medication_vector_size] = 1
                if is_antibiotic:
                    for day in range(start, min(end, longitudinal_period)):
                        week = int((day - 1) / 7)
                        if is_amp_gen:
                            features[int(aid)][week * medication_vector_size + 1] += 1
                        else:
                            features[int(aid)][week * medication_vector_size + 2] += 1

    # convert to pandas dataframe
    ids, features = zip(*list(features.items()))
    ids_day, features_day = zip(*list(features_day.items()))
    assert ids == ids_day
    features = np.array(features)
    features_day = np.array(features_day)
    table_dict = {}
    table_dict['Astarte ID'] = ids

    # add day features if necessary
    if use_day_features:
        table_dict[f'medication_day_1_{MEDICATION_FEATURE_STR[0]}'] = features_day[:, 0].tolist()
        table_dict[f'medication_day_1_{MEDICATION_FEATURE_STR[1]}'] = features_day[:, 1].tolist()
        table_dict[f'medication_day_1_{MEDICATION_FEATURE_STR[2]}'] = features_day[:, 2].tolist()
        if features_day.shape[1] >= 4:
            table_dict[f'medication_day_15_{MEDICATION_FEATURE_STR[0]}'] = features_day[:, 3].tolist()
            table_dict[f'medication_day_15_{MEDICATION_FEATURE_STR[1]}'] = features_day[:, 4].tolist()
            table_dict[f'medication_day_15_{MEDICATION_FEATURE_STR[2]}'] = features_day[:, 5].tolist()
        if features_day.shape[1] >= 6:
            table_dict[f'medication_day_29_{MEDICATION_FEATURE_STR[0]}'] = features_day[:, 6].tolist()
            table_dict[f'medication_day_29_{MEDICATION_FEATURE_STR[1]}'] = features_day[:, 7].tolist()
            table_dict[f'medication_day_29_{MEDICATION_FEATURE_STR[2]}'] = features_day[:, 8].tolist()

    # add week features
    for i in range(features.shape[1]):
        table_dict[f'medication_week{int(i / medication_vector_size) + 1}_{MEDICATION_FEATURE_STR[i % medication_vector_size]}'] = features[:, i].tolist()

    result = pd.DataFrame(table_dict)
    return result


def feeding_features(feeding_fpath, longitudinal_period = 1, use_day_features = False):
    """
    Process the feeding features into day2-8, etc and return a processed pandas dataframe

    Parameters:
    feeding_fpath         (str): File path to the feeding csv file
    longitudinal_period   (int): end date of the longitudinal period in use
    use_day_features     (bool): Include day features or not (day 1, 15, 29, 
                                 etc). Default to false. 

    Returns:
    Pandas DataFrame: pandas table containing feeding features
    """
    FEEDING_VECTOR_SIZE = 6
    max_day = int(longitudinal_period / 7)
    feeding_table = pd.read_csv(feeding_fpath).drop_duplicates()
    features = []
    features_days = []
    ids = []

    # get numbers from the raw csv file
    g = feeding_table.groupby('Astarte ID')
    for aid, i in g:
        flag = False
        raw_days = []

        r_d = []

        for index, item in i.iterrows():
            if type(item['Enterals (name=amount/caloric-density;)']).__name__ == 'str':
                match = re.match(ENTERAL_PATTERN, item['Enterals (name=amount/caloric-density;)'])
                if match is not None:
                    flag = True
                    feed_type, feed_amt, feed_den = match.groups()
                    raw_days.append((item['DOL'], feed_type, feed_amt, feed_den))
                    r_d.append(item['DOL'])
            else:
                if item['LipOrder'] == True:
                    r_d.append(item['DOL'])
                else:
                    continue

        # for first week, they must have feeding
        for day in range(1, 9):
            if day not in r_d:
                raw_days.append((day, 'Breastfeeding', DEFAULT_FEEDING_QTY_D[day][0], 23))
        ids.append(aid)
        raw_days = sorted(raw_days)
        entry = [0] * FEEDING_VECTOR_SIZE * max_day
        entry_len = FEEDING_VECTOR_SIZE * max_day

        day_type_list = []

        # initialize result arrays
        if longitudinal_period < 15:
            features_day = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
        elif longitudinal_period < 29:
            features_day = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
        else:
            features_day = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

        # populate day features
        if 1 in r_d:
            features_day[:6] = [0, 0, 0, 0, 0, 0]
        if 15 in r_d and len(features_day) >= 12:
            features_day[6:12] = [0, 0, 0, 0, 0, 0]
        if 29 in r_d and len(features_day) >= 18:
            features_day[12:] = [0, 0, 0, 0, 0, 0]

        for day, feed_type, feed_amt, feed_den in raw_days:
            if day == 1:
                features_day[:6] = [0, 0, 0, 0, 0, 0]
                features_day[FEEDING_TYPE[feed_type]] = 1
                if float(feed_amt) > 0.1:
                    features_day[3+FEEDING_TYPE[feed_type]] = float(feed_amt)
                else:
                    features_day[3+FEEDING_TYPE[feed_type]] = DEFAULT_FEEDING_QTY_D[1][FEEDING_TYPE[feed_type]]
            elif day == 15 and len(features_day) >= 12:
                features_day[6:12] = [0, 0, 0, 0, 0, 0]
                features_day[6+FEEDING_TYPE[feed_type]] = 1
                if float(feed_amt) > 0.1:
                    features_day[9+FEEDING_TYPE[feed_type]] = float(feed_amt)
                else:
                    features_day[9+FEEDING_TYPE[feed_type]] = DEFAULT_FEEDING_QTY_D[-1][FEEDING_TYPE[feed_type]]
            elif day == 29 and len(features_day) >= 18:
                features_day[12:] = [0, 0, 0, 0, 0, 0]
                features_day[12+FEEDING_TYPE[feed_type]] = 1
                if float(feed_amt) > 0.1:
                    features_day[15+FEEDING_TYPE[feed_type]] = float(feed_amt)
                else:
                    features_day[15+FEEDING_TYPE[feed_type]] = DEFAULT_FEEDING_QTY_D[-1][FEEDING_TYPE[feed_type]]

        # populate week features
        first_week_flag = True
        for i in features_day[:6]:
            first_week_flag = first_week_flag and (math.isnan(i))

        if first_week_flag:
            features_day[:6] = [1, 0, 0, DEFAULT_FEEDING_QTY_D[1][0], 0, 0]

        features_days.append(features_day)

        for day, feed_type, feed_amt, feed_den in raw_days:
            if day == 1:
                continue
            day = day - 1
            idx = int((day - 1) / 7) * FEEDING_VECTOR_SIZE + FEEDING_TYPE[feed_type]
            if idx >= entry_len:
                break
            if (day, FEEDING_TYPE[feed_type]) not in day_type_list:
                entry[idx] += 1
                day_type_list.append((day, FEEDING_TYPE[feed_type]))
            if float(feed_amt) > 0.1:
                entry[idx + int(FEEDING_VECTOR_SIZE / 2)] += float(feed_amt)
            else:
                if day < 14:
                    entry[idx + int(FEEDING_VECTOR_SIZE / 2)] += DEFAULT_FEEDING_QTY_D[day+1][FEEDING_TYPE[feed_type]]
                else:
                    entry[idx + int(FEEDING_VECTOR_SIZE / 2)] += DEFAULT_FEEDING_QTY_D[-1][FEEDING_TYPE[feed_type]]
        for day in range(0, entry_len, FEEDING_VECTOR_SIZE):
            if sum(entry[day:day + int(FEEDING_VECTOR_SIZE / 2)]) == 0:
                entry[day:day + FEEDING_VECTOR_SIZE] = [np.NaN for _ in range(FEEDING_VECTOR_SIZE)]
        features.append(entry)

    # add default numbers for the first week
    for i in range(len(features)):
        if len(features[i]) > 0 and np.isnan(features[i][0]):
            features[i][:FEEDING_VECTOR_SIZE] = [7, 0, 0, sum([DEFAULT_FEEDING_QTY_D[period][0] for period in range(2, 9)]), 0, 0]

    # convert to pandas dataframe
    features = np.array(features)
    features_days = np.array(features_days)
    table_dict = {}
    table_dict['Astarte ID'] = ids

    if use_day_features:
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[0]}'] = features_days[:, 0]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[1]}'] = features_days[:, 1]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[2]}'] = features_days[:, 2]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[3]}'] = features_days[:, 3]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[4]}'] = features_days[:, 4]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[5]}'] = features_days[:, 5]

        if features_days.shape[1] >= 12:
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[0]}'] = features_days[:, 6]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[1]}'] = features_days[:, 7]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[2]}'] = features_days[:, 8]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[3]}'] = features_days[:, 9]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[4]}'] = features_days[:, 10]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[5]}'] = features_days[:, 11]

        if features_days.shape[1] >= 18:
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[0]}'] = features_days[:, 12]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[1]}'] = features_days[:, 13]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[2]}'] = features_days[:, 14]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[3]}'] = features_days[:, 15]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[4]}'] = features_days[:, 16]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[5]}'] = features_days[:, 17]

    for i in range(features.shape[1]):
        table_dict[f'feeding_day_{int(i / FEEDING_VECTOR_SIZE) * 7 + 2}-{min((int(i / FEEDING_VECTOR_SIZE) + 1) * 7 + 1, longitudinal_period)}_{FEEDING_FEATURE_STR[i % FEEDING_VECTOR_SIZE]}'] = features[:, i].tolist()

    result = pd.DataFrame(table_dict)
    return result


def feeding_features_weekly(feeding_fpath, longitudinal_period = 1, use_day_features = False):
    """
    Process the feeding features into weeks and return a processed pandas dataframe

    Parameters:
    feeding_fpath         (str): File path to the feeding csv file
    longitudinal_period   (int): end date of the longitudinal period in use
    use_day_features     (bool): Include day features or not (day 1, 15, 29, 
                                 etc). Default to false. 

    Returns:
    Pandas DataFrame: pandas table containing feeding features
    """
    FEEDING_VECTOR_SIZE = 6
    max_day = int(longitudinal_period / 7)
    feeding_table = pd.read_csv(feeding_fpath).drop_duplicates()
    features = []
    features_days = []
    ids = []

    # get numbers from the raw csv file
    g = feeding_table.groupby('Astarte ID')
    for aid, i in g:
        flag = False
        raw_days = []

        r_d = []

        for index, item in i.iterrows():
            if type(item['Enterals (name=amount/caloric-density;)']).__name__ == 'str':
                match = re.match(ENTERAL_PATTERN, item['Enterals (name=amount/caloric-density;)'])
                if match is not None:
                    flag = True
                    feed_type, feed_amt, feed_den = match.groups()
                    raw_days.append((item['DOL'], feed_type, feed_amt, feed_den))
                    r_d.append(item['DOL'])
            else:
                if item['LipOrder'] == True:
                    r_d.append(item['DOL'])
                else:
                    continue

        # they must have feeding for first week
        for day in range(1, 8):
            if day not in r_d:
                raw_days.append((day, 'Breastfeeding', DEFAULT_FEEDING_QTY_D[day][0], 23))
        ids.append(aid)
        raw_days = sorted(raw_days)
        entry = [0] * FEEDING_VECTOR_SIZE * max_day
        entry_len = FEEDING_VECTOR_SIZE * max_day

        day_type_list = []

        # initialize result arrays
        if longitudinal_period < 15:
            features_day = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
        elif longitudinal_period < 29:
            features_day = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
        else:
            features_day = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

        # populate day features
        if 1 in r_d:
            features_day[:6] = [0, 0, 0, 0, 0, 0]
        if 15 in r_d and len(features_day) >= 12:
            features_day[6:12] = [0, 0, 0, 0, 0, 0]
        if 29 in r_d and len(features_day) >= 18:
            features_day[12:] = [0, 0, 0, 0, 0, 0]

        for day, feed_type, feed_amt, feed_den in raw_days:
            if day == 1:
                features_day[:6] = [0, 0, 0, 0, 0, 0]
                features_day[FEEDING_TYPE[feed_type]] = 1
                if float(feed_amt) > 0.1:
                    features_day[3+FEEDING_TYPE[feed_type]] = float(feed_amt)
                else:
                    features_day[3+FEEDING_TYPE[feed_type]] = DEFAULT_FEEDING_QTY_D[1][FEEDING_TYPE[feed_type]]
            elif day == 15 and len(features_day) >= 12:
                features_day[6:12] = [0, 0, 0, 0, 0, 0]
                features_day[6+FEEDING_TYPE[feed_type]] = 1
                if float(feed_amt) > 0.1:
                    features_day[9+FEEDING_TYPE[feed_type]] = float(feed_amt)
                else:
                    features_day[9+FEEDING_TYPE[feed_type]] = DEFAULT_FEEDING_QTY_D[-1][FEEDING_TYPE[feed_type]]
            elif day == 29 and len(features_day) >= 18:
                features_day[12:] = [0, 0, 0, 0, 0, 0]
                features_day[12+FEEDING_TYPE[feed_type]] = 1
                if float(feed_amt) > 0.1:
                    features_day[15+FEEDING_TYPE[feed_type]] = float(feed_amt)
                else:
                    features_day[15+FEEDING_TYPE[feed_type]] = DEFAULT_FEEDING_QTY_D[-1][FEEDING_TYPE[feed_type]]

        features_days.append(features_day)

        # populate week features
        for day, feed_type, feed_amt, feed_den in raw_days:
            day = day - 1
            idx = int(day / 7) * FEEDING_VECTOR_SIZE + FEEDING_TYPE[feed_type]
            if idx >= entry_len:
                break
            if (day, FEEDING_TYPE[feed_type]) not in day_type_list:
                entry[idx] += 1
                day_type_list.append((day, FEEDING_TYPE[feed_type]))
            if float(feed_amt) > 0.1:
                entry[idx + int(FEEDING_VECTOR_SIZE / 2)] += float(feed_amt)
            else:
                if day < 14:
                    entry[idx + int(FEEDING_VECTOR_SIZE / 2)] += DEFAULT_FEEDING_QTY_D[day+1][FEEDING_TYPE[feed_type]]
                else:
                    entry[idx + int(FEEDING_VECTOR_SIZE / 2)] += DEFAULT_FEEDING_QTY_D[-1][FEEDING_TYPE[feed_type]]
        for day in range(0, entry_len, FEEDING_VECTOR_SIZE):
            if sum(entry[day:day + int(FEEDING_VECTOR_SIZE / 2)]) == 0:
                entry[day:day + FEEDING_VECTOR_SIZE] = [np.NaN for _ in range(FEEDING_VECTOR_SIZE)]
        features.append(entry)

    # add default numbers for the first week
    for i in range(len(features)):
        if len(features[i]) > 0 and np.isnan(features[i][0]):
            features[i][:FEEDING_VECTOR_SIZE] = [7, 0, 0, sum([DEFAULT_FEEDING_QTY_D[period][0] for period in range(1, 8)]), 0, 0]

    # convert to pandas dataframe
    features = np.array(features)
    features_days = np.array(features_days)
    table_dict = {}
    table_dict['Astarte ID'] = ids

    # add day features if necessary
    if use_day_features:
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[0]}'] = features_days[:, 0]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[1]}'] = features_days[:, 1]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[2]}'] = features_days[:, 2]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[3]}'] = features_days[:, 3]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[4]}'] = features_days[:, 4]
        table_dict[f'feeding_day_1_{FEEDING_FEATURE_STR[5]}'] = features_days[:, 5]

        if features_days.shape[1] >= 12:
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[0]}'] = features_days[:, 6]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[1]}'] = features_days[:, 7]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[2]}'] = features_days[:, 8]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[3]}'] = features_days[:, 9]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[4]}'] = features_days[:, 10]
            table_dict[f'feeding_day_15_{FEEDING_FEATURE_STR[5]}'] = features_days[:, 11]

        if features_days.shape[1] >= 18:
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[0]}'] = features_days[:, 12]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[1]}'] = features_days[:, 13]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[2]}'] = features_days[:, 14]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[3]}'] = features_days[:, 15]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[4]}'] = features_days[:, 16]
            table_dict[f'feeding_day_29_{FEEDING_FEATURE_STR[5]}'] = features_days[:, 17]

    # convert to pandas dataframe
    for i in range(features.shape[1]):
        table_dict[f'feeding_week{int(i / FEEDING_VECTOR_SIZE + 1)}_{FEEDING_FEATURE_STR[i % FEEDING_VECTOR_SIZE]}'] = features[:, i].tolist()

    result = pd.DataFrame(table_dict)
    return result

def probiotics_features(fpath, label_fpath, longitudinal_period = 1, is_regression = False, use_day_features = False):
    """
    Process the probiotics features into day 2-8, etc and return a processed pandas dataframe
    Note it's generating boolean values for specific types of probiotics

    Parameters:
    fpath                 (str): File path to the probiotics csv file
    label_fpath           (str): File path to the labels csv file
    longitudinal_period   (int): end date of the longitudinal period in use
    is_regression        (bool): Use as regression feature. Default to false
    use_day_features     (bool): Include day features or not (day 1, 15, 29, 
                                 etc). Default to false. 

    Returns:
    Pandas DataFrame: pandas table containing probiotics features
    """
    t = pd.read_csv(fpath).drop(labels = 'Development at 2', axis = 1).drop_duplicates()
    label_t = pd.read_csv(label_fpath)

    # has probiotics -> 0, Infloran -> 1, LB2 -> 2
    raw_entries = {}
    features_day = {}

    # initialize empty arrays
    for index, item in label_t.iterrows():
        raw_entries[item['Astarte ID']] = [0] * (3 * int(longitudinal_period / 7))
        if longitudinal_period < 15:
            features_day[item['Astarte ID']] = [0, 0, 0]
        elif longitudinal_period < 29:
            features_day[item['Astarte ID']] = [0, 0, 0, 0, 0, 0]
        else:
            features_day[item['Astarte ID']] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # getting the numbers from the raw csv file
    for index, item in t.iterrows():
        if item['Probiotics Type'] == 'Infloran':
            if item['Astarte ID'] not in raw_entries:
                continue
            if np.isnan(item['Probiotic stop']) or item['Probiotic stop'] > longitudinal_period-1:
                item['Probiotic stop'] = longitudinal_period

            aid = item['Astarte ID']
            start = int(item['Probiotic start'])
            end = int(item['Probiotic stop'])

            # populate day features
            if start <= 1 and end > 1:
                features_day[aid][0] = 1
                features_day[aid][1]+= 1
            if longitudinal_period >= 15 and start <= 15 and end > 15:
                features_day[aid][3] = 1
                features_day[aid][4] += 1
            if longitudinal_period >= 29 and start <= 29 and end > 29:
                features_day[aid][6] = 1
                features_day[aid][7] += 1

            # populate week features
            start_week = int((item['Probiotic start'] - 2) / 7)
            stop_week = int((item['Probiotic stop'] - 2) / 7)
            if stop_week == 0:
                stop_week = -1
            for i in range(start_week, stop_week + 1):
                raw_entries[item['Astarte ID']][3 * i] = 1
                raw_entries[item['Astarte ID']][3 * i + 1] += 1
        elif item['Probiotics Type'] == 'LB2':
            if item['Astarte ID'] not in raw_entries:
                continue
            if np.isnan(item['Probiotic stop']) or item['Probiotic stop'] >= longitudinal_period-1:
                item['Probiotic stop'] = longitudinal_period-1

            aid = item['Astarte ID']
            start = int(item['Probiotic start'])
            end = int(item['Probiotic stop'])

            # populate day features
            if start <= 1 and end > 1:
                features_day[aid][0] = 1
                features_day[aid][2] += 1
            if longitudinal_period >= 15 and start <= 15 and end > 15:
                features_day[aid][3] = 1
                features_day[aid][5] += 1
            if longitudinal_period >= 29 and start <= 29 and end > 29:
                features_day[aid][6] = 1
                features_day[aid][8] += 1

            # populate week features
            start_week = int((item['Probiotic start'] - 2) / 7)
            stop_week = int((item['Probiotic stop'] - 2) / 7)
            if stop_week == 0:
                stop_week = -1
            for i in range(start_week, stop_week + 1):
                raw_entries[item['Astarte ID']][3 * i] = 1
                raw_entries[item['Astarte ID']][3 * i + 2] += 1
    
    # convert to pandas dataframe
    features = {}
    features['Astarte ID'] = []

    if use_day_features:
        features[f'probiotics_day_1_has_probiotics'] = []
        features[f'probiotics_day_1_has_infloran'] = []
        features[f'probiotics_day_1_has_lb2'] = []

        if longitudinal_period >= 15:
            features[f'probiotics_day_15_has_probiotics'] = []
            features[f'probiotics_day_15_has_infloran'] = []
            features[f'probiotics_day_15_has_lb2'] = []
        if longitudinal_period >= 29:
            features[f'probiotics_day_29_has_probiotics'] = []
            features[f'probiotics_day_29_has_infloran'] = []
            features[f'probiotics_day_29_has_lb2'] = []  

    for i in range(int(longitudinal_period / 7)):
        features[f'probiotics_day_{i * 7 + 2}-{(i + 1) * 7 + 1}_has_probiotics'] = []
        features[f'probiotics_day_{i * 7 + 2}-{(i + 1) * 7 + 1}_infloran'] = []
        features[f'probiotics_day_{i * 7 + 2}-{(i + 1) * 7 + 1}_lb2'] = []
    
    for i in raw_entries:
        features['Astarte ID'].append(i)

        if use_day_features:
            features[f'probiotics_day_1_has_probiotics'].append(features_day[i][0])
            features[f'probiotics_day_1_has_infloran'].append(features_day[i][1])
            features[f'probiotics_day_1_has_lb2'].append(features_day[i][2])

            if longitudinal_period >= 15:
                features[f'probiotics_day_15_has_probiotics'].append(features_day[i][3])
                features[f'probiotics_day_15_has_infloran'].append(features_day[i][4])
                features[f'probiotics_day_15_has_lb2'].append(features_day[i][5])
            if longitudinal_period >= 29:
                features[f'probiotics_day_29_has_probiotics'].append(features_day[i][6])
                features[f'probiotics_day_29_has_infloran'].append(features_day[i][7])
                features[f'probiotics_day_29_has_lb2'].append(features_day[i][8])

        for j in range(int(longitudinal_period / 7)):
            features[f'probiotics_day_{j * 7 + 2}-{(j + 1) * 7 + 1}_has_probiotics'].append(raw_entries[i][3 * j])
            features[f'probiotics_day_{j * 7 + 2}-{(j + 1) * 7 + 1}_infloran'].append(raw_entries[i][3 * j + 1])
            features[f'probiotics_day_{j * 7 + 2}-{(j + 1) * 7 + 1}_lb2'].append(raw_entries[i][3 * j + 2])
    
    return pd.DataFrame(features)

def probiotics_features_weekly(fpath, label_fpath, longitudinal_period = 1, is_regression = False, use_day_features = False):
    """
    Process the probiotics features into weeks and return a processed pandas dataframe

    Parameters:
    fpath                 (str): File path to the probiotics csv file
    label_fpath           (str): File path to the labels csv file
    longitudinal_period   (int): end date of the longitudinal period in use
    is_regression        (bool): Use as regression feature. Default to false
    use_day_features     (bool): Include day features or not (day 1, 15, 29, 
                                 etc). Default to false. 

    Returns:
    Pandas DataFrame: pandas table containing probiotics features
    """
    if is_regression:
        longitudinal_period += 1
    t = pd.read_csv(fpath).drop(labels = 'Development at 2', axis = 1).drop_duplicates()
    label_t = pd.read_csv(label_fpath)

    # has probiotics -> 0, Infloran -> 1, LB2 -> 2
    raw_entries = {}
    features_day = {}

    for index, item in label_t.iterrows():
        raw_entries[item['Astarte ID']] = [0] * (3 * int(longitudinal_period / 7))
        if longitudinal_period < 15:
            features_day[item['Astarte ID']] = [0, 0, 0]
        elif longitudinal_period < 29:
            features_day[item['Astarte ID']] = [0, 0, 0, 0, 0, 0]
        else:
            features_day[item['Astarte ID']] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # getting the numbers from the raw csv file
    for index, item in t.iterrows():
        if item['Probiotics Type'] == 'Infloran':
            if item['Astarte ID'] not in raw_entries:
                continue
            if np.isnan(item['Probiotic stop']) or item['Probiotic stop'] >= longitudinal_period-1:
                item['Probiotic stop'] = longitudinal_period-1

            aid = item['Astarte ID']
            start = int(item['Probiotic start'])
            end = int(item['Probiotic stop'])

            # populate day features
            if start <= 1 and end > 1:
                features_day[aid][0] = 1
                features_day[aid][1] += 1
            if longitudinal_period >= 15 and start <= 15 and end > 15:
                features_day[aid][3] = 1
                features_day[aid][4] += 1
            if longitudinal_period >= 29 and start <= 29 and end > 29:
                features_day[aid][6] = 1
                features_day[aid][7] += 1

            # populate week features
            start_week = int((item['Probiotic start'] - 1) / 7)
            stop_week = int((item['Probiotic stop'] - 1) / 7)
            if stop_week == 0:
                stop_week = -1
            for i in range(start_week, stop_week + 1):
                raw_entries[item['Astarte ID']][3 * i] = 1
                raw_entries[item['Astarte ID']][3 * i + 1] += min((i + 1) * 7, longitudinal_period-1, end) - max(i * 7 + 1, start) + 1

        elif item['Probiotics Type'] == 'LB2':
            if item['Astarte ID'] not in raw_entries:
                continue
            if np.isnan(item['Probiotic stop']) or item['Probiotic stop'] >= longitudinal_period:
                item['Probiotic stop'] = longitudinal_period-1

            aid = item['Astarte ID']
            start = int(item['Probiotic start'])
            end = int(item['Probiotic stop'])

            # populate day features
            if start <= 1 and end > 1:
                features_day[aid][0] = 1
                features_day[aid][2] += 1
            if longitudinal_period >= 15 and start <= 15 and end > 15:
                features_day[aid][3] = 1
                features_day[aid][5] += 1
            if longitudinal_period >= 29 and start <= 29 and end > 29:
                features_day[aid][6] = 1
                features_day[aid][8] += 1

            # populate week features
            start_week = int((item['Probiotic start'] - 1) / 7)
            stop_week = int((item['Probiotic stop'] - 1) / 7)
            if stop_week == 0:
                stop_week = -1
            for i in range(start_week, stop_week + 1):
                raw_entries[item['Astarte ID']][3 * i] = 1
                raw_entries[item['Astarte ID']][3 * i + 2] += min((i + 1) * 7, longitudinal_period-1, end) - max(i * 7 + 1, start) + 1
    
    # convert to pandas dataframe
    features = {}
    features['Astarte ID'] = []

    if use_day_features:
        features[f'probiotics_day_1_has_probiotics'] = []
        features[f'probiotics_day_1_has_infloran'] = []
        features[f'probiotics_day_1_has_lb2'] = []

        if longitudinal_period >= 15:
            features[f'probiotics_day_15_has_probiotics'] = []
            features[f'probiotics_day_15_has_infloran'] = []
            features[f'probiotics_day_15_has_lb2'] = []
        if longitudinal_period >= 29:
            features[f'probiotics_day_29_has_probiotics'] = []
            features[f'probiotics_day_29_has_infloran'] = []
            features[f'probiotics_day_29_has_lb2'] = []  

    for i in range(int(longitudinal_period / 7)):
        features[f'probiotics_week{i+1}_has_probiotics'] = []
        features[f'probiotics_week{i+1}_infloran'] = []
        features[f'probiotics_week{i+1}_lb2'] = []
    for i in raw_entries:
        features['Astarte ID'].append(i)

        if use_day_features:
            features[f'probiotics_day_1_has_probiotics'].append(features_day[i][0])
            features[f'probiotics_day_1_has_infloran'].append(features_day[i][1])
            features[f'probiotics_day_1_has_lb2'].append(features_day[i][2])

            if longitudinal_period >= 15:
                features[f'probiotics_day_15_has_probiotics'].append(features_day[i][3])
                features[f'probiotics_day_15_has_infloran'].append(features_day[i][4])
                features[f'probiotics_day_15_has_lb2'].append(features_day[i][5])
            if longitudinal_period >= 29:
                features[f'probiotics_day_29_has_probiotics'].append(features_day[i][6])
                features[f'probiotics_day_29_has_infloran'].append(features_day[i][7])
                features[f'probiotics_day_29_has_lb2'].append(features_day[i][8])

        for j in range(int(longitudinal_period / 7)):
            features[f'probiotics_week{j+1}_has_probiotics'].append(raw_entries[i][3 * j])
            features[f'probiotics_week{j+1}_infloran'].append(raw_entries[i][3 * j + 1])
            features[f'probiotics_week{j+1}_lb2'].append(raw_entries[i][3 * j + 2])
    
    return pd.DataFrame(features)

def merge_features(features, method = 'inner'):
    """
    Utility function to merge all features

    Parameters:
    features             (list): List of feature tables to merge
    method                (str): method to merge (inner, outer, left, right)

    Returns:
    Pandas DataFrame: pandas table containing merged features
    """
    res = features[0]
    for i in features[1:]:
        res = res.merge(i, on='Astarte ID', how=method)
    return res
