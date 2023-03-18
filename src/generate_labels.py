# Filename: generate_labels.py
# Author: Siwei Xu
# Date: 09/17/2020
# 
# Usage: python generate_labels.py <patients_fpath> <comorbidities_fpath> 
#                                  <output_fpath> <option> <flip_label>
# 
# Option: td (for TD/NTD labels), gf (for GF/NGF labels), regress (for 36 weeks weight)
# Flip label: True for flipped label (NTD or NGF = 1); False otherwise
#
# Description: process the labels from the csv files of "patients" and 
# "comorbidities" sections. If the file itself is ran, it will try to find 
# the file paths and save the table as a csv file. 
#
# Note: for regression task, the default limit for PMA is 7 (week 35-37). It can be 
#       changed through the parameter at line 25.


import pandas as pd 
import numpy as np 

LABELS = ['NonTypDev', 'TypDev']
COMORBIDITIES_DTYPES = {'Astarte ID': float, 'Diagnosis': str}

REGRESSION_PMA_LIMIT = 7

# helper function to compute bitwise and in a pandas group
def bit_and(x):
    if x.dtype == object:
        return x.tolist()[0]
    elif x.dtype == float:
        return x.tolist()[0]
    l = x.tolist()
    if len(l) == 1:
        return l[0]
    else:
        r = l[0]
        for i in l[1:]:
            r = r | i 
        return r

def generate_labels(patients_fpath, comorbidities_fpath, opt = 'td', flip = False):
    """
    Process the labels from the csv files of "patients" and 
    "comorbidities" sections

    Parameters:
    patients_fpath      (str): File path to the patients csv file
    comorbidities_fpath (str): File path to the comorbidities csv file
    opt                 (str): Option for generating labels. 
                               - td for TD/NTD
                               - gf for GF/NGF
                               - regress for 36 weeks weight
    flip               (bool): Whether want flipped label (NTD or NGF = 1) or not

    Returns:
    Pandas DataFrame: pandas table containing both the Astarte ID and the labels
    """

    # load csv files
    patients_table = pd.read_csv(patients_fpath)
    comorbidities_table = pd.read_csv(comorbidities_fpath, header = 0, dtype = COMORBIDITIES_DTYPES)[['Astarte ID', 'Diagnosis']].dropna()
    
    # grab only essential parts from patients csv and ignore NULL values
    if opt == 'regress':
        patients_info = patients_table[['Astarte ID', 'Birth Z-Score', 'D/C Weight', 
                                        'D/C Z-Score', 'Died', 'Sequenced', 'Site', '36 Weeks PMA', '36 Weeks Weight']].dropna()
    else:
        patients_info = patients_table[['Astarte ID', 'Birth Z-Score', 'D/C Weight', 
                                        'D/C Z-Score', 'Died', 'Sequenced', 'Site']].dropna()
    patients_info = patients_info[patients_info['Died'] < 0.5]

    # find boolean values of growth failure and death
    patients_info['gf'] = (patients_info['D/C Z-Score'] - 
                          patients_info['Birth Z-Score']) <= -1.2
    patients_info['death'] = patients_info['Died'] == 1
    
    # join the diagnosis part to patients table to get sepsis information
    sepsis = comorbidities_table[['Astarte ID', 'Diagnosis']]
    patients_info = patients_info.merge(sepsis, left_on='Astarte ID', right_on='Astarte ID', how='left')

    # compute boolean values for diagnosis and sequenced
    if opt == 'td' or opt == 'gf':
        patients_info['sepsis'] = patients_info['Diagnosis'] == 'Sepsis'
        patients_info['nec'] = patients_info['Diagnosis'] == 'NEC'
        patients_info['Sequenced'] = patients_info['Sequenced'] > 0.5
        patients_info = patients_info[['Astarte ID', 'gf', 'sepsis', 'nec', 'Sequenced', 'Site']]
        patients_info = patients_info.groupby('Astarte ID').aggregate(bit_and)
    elif opt == 'regress':
        patients_info = patients_info[['Astarte ID', '36 Weeks PMA', '36 Weeks Weight', 'Sequenced', 'Site']]
        patients_info = patients_info[patients_info['36 Weeks PMA'] >= 36 * 7 - REGRESSION_PMA_LIMIT]
        patients_info = patients_info[patients_info['36 Weeks PMA'] <= 36 * 7 + REGRESSION_PMA_LIMIT]
        patients_info = patients_info.groupby('Astarte ID').aggregate(bit_and)

    # compute logical and for final labels
    if opt == 'td':
        patients_info['TypDev'] = (~patients_info['gf']) & \
                                  (~patients_info['sepsis']) & \
                                  (~patients_info['nec'])
    elif opt == 'gf':
        patients_info['TypDev'] = (~patients_info['gf'])
    elif opt == 'regress':
        patients_info['TypDev'] = patients_info['36 Weeks Weight']

    if opt != 'regress' and flip:
        patients_info['TypDev'] = ~patients_info['TypDev']
    
    # return as pandas dataframe
    return patients_info[['TypDev', 'Sequenced', 'Site']]

# Driver function, if being ran standalone, to retrieve args from command line
# and save the labels table as another csv file
if __name__ == '__main__':
    
    # grab command line args
    import sys
    patients_fpath = sys.argv[1]
    comorbidities_fpath = sys.argv[2]
    output_fpath = sys.argv[3]
    opt = sys.argv[4]
    flip = bool(sys.argv[5])
    if flip:
        LABELS = ['TypDev', 'NonTypDev']
    assert (opt == 'td' or opt == 'gf' or opt == 'regress')

    result = generate_labels(patients_fpath, comorbidities_fpath, opt = opt, flip = flip)
    result[['TypDev', 'Sequenced', 'Site']].to_csv(output_fpath)

    print(f'Successfully processed {len(result)} patients')
    if opt == 'td' or opt == 'gf':
        print(f'{sum(result["TypDev"])} positive labels, {sum(~result["TypDev"])} negative labels')
