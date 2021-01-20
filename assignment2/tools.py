import numpy as np
import sklearn
import pandas as pd

'''
Toolbox for Assignment 2 - ee511 - Statistical Learning
University of Washington
Code by John Ragland and Doruk Arisoy
Winter 2021
'''

def load_data():
    '''
    load_data() - read data from url and seperate into training, validation
        and testing
    
    Returns
    -------
    train : pandas.dataframe()
        training data set
    valid : pandas.dataframe()
        validation data set
    test : pandas.dataframe()
        testing data set
    '''
    url = 'http://jse.amstat.org/v19n3/decock/AmesHousing.txt'

    df = pd.read_csv(url, sep='\t')

    valid = df[(df['Order'] % 5) == 3]
    test = df[(df['Order'] % 5) == 4]
    mask = ((df['Order'] % 5) != 3) & ((df['Order'] % 5) != 4)
    train = df[mask]

    return train, valid, test