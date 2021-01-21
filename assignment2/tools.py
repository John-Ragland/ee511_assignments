import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

'''
Toolbox for Assignment 2 - ee511 - Statistical Learning
University of Washington
Code by John Ragland and Doruk Arisoy
Winter 2021
'''

def check_sklearn_version():
    required = "0.24.1"
    if sklearn.__version__ < required:
        raise Exception("Please update sklearn. Sklearn version %s or higher required!" % required)

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

    # Throw out unneeded variables
    df = df[
        _get_numerical_variables() + _get_categorical_variables() + \
        ['Order'] + ['SalePrice']]

    df = _replace_categorical_with_one_hot(df)
    df = df.fillna(0)
    
    # Seperate into training, testing, and validation (and remove Order column)
    valid = df[(df['Order'] % 5) == 3].loc[:, df.columns != 'Order']
    test = df[(df['Order'] % 5) == 4].loc[:, df.columns != 'Order']
    
    mask = ((df['Order'] % 5) != 3) & ((df['Order'] % 5) != 4)
    train = df[mask].loc[:, df.columns != 'Order']

    return train, valid, test

def _get_numerical_variables():
    return ['Lot Area', 'Lot Frontage', 'Year Built',
        'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
        'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',
        '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area',
        'Garage Area', 'Wood Deck SF', 'Open Porch SF',
        'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
        'Pool Area']

def _get_categorical_variables():
    return ['MS SubClass', 'MS Zoning', 'Street',
        'Alley', 'Lot Shape', 'Land Contour',
        'Utilities', 'Lot Config', 'Land Slope',
        'Neighborhood', 'Condition 1', 'Condition 2',
        'Bldg Type', 'House Style', 'Overall Qual',
        'Overall Cond', 'Roof Style', 'Roof Matl',
        'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
        'Exter Qual', 'Exter Cond', 'Foundation',
        'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
        'BsmtFin Type 1', 'Heating', 'Heating QC',
        'Central Air', 'Electrical', 'Bsmt Full Bath',
        'Bsmt Half Bath', 'Full Bath', 'Half Bath',
        'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual',
        'TotRms AbvGrd', 'Functional', 'Fireplaces',
        'Fireplace Qu', 'Garage Type', 'Garage Cars',
        'Garage Qual', 'Garage Cond', 'Paved Drive',
        'Pool QC', 'Fence', 'Sale Type', 'Sale Condition']

def _replace_categorical_with_one_hot(df):
    '''
    replace_categorical_with_one_hot - replace the categorical and ordinal columns
        of the data set with one-hot encoded columns

    df : pandas.dataframe()
        data set that contains categorical and ordinal cols

    Returns : pandas.dataframe()
        data set with one-hot encoded columns
    '''
    for original_col in _get_categorical_variables():
        one_hot_matrix = _one_hot(np.array(df[original_col]))
        _, new_cols = one_hot_matrix.shape
        for i in range(new_cols):
            df["%s_%d" % (original_col, i)] = one_hot_matrix[:,i]
        df = df.drop(original_col, 1)
    return df

def _one_hot(column):
    '''
    one_hot - get the one-hot matrix of the given vector

    column : numpy.array()
        array of data that will be one hot encoded

    Return : array
        2D array of all one hot econded columns
    '''
    column = column.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(column)
    return enc.transform(column).toarray()