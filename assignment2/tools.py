from matplotlib.pyplot import axis
import numpy as np
from pandas.core.arrays.sparse import dtype
import sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

'''
Toolbox for Assignment 2 - ee511 - Statistical Learning
University of Washington
Code by John Ragland and Doruk Arisoy
Winter 2021
'''

def get_numerical_variables():
    '''
    get_numerical_variables() - get the list of numerical class names

    Returns: array
        list of numerical class names
    '''
    # numerical from Doruk
    """
    return [
        'SalePrice', 'Lot Frontage', 'Lot Area', 'Mas Vnr Area', 
        'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 
        '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 
        'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', 
        '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 
        'Order', 'Year Built', 'Year Remod/Add', 'Bsmt Full Bath', 
        'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom', 
        'Kitchen', 'TotRmsAbvGrd', 'Fireplaces', 'Garage Yr Blt', 
        'Garage Cars', 'Mo Sold', 'Yr Sold']
    """
    numerical_variables = [
        'Lot Area', 'Lot Frontage', 'Year Built',
        'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
        'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',
        '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area',
        'Garage Area', 'Wood Deck SF', 'Open Porch SF',
        'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
        'Pool Area']

    return numerical_variables

def get_categorical_variables():
    '''
    get_categorical_variables() - get the list of categorical variables

    Returns: array
        list of categorical class names
    '''
    # categorical_variables from Doruk
    """return [
        'MS SubClass', 'MS Zoning', 'Street', 'Alley',
        'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1',
        'Condition 2', 'Bldg Type', 'House Style', 'Roof Style',
        'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
        'Foundation', 'Heating', 'Central Air', 'Garage Type',
        'Misc Feature', 'Sale Type', 'Sale Condition', 'PID']
    """
    categorical_variables = ['MS SubClass', 'MS Zoning', 'Street',
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
    return categorical_variables

def get_ordinal_variables():
    '''
    get_ordinal_variables() - get the list of ordinal variables

    Returns: array
        list of ordinal class names
    '''
    return [
        'Lot Shape', 'Utilities', 'Land Slope', 'Overall Qual', 
        'Overall Cond', 'Exter Qual', 'Exter Cond', 'Bsmt Qual', 
        'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 
        'Heating QC', 'Electrical', 'Kitchen Qual', 'Functional', 
        'Fireplace Qu', 'Garage Finish', 'Garage Qual', 'Garage Cond', 
        'Paved Drive', 'Pool QC', 'Fence']

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
        get_numerical_variables() + get_categorical_variables() + \
        ['Order'] + ['SalePrice']]

    # Change numerical columns with NaN to 0
    for name in get_numerical_variables():
        df[name] = df[name].fillna(0)

    # Change string columns with NaN to 'null'
    for name in get_categorical_variables():
        df[name] = df[name].fillna('null')
        df[name] = df[name].astype(str)

    df = replace_categorical_with_one_hot(df)
    
    # Seperate into training, testing, and validation (and remove Order column)
    df.loc[:, df.columns != 'b']
    valid = df[(df['Order'] % 5) == 3].loc[:, df.columns != 'Order']
    test = df[(df['Order'] % 5) == 4].loc[:, df.columns != 'Order']
    
    mask = ((df['Order'] % 5) != 3) & ((df['Order'] % 5) != 4)
    train = df[mask].loc[:, df.columns != 'Order']

    return train, valid, test

def replace_categorical_with_one_hot(df):
    '''
    replace_categorical_with_one_hot - replace the categorical and ordinal columns
        of the data set with one-hot encoded columns

    df : pandas.dataframe()
        data set that contains categorical and ordinal cols

    Returns : pandas.dataframe()
        data set with one-hot encoded columns
    '''
    for original_col in (get_categorical_variables()):
        one_hot_matrix = one_hot(np.array(df[original_col]))
        _, new_cols = one_hot_matrix.shape
        for i in range(new_cols):
            df["%s_%d" % (original_col, i)] = one_hot_matrix[:,i]
        df = df.drop(original_col, 1)
    return df

def one_hot(column):
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

def calculate_OLS(data_matrix, responses):
    '''
    calculate_OLS - calculates the weights and shift for ordinary least squares
        linear regression

    data_matrix : numpy.matrix()
        input training data
    responses : numpy.matrix()
        column vector output of the training data

    Returns : numpy.matrix()
        weights of the least squares linear regression model
    '''
    data_matrix = add_col_of_ones(data_matrix).T
    return np.linalg.lstsq(data_matrix, responses, rcond=None)[0]

def predict(x, weights):
    '''
    predict - given the linear regression weights and input data, predict
        the output

    x : numpy.matrix()
        input data
    weights : numpy.matrix()
        weights of the least squares linear regression model

    Returns : numpy.matrix()
        column vector of prediction results
    '''
    x = add_col_of_ones(x)
    return np.matmul(x.T, weights)

def add_col_of_ones(x):
    '''
    add_col_of_ones - appends a column of ones to the given matrix

    x : numpy.matrix()
        input matrix

    Returns: numpy.matrix()
        intput matrix with an additional column of ones
    '''
    return np.vstack([x.T, np.ones(len(x))])

def rmse(actual, prediction):
    '''
    rmse - calculate the root mean squared error of given data

    actual : numpy.array()
        actaul data to compare predictions to
    predictions : numpy.array()
        predicted data to calculate the error of

    Return : float
        the root mean squared error of the predictions
    '''
    return mean_squared_error(actual, prediction, squared=False)

