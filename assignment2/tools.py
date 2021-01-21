from matplotlib.pyplot import axis
import numpy as np
import sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error

'''
Toolbox for Assignment 2 - ee511 - Statistical Learning
University of Washington
Code by John Ragland and Doruk Arisoy
Winter 2021
'''

def load_data(numerical_variables):
    '''
    load_data() - read data from url and seperate into training, validation
        and testing
    
    numerical_variables : set
        name of the columns that are numerical

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
    df = fill_nans(df, numerical_variables)
    valid = df[(df['Order'] % 5) == 3]
    test = df[(df['Order'] % 5) == 4]
    mask = ((df['Order'] % 5) != 3) & ((df['Order'] % 5) != 4)
    train = df[mask]

    return train, valid, test

def fill_nans(df, numerical_variables):
    '''
    fill_nans - fill missing data with 0 or empty string

    df : pandas.dataframe()
        data set that contains nans
    numerical_variables : set
        name of the columns that are numerical

    Returns
    -------
    df : pandas.dataframe()
        data set with replaced nans
    '''
    for col in df.columns:
        if col in numerical_variables:
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna('', inplace=True)
    return df

def calculate_OLS(data_matrix, responses):
    '''
    calculate_OLS - calculates the weights and shift for ordinary least squares
        linear regression

    data_matrix : numpy.matrix()
        input training data
    responses : numpy.matrix()
        column vector output of the training data

    Returns
    -------
    weights : numpy.matrix()
        weights of the least squares linear regression model
    shift : float
        how much the linear model be shifted
    '''
    weights = data_matrix.T.dot(data_matrix)
    if weights.ndim < 2:
         weights = data_matrix.T / weights
    else :
        weights = np.linalg.inv(weights)
        weights = weights.dot(data_matrix.T)

    weights = weights.dot(responses)

    shift = responses.mean()
    if weights.ndim < 2:
        shift -= weights * data_matrix.mean(axis=0)
    else:
        shift -= weights.T.dot(data_matrix.mean(axis=0))
    return weights, shift

def predict(x, weights, shift):
    '''
    predict - given the linear regression weights and input data, predict
        the output

    x : numpy.matrix()
        input data
    weights : numpy.matrix()
        weights of the least squares linear regression model
    shift : float
        how much the linear model be shifted

    Returns
    predictions : numpy.matrix()
        column vector of prediction results
    '''
    predictions = weights * x + shift
    return predictions

def rmse(actual, prediction):
    '''
    rmse - calculate the root mean squared error of given data

    actual : numpy.array()
        actaul data to compare predictions to
    predictions : numpy.array()
        predicted data to calculate the error of

    Return
    ------
    the root mean squared error of the predictions
    '''
    return mean_squared_error(actual, prediction, squared=False)