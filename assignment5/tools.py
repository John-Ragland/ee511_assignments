import numpy as np
import pandas as pd
import os

class Data():
    def __init__(self):
        pass

    def load_data(self):
        '''
        load's data from file and formats to relevent format
        '''

        # Check if data files exist
        if not (os.path.exists('Data/test.tsv') and os.path.exists('Data/train.tsv') and os.path.exists('Data/val.tsv')):
            raise Exception ('Data files missings, please add "train.tsv", "test.tsv", or "val.tsv" to ./Data')

        train_file = 'Data/train.tsv'
        test_file = 'Data/test.tsv'
        val_file = 'Data/val.tsv'

        col_names = ['lan','tweet']
        val = pd.read_csv(val_file, header=None, sep='\t', quoting=3)
        val.columns = col_names
        train = pd.read_csv(train_file, header=None, sep='\t', quoting=3)
        train.columns = col_names
        test = pd.read_csv(test_file, header=None, sep='\t', quoting=3)
        test.columns = col_names

        return val, train, test