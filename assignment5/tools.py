import numpy as np
import pandas as pd
import os
from collections import Counter
import torch

Languages = {'en':'English', 'es':'Spanish', 'pt':'Portuguese', 'gl':'Galician', 'eu':'Basque', 'ca':'Catalan', 'fr':'French', 'it':'Italian', 'de':'German'}

class Data():
    def __init__(self):
        check_files_exists()

    def load_data(self):
        '''
        load's data from file and formats to relevent format
        '''
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

        self.get_vocab(train)

        # Convert Train, Test, Validation to list of Tensors
        print('Converting Training Data...')
        self.train = self.list_to_tensor(train['tweet'].to_list())
        print('Converting Test Data...')
        self.test = self.list_to_tensor(test['tweet'].to_list())
        print('Converting Validation Data...')
        self.val = self.list_to_tensor(val['tweet'].to_list())
    
    def get_vocab(self, train):
        train_char = [i for ele in train.tweet.to_list() for i in ele]
        most_common = Counter(train_char).most_common()
        char_limit = 10
        for k in range(len(most_common)):
            if most_common[k][1] < char_limit:
                break

        vocab = most_common[:k]
        vocab = [i[0] for i in vocab]

        vocab.insert(0,'<S>')   # start token
        vocab.insert(0,'</S>')  # end token
        vocab.insert(0,'<N>')   # out-of-vocabulary token
        self.vocab = vocab
    
    def list_to_tensor(self, data):
        tensor = [0]*len(data)
        for k, item in enumerate(data):
            tensor[k] = self.string_to_tensor(item)
            print(k/len(data)*100, end='\r')
        return tensor

    def string_to_tensor(self, line):
        # Based on https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
        tensor = torch.zeros(len(line)+2, 1, len(self.vocab))
        #add start character
        tensor[0][0][2] = 1
        
        for li, letter in enumerate(line):
            try:
                tensor[li+1][0][self.vocab.index(letter)] = 1
            #Add not in vocab character
            except(ValueError):
                tensor[li+1][0][0] = 1
                
        # add end character
        tensor[-1][0][1] = 1
        return tensor

def check_files_exists():
    # Check if data files exist
    if not (os.path.exists('Data/test.tsv') and 
            os.path.exists('Data/train.tsv') and 
            os.path.exists('Data/val.tsv')):
        raise Exception ('Data files missings, please add "train.tsv", "test.tsv", or "val.tsv" to ./Data')