import numpy as np
import pandas as pd
import os
from collections import Counter
import torch

Languages = {'en':'English', 'es':'Spanish', 'pt':'Portuguese', 'gl':'Galician', 'eu':'Basque', 'ca':'Catalan', 'fr':'French', 'it':'Italian', 'de':'German'}

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
        


        vocab = self.get_vocab(train)

        # Convert Train, Test, Validation to list of Tensors
        train_tens = []
        test_tens = []
        val_tens = []

        train_ls = train['tweet'].to_list()
        test_ls = test['tweet'].to_list()
        val_ls = val['tweet'].to_list()
        print('Converting Training Data...')
        for k, item in enumerate(train_ls):
            train_tens.append(stringToTensor(item, vocab))
            print(k/len(train_ls)*100,end='\r')
        print('Converting Test Data...')
        for k, item in enumerate(test_ls):
            test_tens.append(stringToTensor(item, vocab))
            print(k/len(test_ls)*100,end='\r')
        print('Converting Validation Data...')
        for k, item in enumerate(val_ls):
            val_tens.append(stringToTensor(item, vocab))
            print(k/len(val_ls)*100,end='\r')


        self.val = val_tens
        self.train = train_tens
        self.test = test_tens
        return

    def get_vocab(self, train):
        train_list = train.tweet.to_list()
        train_char = [i for ele in train_list for i in ele]
        most_common = Counter(train_char).most_common()
        char_limit = 10
        for k in range(len(most_common)):
            if most_common[k][1] < 10:
                break
            pass

        vocab = most_common[:k]
        vocab = [i[0] for i in vocab]

        vocab.insert(0,'<S>')
        vocab.insert(0,'</S>')
        vocab.insert(0,'<N>')
        return vocab

# Based on https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
def stringToTensor(line, vocab):
    tensor = torch.zeros(len(line)+2, 1, len(vocab))
    #add start character
    tensor[0][0][2] = 1
    
    for li, letter in enumerate(line):
        try:
            tensor[li+1][0][vocab.index(letter)] = 1
        #Add not in vocab character
        except(ValueError):
            tensor[li+1][0][0] = 1
            
    # add end character
    tensor[-1][0][1] = 1
    return tensor