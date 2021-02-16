import numpy as np
import pandas as pd
import os
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack

Languages = {'en':'English', 'es':'Spanish', 'pt':'Portuguese', 'gl':'Galician', 'eu':'Basque', 'ca':'Catalan', 'fr':'French', 'it':'Italian', 'de':'German'}
lan_id = {
    'en':0, 
    'es':1, 
    'pt':2, 
    'gl':3, 
    'eu':4, 
    'ca':5, 
    'fr':6, 
    'it':7, 
    'de':8
}
class Data():
    def __init__(self):
        check_files_exists()

    def load_data(self):
        '''
        load's data from file and formats to relevent format
        
        Attributes
        ----------
        train : tensor
        test : tensor
        val : tensor
        test_labels : tensor
        train_labels : tensor
        val_labels : tensor
        train_pd : pd.df
        val_pd : pd.df
        test_pd : pd.df
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

        #save pandas dfs
        self.train_pd = train
        self.val_pd = val
        self.test_pd = test

        self.get_vocab(train)

        # Convert Train, Test, Validation to list of Tensors
        print('Converting Training Data...')
        trainls = self.list_to_tensor(train['tweet'].to_list())
        print('     Stacking Tensor...', end='\r')
        self.train = torch.cat(trainls, dim=1)
        print('Converting Test Data...')
        testls = self.list_to_tensor(test['tweet'].to_list())
        print('     Stacking Tensor...', end='\r')
        self.test = torch.cat(testls, dim=1)
        print('Converting Validation Data...')
        valls = self.list_to_tensor(val['tweet'].to_list())
        print('     Stacking Tensor...', end='\r')
        self.val = torch.cat(valls, dim=1)

        # Convert Labels to Tensors
        self.test_labels = self.label_to_tensor(test['lan'].to_numpy())
        self.train_labels = self.label_to_tensor(train['lan'].to_numpy())
        self.val_labels = self.label_to_tensor(val['lan'].to_numpy())

        print('Complete')
    
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
        tensor = torch.zeros(161, 1, len(self.vocab))
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

    def label_to_tensor(self,labels):
        '''
        labels : list
            list of labels for data
        '''
        labels_int = []
        for label in labels:
            labels_int.append(lan_id[label])
        labels_tensor = torch.tensor(labels_int)
        onehot = F.one_hot(labels_tensor)
        return onehot

def check_files_exists():
    # Check if data files exist
    if not (os.path.exists('Data/test.tsv') and 
            os.path.exists('Data/train.tsv') and 
            os.path.exists('Data/val.tsv')):
        raise Exception ('Data files missings, please add "train.tsv", "test.tsv", or "val.tsv" to ./Data')


class RNN_lan_class(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_lan):
        super(RNN_lan_class, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Create Character and Language embeddings (embed size hard set)
        self.char_embed = nn.Embedding(vocab_size, 14)
        self.lan_embed = nn.Embedding(num_lan, 5)

    def forward(self, input, hidden):
        '''
        input : tuple
            tuple of [current character, language]
        hidden : tensor
            previous hidden layer output
        '''

        character = self.char_embed(input[0])
        return character

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
