import numpy as np
import pandas as pd
import os
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import math

class Data():
    def __init__(self):
        self.train = load_data('Data/train.tsv')
        self.val = load_data('Data/val.tsv')
        self.test = load_data('Data/test.tsv')
        self.get_vocab()
        self.train_freq = get_freq(self.train, self.vocab)
        self.val_freq = get_freq(self.val, self.vocab)

    def get_vocab(self):
        chars = [i for ele in self.train.tweet.to_list() for i in ele]
        most_common = Counter(chars).most_common()
        char_limit = 10
        for k in range(len(most_common)):
            if most_common[k][1] < char_limit:
                break

        vocab = [i[0] for i in most_common[:k]]
        vocab.insert(0,'<S>')   # start token
        vocab.insert(0,'</S>')  # end token
        vocab.insert(0,'<N>')   # out-of-vocabulary token
        
        self.vocab = vocab
    
    def get_perplexity(self):
        train_freq = self.train_freq / self.train_freq.sum()
        val_freq = self.val_freq / self.val_freq.sum()
        train_freq[self.vocab.index('<S>')] = 1
        return np.exp(-(val_freq * np.log(train_freq)).sum())

def load_data(filename):
    check_files_exists(filename)
    data = pd.read_csv(filename, header=None, sep='\t', quoting=3)
    data.columns = ['lan','tweet']
    return data

def get_freq(data, vocab):
    freq = np.zeros(len(vocab))
    for tweet in data.tweet:
        for char in tweet:
            if char in vocab:
                freq[vocab.index(char)] += 1
            else:
                freq[vocab.index('<N>')] += 1
        freq[vocab.index('</S>')] += 1
    return freq

def get_data_loader(tweets, lans, batch_size, shuffle=False):
    data_tensor = torch.tensor(tweets, dtype=torch.long, device=torch.device("cpu"))
    label_tensor = torch.tensor(lans, dtype=torch.long, device=torch.device("cpu"))
    train_dataset = TensorDataset(data_tensor, label_tensor)
    return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle) 

def data_encoding(data, vocab, languages):
    tweets = tweet_enconding(data.tweet, vocab)
    langs = lang_encoding(data.lan, languages)
    return tweets, langs

def tweet_enconding(tweets, vocab, tweet_length=282):
    encoded = np.zeros((len(tweets), tweet_length))
    for t, tweet in enumerate(tweets):
        encoded[t][0] = vocab.index('<S>')
        for char in range(1, tweet_length-1):
            if char < len(tweet) and tweet[char] in vocab:
                encoded[t][char] = vocab.index(tweet[char])
            elif char < len(tweet):
                encoded[t][char] = vocab.index('<N>')
            else:
                encoded[t][char] = vocab.index('</S>')
        encoded[t][tweet_length-1] = vocab.index('</S>')
    return encoded

def lang_encoding(labels, languages, tweet_length=282):
    encoded = np.zeros((len(labels), tweet_length))
    for l, lang in enumerate(labels):
        idx = languages.index(lang)
        for char in range(0, tweet_length):
            encoded[l][char] = idx
    return encoded

def check_files_exists(filename):
    if not os.path.exists(filename):
        raise Exception ('Data files missings, please add %s.' % filename)

def train(model, device, train_loader, optimizer, epochs, log_interval, verbose=False):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output, hidden = model(data, label)
            loss = model.loss(output, label)
            loss.backward()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, pad):
    model.eval()
    loss = 0
    perp = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, label)
            loss += model.loss(output, label).item()
            perp += math.exp(F.cross_entropy(output.view(-1, 509), label.view(-1), ignore_index=pad))
    return loss, perp

def predict(model, device, data):
    '''
    lan - language id (0-8)
    '''
    model.eval()

    with torch.no_grad():

        for lan in range(9):
            label = torch.ones(data.size(), dtype=torch.long)*lan
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, label)
            
            # output = F.log_softmax(output, dim=2)
            #convert to numpy
            data_np = data.numpy()
            output_np = output.numpy()

            # calculate log prob for each letter of sequence (using output matrix)     
            prob = np.zeros(data_np.shape)
            for batch in range(output_np.shape[0]):
                for char in range(output_np.shape[1]):
                    prob[batch, char] = output_np[batch, char, data_np[batch, char]]

            if lan == 0:
                total_prob = np.sum(prob, axis=1)
            else:
                total_prob = np.vstack((np.sum(prob, axis=1),total_prob))
        
        # Choose language with highest character probability
        output = np.argmax(total_prob,axis=0)
        return output