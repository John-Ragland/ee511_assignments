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

def data_encoding(data, vocab, languages, tweet_length=282):
    tweets = tweet_enconding(data.tweet, vocab, tweet_length)
    langs = lang_encoding(data.lan, languages, tweet_length)
    return tweets, langs

def tweet_enconding(tweets, vocab, tweet_length):
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

def lang_encoding(labels, languages, tweet_length):
    encoded = np.zeros((len(labels), tweet_length))
    for l, lang in enumerate(labels):
        idx = languages.index(lang)
        for char in range(1, tweet_length-1):
            encoded[l][char] = idx
    return encoded

def check_files_exists(filename):
    # Check if data files exist
    if not os.path.exists(filename):
        raise Exception ('Data files missings, please add %s.' % filename)

def train(model, device, train_loader, optimizer, epochs, log_interval, verbose=False):
    for epoch in range(epochs + 1):
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output, hidden = model(data, label)
            # Doruk's little experiment
            # if batch_idx == 0:
            #     log = 0.0
            #     for k, vocab in enumerate(output[0]):
            #         index = data[0][k]
            #         log += vocab[index].item()
            #     print(log)
            loss = model.loss(output, label)
            loss.backward()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_ppl = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, label)
            test_loss += model.loss(output, label).item()
            test_ppl += math.exp(F.cross_entropy(output.view(-1, 509), label.view(-1), ignore_index=507))

    test_loss /= len(test_loader.dataset)
    test_ppl /= len(test_loader.dataset)
    print('test_ppl : ' + str(test_ppl))
    print('test_loss : ' + str(test_loss))
    
    return test_loss, test_ppl

def test2(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_ppl = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            print(data.size())
            print(label.size())
            output, hidden = model(data, label)

            return output, hidden, data
            test_loss += model.loss(output, label).item()
            test_ppl += math.exp(F.cross_entropy(output.view(-1, 509), label.view(-1), ignore_index=507))

    test_loss /= len(test_loader.dataset)
    test_ppl /= len(test_loader.dataset)
    print('test_ppl : ' + str(test_ppl))
    print('test_loss : ' + str(test_loss))
    
    return test_loss, test_ppl