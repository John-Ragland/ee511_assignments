import numpy as np
import pandas as pd
import os
from collections import Counter
import torch

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

        vocab = most_common[:k]
        vocab = [i[0] for i in vocab]

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

def list_to_tensor(data, verbose=False):
    tensor = [0]*len(data)
    for k, item in enumerate(data):
        tensor[k] = string_to_tensor(item)
        if verbose:
            print('%d %% complete' % k/len(data)*100, end='\r')
    return tensor

def string_to_tensor(line, vocab):
    # Based on https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
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

def check_files_exists(filename):
    # Check if data files exist
    if not os.path.exists(filename):
        raise Exception ('Data files missings, please add %s.' % filename)

def train(model, device, train_loader, optimizer, epochs, log_interval, criterion, verbose=False):
    for epoch in range(epochs + 1):
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output, hidden = model(data)
            loss = criterion(output, label) 
            loss.backward()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output, hidden = model(data)
            test_loss += criterion(output, label) 
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)
