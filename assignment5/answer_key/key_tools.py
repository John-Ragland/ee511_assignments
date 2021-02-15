import collections
import numpy as np
import pandas
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
import math

class Dataset(object):
    def __init__(self, data_file, batch_size, PAD):
        self.batch_size = batch_size
        self.PAD = PAD
        with open(data_file, 'rb') as data_pkl:
            dataset = pickle.load(data_pkl)

        # a list of sequences of char indices
        self.data = [torch.LongTensor(seq) for seq in dataset['chars']]
        self.langs = dataset['langs']
        self.numBatches = math.ceil(len(self.data)/batch_size)

            
    def _batchify(self, data, langs, align_right=False, include_lengths=False):
        lengths = [x.size(0)-1 for x in data]
        max_length = max(lengths)
        src_out = data[0].new(len(data), max_length).fill_(self.PAD)
        lang_out = data[0].new(len(data), max_length).fill_(self.PAD)
        tgt_out = data[0].new(len(data), max_length).fill_(self.PAD)
        for i in range(len(data)):
            data_length = lengths[i]
            offset = max_length - data_length if align_right else 0
            src_out[i].narrow(0, offset, data_length).copy_(data[i][:-1])
            tgt_out[i].narrow(0, offset, data_length).copy_(data[i][1:])
            lang_out[i].fill_(langs[i])
        if include_lengths:
            return src_out, tgt_out, lang_out, lengths
        else:
            return src_out, tgt_out, lang_out
        
    def __len__(self):
        return self.numBatches
        
    def __getitem__(self, index):
        start_idx = index*self.batch_size
        end_idx = (index+1)*self.batch_size

        srcBatch, tgtBatch, langBatch, lengths = self._batchify(self.data[start_idx:end_idx], self.langs[start_idx:end_idx], include_lengths=True)
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch, tgtBatch, langBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, srcBatch, tgtBatch, langBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).transpose(0, 1).contiguous()
            b = Variable(b, volatile=False)
            return b
        
        # srcSeqLength * bs, srcSeqLength * bs, tgtSeqLength * bs
        return (wrap(srcBatch), wrap(langBatch), list(lengths)), wrap(tgtBatch), indices


class MyRNN(nn.Module):
    def __init__(self, vocab_size, lang_size=9, char_vec_size=12, lang_vec_size=2, hidden_size=50, PAD=0):
        
        super(MyRNN, self).__init__()
        self.vocab_size = vocab_size
        self.lang_size = lang_size
        self.char_vec_size = char_vec_size
        self.lang_vec_size = lang_vec_size
        self.hidden_size = hidden_size
        
        self.char_encoder = nn.Embedding(self.vocab_size, self.char_vec_size)
        self.lang_encoder = nn.Embedding(self.lang_size, self.lang_vec_size)
        # the current hidden size = char_vec_size
        self.gru = nn.GRU(self.char_vec_size+self.lang_vec_size, self.hidden_size, num_layers=1)
        self.linear = nn.Linear(self.hidden_size, self.char_vec_size)
        self.decoder = nn.Linear(self.char_vec_size, self.vocab_size)
        
        # This shares the encoder and decoder weights as described in lecture.
        self.decoder.weight = self.char_encoder.weight
        self.decoder.bias.data.zero_()
        
        
        weight = torch.ones(vocab_size)
        # scores over PAD is not counted
        weight[PAD] = 0
        self.sm = nn.LogSoftmax(dim=1)
        self.crit = nn.NLLLoss(weight, size_average=False)

    def forward(self, input, hidden=None):
        emb = pack(torch.cat((self.char_encoder(input[0]), self.lang_encoder(input[1])), -1), input[2])
        output, hidden_t = self.gru(emb, hidden)
        output = unpack(output)[0]
        output = F.tanh(self.linear(output))
        output = self.decoder(output)
        return output, hidden_t

    # Predefined loss function
    def loss(self, prediction, label, reduction='elementwise_mean'):
        prediction = prediction.view(-1, self.vocab_size)
        prediction = self.sm(prediction)
        loss_val = self.crit(prediction, label.view(-1))
        return loss_val

# train process for a epoch
def train(model, trainData, epoch, optimizer, PAD):
    model.train()
    train_ppl = 0
    train_loss = 0
    for i in range(len(trainData)):
        data, label, _ = trainData[i]
        
        optimizer.zero_grad()
        output, hidden = model(data)
        train_ppl += math.exp(F.cross_entropy(output.view(-1, model.vocab_size), label.view(-1), ignore_index=PAD))
        loss = model.loss(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_ppl /= len(trainData)
    train_loss /= len(trainData)
    print('epoch : ' + str(epoch))
    print('     train_ppl : ' + str(train_ppl))
    print('     train_loss : ' + str(train_loss))
    return train_ppl
  
  

def test(model, testData, PAD):
    model.eval()
    test_loss = 0
    test_ppl = 0
    with torch.no_grad():
        for i in range(len(testData)):
            data, label, _ = testData[i]
            output, hidden = model(data)
            test_loss += model.loss(output, label).item()
            test_ppl += math.exp(F.cross_entropy(output.view(-1, model.vocab_size), label.view(-1), ignore_index=PAD))

    test_loss /= len(testData)
    test_ppl /= len(testData)
    print('test_ppl : ' + str(test_ppl))
    print('test_loss : ' + str(test_loss))

    return test_loss, test_ppl

def get_predictions(model, testData):
    model.eval()
    predictions = []
    batch_size = testData.batch_size
    with torch.no_grad():
        for i in range(len(testData)):
            data, label, indices = testData[i]
            new_src = torch.cat([data[0][:,i].unsqueeze(-1).repeat(1,9) for i in range(data[0].size(1))], -1)
            new_lang = torch.cat([data[1][:,i].unsqueeze(-1).repeat(1,9) for i in range(data[1].size(1))], -1)
            for j in range(9):
                new_lang[:, [col for col in range(new_lang.size(1)) if col%9==j]].fill_(j)
            new_lengths = []
            for l in data[2]:
                new_lengths += [l]*9
            data = (new_src, new_lang, new_lengths)
            output, hidden = model(data)
            outputs = output.split(9, dim=1)
            batch_predictions = []
            for e_idx in range(len(outputs)):
                e = outputs[e_idx]
                smallest_loss = 10000000
                pred = None
                for j in range(9):
                    loss = model.loss(e[:,j], label[:,e_idx]).item()
                    if loss < smallest_loss:
                        smallest_loss = loss
                        pred = j
                batch_predictions.append(pred)
            batch_predictions, indices = zip(*sorted(zip(batch_predictions, indices), key=lambda x: x[1]))
            predictions += batch_predictions
    return predictions