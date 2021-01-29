from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class Data:
    '''
    Methods
    -------
    _calculate_vocab(self)

    _load_data(self, file)

    get_post_vectors(self, post)

    Attributes
    ----------
    self.vocab_size : int
        number of words in the vocabulary
    self.cat_size : int
        number of categories

    self.vocab : list
        list containing 5000 most frequency words
    self.catergories : numpy array
        shape (self.cat_size,)

    self.test_bin : numpy array
        shape (test_lines, self.vocab_size)
    self.test_max : numpy array
        shape (test_lines, self.vocab_size)
    self.test_log : numpy array
        shape (test_lines, self.vocab_size) 
    self.test_labels  : numpy array
        shape (test_lines,)

    self.train_bin : numpy array
        shape (train_lines, self.vocab_size)
    self.train_max : numpy array
        shape (train_lines, self.vocab_size)
    self.train_log : numpy array
        shape (train_lines, self.vocab_size)
    self.train_labels : numpy array
        shape (train_lines,)
    '''
    def __init__(self, categories, test_name='data/20ng-test-all-terms.txt', test_lines=7528, 
                       train_name='data/20ng-train-all-terms.txt', train_lines=11293, 
                       vocab_size=5000, cat_size=20):
        self.vocab_size = vocab_size
        self.cat_size = cat_size
        self.categories = categories

        print('Calculating Vocabulary...')
        self._calculate_vocab(open(train_name, "r"))
        
        print('Loading Testing Data...')
        self.test_bin, self.test_max, self.test_log, self.test_labels = self._load_data(open(test_name, "r"), test_lines)

        print('Loading Training Data...')
        self.train_bin, self.train_max, self.train_log, self.train_labels = self._load_data(open(train_name, "r"), train_lines)

        print('Normalizing training Data...')
        self._normalize_data()

        print('Splitting Training Data into Training and Validation...')
        self._train_test_split()

    def _calculate_vocab(self, file):
        # TODO: try the other vocabulary option from part b
        vocab = []
        file.seek(0)
        for line in file:
            words = line.split()[1:]
            vocab.extend(words)

        cnt = Counter(vocab)
        self.vocab = [word[0] for word in cnt.most_common(self.vocab_size)]

    def _load_data(self, file, num_of_lines):
        data_max = np.zeros(shape=(num_of_lines, self.vocab_size))
        data_log = np.zeros(shape=(num_of_lines, self.vocab_size))
        data_bin = np.zeros(shape=(num_of_lines, self.vocab_size))
        labels = np.empty(num_of_lines)
        file.seek(0)
        line = 0
        for post in file:
            words = post.split()
            labels[line] = self.categories.index(words[0])
            data_bin[line], data_max[line], data_log[line] = self.get_post_vectors(words[1:])
            line += 1

        data_bin = np.reshape(data_bin, (-1, self.vocab_size))
        data_max = np.reshape(data_max, (-1, self.vocab_size))
        data_log = np.reshape(data_log, (-1, self.vocab_size))
        return data_bin, data_max, data_log, labels

    def get_post_vectors (self, post):
        max = np.zeros(self.vocab_size)
        binary = np.zeros(self.vocab_size)
        for word in post:
            if word in self.vocab:
                index = self.vocab.index(word)
                max[index] += 1
                binary[index] = 1
        
        log = np.log(max + 1)
        return binary, max, log

    def _normalize_data(self):
        scaler_train_bin = preprocessing.StandardScaler().fit(self.train_bin)
        scaler_train_max = preprocessing.StandardScaler().fit(self.train_max)
        scaler_train_log = preprocessing.StandardScaler().fit(self.train_log)
        scaler_test_bin = preprocessing.StandardScaler().fit(self.test_bin)
        scaler_test_max = preprocessing.StandardScaler().fit(self.test_max)
        scaler_test_log = preprocessing.StandardScaler().fit(self.test_log)

        self.train_bin = scaler_train_bin.transform(self.train_bin)
        self.train_max = scaler_train_max.transform(self.train_max)
        self.train_log = scaler_train_log.transform(self.train_log)
        self.test_bin = scaler_test_bin.transform(self.test_bin)
        self.test_max = scaler_test_max.transform(self.test_max)
        self.test_log = scaler_test_log.transform(self.test_log)

    def _train_test_split(self, valid_size=0.1):
        self.train_bin, self.valid_bin, self.train_bin_labels, self.valid_bin_labels = \
            train_test_split(self.train_bin, self.train_labels, test_size=valid_size)
        self.train_max, self.valid_max, self.train_max_labels, self.valid_max_labels = \
            train_test_split(self.train_max, self.train_labels, test_size=valid_size)
        self.train_log, self.valid_log, self.train_log_labels, self.valid_log_labels = \
            train_test_split(self.train_log, self.train_labels, test_size=valid_size)