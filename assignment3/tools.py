from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
import math

class Data:
    '''
    Methods
    -------
    _calculate_vocab(self)

    _load_data(self, file)

    _train_test_split(self, valid_size=0.1)

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
    self.train_bin_labels : numpy array
        shape (train_lines,)
    self.train_max : numpy array
        shape (train_lines, self.vocab_size)
    self.train_max_labels : numpy array
        shape (train_lines,)
    self.train_log : numpy array
        shape (train_lines, self.vocab_size)
    self.train_log_labels : numpy array
        shape (train_lines,)
    self.train_labels : numpy array
        shape (train_lines,)

    self.valid_bin : numpy array
        shape (valid_lines, self.vocab_size)
    self.valid_bin_labels : numpy array
        shape (valid_lines,)
    self.valid_max : numpy array
        shape (valid_lines, self.vocab_size)
    self.valid_max_labels : numpy array
        shape (valid_lines,)
    self.valid_log : numpy array
        shape (valid_lines, self.vocab_size)
    self.valid_log_labels : numpy array
        shape (valid_lines,)
    '''
    def __init__(self, categories, test_name='data/20ng-test-all-terms.txt', test_lines=7528, 
                       train_name='data/20ng-train-all-terms.txt', train_lines=11293, 
                       vocab_size=5000, cat_size=20):
        self.vocab_size = vocab_size
        self.cat_size = cat_size
        self.categories = categories

        print('Calculating Vocabulary...')
        self._calculate_vocab(open(train_name, "r"), train_lines)
        
        print('Loading Testing Data...')
        self.test_bin, self.test_max, self.test_log, self.test_labels = self._load_data(open(test_name, "r"), test_lines)

        print('Loading Training Data...')
        self.train_bin, self.train_max, self.train_log, self.train_labels = self._load_data(open(train_name, "r"), train_lines)

        print('Splitting Training Data into Training and Validation...')
        self._train_test_split()

    def _calculate_vocab(self, file, num_of_lines):
        self.makeShiftMI(file, num_of_lines) # about 15 minutes
        # self.sklearnMI(file, num_of_lines) # did not finish executing after 20+ minutes
        # self.mostFrequentVocab(file, num_of_lines) # a few seconds

    def makeShiftMI(self, file, num_of_lines):
        file.seek(0)
        vocab = [[] for _ in range(num_of_lines)]
        labels = ['' for _ in range(num_of_lines)]
        for index, line in enumerate(file):
            words = line.split("\t", 1)
            labels[index] = self.categories.index(words[0])
            vocab[index] = words[1]
        
        # get all of the unique words in the training data
        vect = CountVectorizer(binary = True)
        cv_fit = vect.fit_transform(vocab).toarray()
        num_of_words = len(vect.get_feature_names())

        print(num_of_words)

        # get the number of files for each label
        label_cnt = Counter(labels)
        label_counts = [0 for _ in range(self.cat_size)]
        for label, count in label_cnt.most_common(self.cat_size):
            label_counts[label] = count
        
        # get the number of files each word is in
        word_counts = cv_fit.sum(axis=0)

        mi = [0 for _ in range(num_of_words)]
        for word in range(num_of_words):
            print(word, end="\r")
            # get the number of files the word is in for each category
            category_counts = [0 for _ in range(len(self.categories))]
            for index, file_count in enumerate(cv_fit[:, word]):
                category_counts[labels[index]] += file_count
            info = 0
            for y in range(self.cat_size):
                # number of files with category y
                p_y = (label_counts[y] + 1) / (num_of_lines + 20)
                # number of files that have the word
                p_x = (word_counts[word] + 20) / (num_of_lines + 20)
                # number of filed the dont have the word
                p_not_x = (num_of_lines - word_counts[word]) / (num_of_lines + 20)
                # number of category y files that have word
                p_x_y = (category_counts[y] + 1) / (num_of_lines + 20)
                # number of category y files that dont have the word
                p_not_x_y = (label_counts[y] - category_counts[y] + 1) / (num_of_lines + 20)

                info += p_x_y * math.log(p_x_y / (p_x * p_y))
                info += p_not_x_y * math.log(p_not_x_y / (p_not_x * p_y))
            mi[word] = info

        top_5000 = np.argsort(mi)[-5000:]
        self.vocab = [vect.get_feature_names()[i] for i in top_5000]
        self.vocab_vals = [mi[i] for i in top_5000]
        
    def sklearnMI(self, file, num_of_lines):
        file.seek(0)
        vocab = [[] for _ in range(num_of_lines)]
        labels = ['' for _ in range(num_of_lines)]
        for index, line in enumerate(file):
            words = line.split("\t", 1)
            labels[index] = self.categories.index(words[0])
            vocab[index] = words[1]

        vect = CountVectorizer(binary = True)
        cv_fit = vect.fit_transform(vocab).toarray()

        mi = mutual_info_classif(cv_fit, labels)

        print(mi)

        top_5000 = np.argsort(mi)[-5000:]
        self.vocab = [vect.get_feature_names()[i] for i in top_5000]
        self.vocab_vals = [mi[i] for i in top_5000]

    def mostFrequentVocab(self, file, num_of_lines):
        all_words = chain(*(line.split()[1:] for line in file if line))
        cnt = Counter(all_words)
        self.vocab = [word[0] for word in cnt.most_common(self.vocab_size)]

    def _load_data(self, file, num_of_lines):
        file.seek(0)
        data = [[] for _ in range(num_of_lines)]
        labels = ['' for _ in range(num_of_lines)]
        for index, line in enumerate(file):
            words = line.split("\t", 1)
            labels[index] = self.categories.index(words[0])
            data[index] = words[1]

        cv_max = CountVectorizer(vocabulary=self.vocab)
        data_max = cv_max.fit_transform(data).toarray()
        data_bin = np.where(data_max > 0, 1, 0)
        data_log = np.log(data_max + 1)
        return data_bin, data_max, data_log, labels

    def _train_test_split(self, valid_size=0.1):
        self.train_bin, self.valid_bin, self.train_bin_labels, self.valid_bin_labels = \
            train_test_split(self.train_bin, self.train_labels, test_size=valid_size)
        self.train_max, self.valid_max, self.train_max_labels, self.valid_max_labels = \
            train_test_split(self.train_max, self.train_labels, test_size=valid_size)
        self.train_log, self.valid_log, self.train_log_labels, self.valid_log_labels = \
            train_test_split(self.train_log, self.train_labels, test_size=valid_size)