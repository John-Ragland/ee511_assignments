from collections import Counter
import numpy as np

class Data:
    '''
    Methods
    -------

    _calculate_vocab(self)

    _load_categories(self)

    _load_data(self, file)

    get_post_vector_count(self, post)
    
    get_post_vector_log(self, post)

    get_post_vector_bin(self, post)

    Attributes
    ----------
    test_file : _io.TextIOWrapper
        test wrapper for testing data
    train_file : _io.TextIOWrapper
        train wrapper for training data
    test_data : list
        test_data[a][b][c] => a: category, b: vector of a post in that
        category, c: frequency of the word self.vocab[c] in this post    
    train_data_count : list
        train_data_count[a][b][c] => a: category, b: vector of a post in that
        category, c: frequency of the word self.vocab[c] in this post.
        Vocabulary is represented using counted method
    train_data_log : list
        train_data_log[a][b][c] => a: category, b: vector of a post in that
        category, c: frequency of the word self.vocab[c] in this post.
        Vocabulary is represented by log normalized count method
    train_data_bin : list
        train_data_bin[a][b][c] => a: category, b: vector of a post in that
        category, c: frequency of the word self.vocab[c] in this post.
        Vocabulary is represented by binary activation method
    categories : list
        list of the 20 category names. Index corresponds to a index of train
        and test data.
    vocab : list
        list containing 5000 most frequency words

    '''
    def __init__(self, test_name, train_name):
        print('Opening Data Files...')
        self.test_file = open(test_name, "r")
        self.train_file = open(train_name, "r")

        print('Calculating Vocabulary...')
        self._calculate_vocab()

        print('Loading Categories...')
        self._load_categories()
        
        # TODO change test data vocab representation to correct form
        print('Loading Testing Data...')
        _, _, self.test_data = self._load_data(self.test_file)

        print('Loading Training Data...')
        self.train_data_bin, self.train_data_count, self.train_data_log = self._load_data(self.train_file)

    def _calculate_vocab(self):
        vocab = []
        self.train_file.seek(0)
        for line in self.train_file:
            words = line.split()[1:]
            vocab.extend(words)

        cnt = Counter(vocab)
        self.vocab = [word[0] for word in cnt.most_common(5000)]

    def _load_categories(self):
        categories = []
        self.test_file.seek(0)
        for line in self.test_file:
            category = line.split()[0]
            if category not in categories:
                categories.append(category)
            
        self.categories = categories

    def _load_data(self, file):
        # create data variables
        data_count = [[] for _ in range(len(self.categories))]
        data_log = [[] for _ in range(len(self.categories))]
        data_bin = [[] for _ in range(len(self.categories))]
        file.seek(0)
        
        for post in file:
            words = post.split()
            category = words[0]
            
            # Calculate vectors for different vocab representations
            vector_count = self.get_post_vector_count(words[1:])
            vector_log = self.get_post_vector_log(words[1:])
            vector_bin = self.get_post_vector_bin(words[1:])

            # Create Data Variabels
            data_count[self.categories.index(category)].append(vector_count)
            data_log[self.categories.index(category)].append(vector_log)
            data_bin[self.categories.index(category)].append(vector_bin)
        return data_bin, data_count, data_log

    def get_post_vector_count(self, post):
        vector = [0 for _ in range(len(self.vocab))]
        for word in post:
            if word in self.vocab:
                vector[self.vocab.index(word)] += 1
        
        # implement max(1,td)
        for k in range(len(vector)):
            if vector[k] == 0:
                vector[k] = 1
        
        return vector

    def get_post_vector_log(self, post):
        td = [0 for _ in range(len(self.vocab))]
        vector = [0 for _ in range(len(self.vocab))]
        for word in post:
            if word in self.vocab:
                td[self.vocab.index(word)] += 1
        
        # implement max(1,td)
        for k in range(len(td)):
            vector[k] = np.log(1+td[k])
        
        return vector

    def get_post_vector_bin(self, post):
        vector = [0 for _ in range(len(self.vocab))]
        for word in post:
            if word in self.vocab:
                vector[self.vocab.index(word)] = 1
        
        return vector