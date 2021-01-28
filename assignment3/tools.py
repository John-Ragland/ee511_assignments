from collections import Counter

class Data:
    def __init__(self, test_name, train_name):
        self.test_file = open(test_name, "r")
        self.train_file = open(train_name, "r")
        self._calculate_vocab()
        self._load_categories()
        self.test_data = self._load_data(self.test_file)
        self.train_data = self._load_data(self.train_file)

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
        data = [[] for _ in range(len(self.categories))]
        file.seek(0)
        for post in file:
            words = post.split()
            category = words[0]
            vector = self.get_post_vector(words[1:])
            data[self.categories.index(category)].append(vector)
            
        return data

    def get_post_vector(self, post):
        vector = [0 for _ in range(len(self.vocab))]
        for word in post:
            if word in self.vocab:
                vector[self.vocab.index(word)] += 1
        return vector