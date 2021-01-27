import pandas as pd

def load_data():
    training_data = 'data/20ng-test-all-terms.txt'
    
    f = open(training_data, "r")
    train = []
    for x in f:
        train.append(x.split())
    
    return train