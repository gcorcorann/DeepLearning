#!/usr/bin/env python3
"""
Accident CNN Module
"""
# Third-party Libraries
import numpy as np

# set random seed for reproducibility
np.random.seed(1234)
# Constants
GPU = False
if GPU:
    print('Trying to run under a GPU.')
else:
    print('Running with a CPU.')


# Main class used to construct and train networks
class Network()

def shuffle(data):
    X, y = data
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    return X, y

def split_data(data):
    X, y = data
    X_train, X_val, X_test = np.split(X, [int(0.6*len(X)), int(0.8*len(X))])
    y_train, y_val, y_test = np.split(y, [int(0.6*len(y)), int(0.8*len(y))])
    print('X_train:', X_train.shape, 'y_train:', y_train.shape)
    print('X_val:', X_val.shape, '\ty_val:', y_val.shape)
    print('X_test:', X_test.shape, '\ty_test:', y_test.shape)
    print()
    training_data = X_train, y_train
    validation_data = X_val, y_val
    testing_data = X_test, y_test
    return training_data, validation_data, testing_data

def main():
    """ Main Function. """
    # load data
    X = np.load('../data/X.npy')
    y = np.load('../data/y.npy')
    # reshape into num_instances x num_features
    X = np.reshape(X, (len(X), -1))
    data = X, y
    # shuffle data
    data = shuffle(data)
    # split data
    training_data, validation_data, testing_data = split_data(data)

if __name__ == '__main__':
    main()
