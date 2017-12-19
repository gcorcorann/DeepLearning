#!/usr/bin/env python3
"""
Accident CNN Module
"""
# Third-party Libraries
import numpy as np
import torch
import torch.nn.functional as F

# set random seed for reproducibility
np.random.seed(1234)
# Constants
GPU = False
if GPU:
    print('Trying to run under a GPU.')
else:
    print('Running with a CPU.')

# Main class used to construct and train networks
class Network():
    def __init__(self, layers, mini_batch_size):
        """
        Takes list a of `layers` describing the network architecture and a value
        for the `mini_batch_size` to be used during training by stochastic
        gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        



class FullyConnectedLayer():
    def __init__(self, n_in, n_out, activation_fn=F.sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # initialize weights and biases
        self.w = torch.from_numpy(np.asarray(
            np.random.normal(
                loc=0.0, scale=np.sqrt(1.0/n_in), size=(n_in, n_out)), 
            dtype=np.float32))
        self.b = torch.from_numpy(np.asarray(
            np.random.normal(
                loc=0.0, scale=1.0, size=(n_out,)),
            dtype=np.float32))
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = sigmoid(
                (1-self.p_dropout)*torch.dot(self.inpt, self.w) + self.b)
        self.y_out = torch.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
                inpt_dropout.reshape((mini_batch_size, self.n_in)),
                self.p_dropout)
        self.output_dropout = sigmoid(
                torch.dot(self.inpt_dropout, self.w) + self.b)
        
    def accuracy(self, y):
        """
        Return the accuracy for the mini-batch.
        """
        return torch.mean(torch.eq(y, self.y_out))
        

### Miscellaneous functions
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

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
    fc = FullyConnectedLayer(100, 10)
    # load data
#    X = np.load('../data/X.npy')
#    y = np.load('../data/y.npy')
#    # reshape into num_instances x num_features
#    X = np.reshape(X, (len(X), -1))
#    data = X, y
#    # shuffle data
#    data = shuffle(data)
#    # split data
#    training_data, validation_data, testing_data = split_data(data)
    

if __name__ == '__main__':
    main()
