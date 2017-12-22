#!/usr/bin/env python3
"""
pytorch2.py
"""
import time
import mnist_loader
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# constants
GPU = True

# Define Complete Model
class Network():
    def __init__(self, layers):
        """
        Takes list of layers as input.
        """
        self.layers = layers
        self.params = [param for layer in self.layers for param in layer.params]

    def forward(self, x):
        outputs = x
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

# Define Layers
class ConvPoolLayer():
    def __init__(self, filter_shape, image_shape, activation, poolsize=(2,2)):
        """
        `filter_shape` is a tuple of length 4 whose entries are the number of
        filters, the number of input feature maps, the filter height, and the
        filter width,
        `image_shape` is a tuple of length 4 whoes entries are the mini_batch
        size, the number of input feature maps, the image height, and the image
        width.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation = activation
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        # initialize weights and biases
        if GPU:
            self.W = Variable((torch.randn(filter_shape)/np.sqrt(n_out)).cuda(), 
                    requires_grad=True)
            self.b = Variable(torch.randn(filter_shape[0]).cuda(), 
                    requires_grad=True)
        else:
            self.W = Variable(torch.randn(filter_shape)/np.sqrt(n_out), 
                    requires_grad=True)
            self.b = Variable(torch.randn(filter_shape[0]), requires_grad=True)
        self.params = [self.W, self.b]

    def forward(self, x):
        x = x.view(-1, *self.image_shape[1:])
        out = nn.functional.conv2d(x, self.W, self.b)
        out = nn.functional.max_pool2d(out, self.poolsize)
        return out

class FullyConnectedLayer():
    def __init__(self, n_in, n_out, activation, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.p_dropout = p_dropout
        # initialize weights and biases
        if GPU:
            self.W = Variable((torch.randn(n_in, n_out)/np.sqrt(n_in)).cuda(), 
                    requires_grad=True)
            self.b = Variable(torch.randn(n_out).cuda(), requires_grad=True)
        else:
            self.W = Variable(torch.randn(n_in, n_out)/np.sqrt(n_in), 
                    requires_grad=True)
            self.b = Variable(torch.randn(n_out), requires_grad=True)
        self.params = [self.W, self.b]

    def forward(self, x): 
        x = x.view(-1, self.n_in)
        z = torch.mm(x, self.W) + self.b
        a = self.activation(z)
        return a

# Define Cost Function
class CrossEntropyCost():
    """
    Cross Entropy Cost Function.
    """
    @staticmethod
    def fn(a, y):
        loss = y*torch.log(a) + (1-y)*torch.log(1-a)
        loss = -torch.sum(loss)
        return loss

# Miscellaneous Functions
def sigmoid(z):
    z = torch.clamp(z, -4, 4)
    a = 1.0 / (1.0 + torch.exp(-z))
    return a

def shuffle_numpy(X, y):
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    return X, y

def shuffle_tensor(X, y):
    idx = torch.randperm(len(X))
    if GPU:
        idx = idx.cuda()
    X = X[idx]
    y = y[idx]
    return X, y

def split_data(X, y):
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

def load_data(data, num_classes):
    """
    Load data into torch variables.
    """
    X, y = data
    y = np.asarray(y, dtype=np.float32)
    # number of instances
    n = len(X)
    # create tensor for data
    if GPU:
        X = Variable(torch.from_numpy(X).cuda())
        y = Variable(torch.from_numpy(y).cuda())
    else:
        X = Variable(torch.from_numpy(X))
        y = Variable(torch.from_numpy(y))
    return X, y

def main():
    """
    Main Function.
    """
    # hyperparameters
    num_classes = 2
    hidden_size = 100
    output_size = 10
    num_epochs = 30
    learning_rate = 0.1
    batch_size = 10

    # read data
    X = np.load('../data/X.npy')
    y = np.load('../data/y.npy')
    # shuffle data
    X, y = shuffle_numpy(X, y)
    # split data
    training_data, validation_data, testing_data = split_data(X, y)
#    training_data, validation_data, _ = mnist_loader.load_data()

    #TODO remove small sample size
    new_training_data = (training_data[0][:1000], training_data[1][:1000])
    new_validation_data = (validation_data[0][:1000], validation_data[1][:1000])
    # number of training examples
    n = len(new_training_data[0])
    # create Variables
    X_tr, y_tr = load_data(new_training_data, num_classes)
    X_val, y_val = load_data(new_validation_data, num_classes)

    # create network
    conv1 = ConvPoolLayer(filter_shape=(20, 3, 5, 5), 
            image_shape=(batch_size, 3, 100, 100), activation=sigmoid)
    conv2 = ConvPoolLayer(filter_shape=(40, 20, 5, 5),
            image_shape=(batch_size, 20, 48, 48), activation=sigmoid)
    fc1 = FullyConnectedLayer(40*22*22, 200, sigmoid)
    fc2 = FullyConnectedLayer(200, num_classes, sigmoid)
    net = Network([conv1, conv2, fc1, fc2])

    # time network
    t = time.time()
    # train network
    training_losses = []
    validation_accuracies = []
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        running_loss = 0.0
        # shuffle data
        X_tr, y_tr = shuffle_tensor(X_tr, y_tr)
        # generator to loop through mini batches
        mini_batch = ((X_tr[k:k+batch_size], y_tr[k:k+batch_size])
                for k in range(0, n, batch_size))
        # for each mini batch
        for x_batch, y_batch in mini_batch:
            # forward pass
            outs = net.forward(x_batch)
    
            # compute loss
            loss = CrossEntropyCost.fn(outs, y_batch)
            loss /= len(x_batch)
    
            # backpropagation
            loss.backward()
        
            # update the weights
            for param in net.params:
                param.data.sub_(param.grad.data * learning_rate)
    
            # set gradients equal to zero
            for param in net.params:
                param.grad.data.zero_()

            running_loss += loss.data[0]

        # training loss
        print('\tTraining Loss:', running_loss)
        training_losses.append(running_loss)

        # test accuracy
        n_val = len(X_val)
        correct = 0
        # generator to loop through mini batches
        mini_batch = ((X_tr[k:k+batch_size], y_tr[k:k+batch_size])
                for k in range(0, n_val, batch_size))
        # for each mini batch
        for x_batch, y_batch in mini_batch:
            outs = net.forward(x_batch)
            _, y_pred = torch.max(outs.data, 1)
            _, y = torch.max(y_batch.data, 1)
            correct += torch.sum((y_pred == y).int())
            
        accuracy = 100 * correct / n_val
        print('\tValidation Accuracy:', accuracy)
        validation_accuracies.append(accuracy)
        correct = 0

    # print elapsed time
    print('Elapsed Time:', time.time() - t)

    # display plots
    plt.figure()
    plt.subplot(121), plt.plot(training_losses)
    plt.title('Training Losses'), plt.xlabel('Epochs'), plt.ylabel('Loss')
    plt.subplot(122), plt.plot(validation_accuracies)
    plt.title('Validation Accuracy'), plt.xlabel('Epochs'),
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
