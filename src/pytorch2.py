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
GPU = False

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
        out = self.activation(out)
        out = nn.functional.max_pool2d(out, self.poolsize)
        return out

class FullyConnectedLayer():
    def __init__(self, n_in, n_out, activation=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
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
        out = torch.mm(x, self.W) + self.b
        if self.activation is not None:
            out = self.activation(out)
        return out

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
    num_epochs = 30
    learning_rate = 0.001
    batch_size = 100

    # read data
    X = np.load('../data/X.npy')
    y = np.load('../data/y.npy')
    # shuffle data
    X, y = shuffle_numpy(X, y)
    # split data
    training_data, validation_data, _ = split_data(X, y)
    X_train, y_train = training_data
    X_valid, y_valid = validation_data
    # number of examples
    n_train = len(X_train)
    n_valid = len(X_valid)

    # create Variables
    X_tr, y_tr = load_data(training_data, num_classes)
    X_val, y_val = load_data(validation_data, num_classes)

    # create network
    conv1 = ConvPoolLayer(filter_shape=(6, 3, 5, 5), 
            image_shape=(batch_size, 3, 100, 100), activation=nn.ReLU())
    conv2 = ConvPoolLayer(filter_shape=(16, 6, 5, 5),
            image_shape=(batch_size, 6, 48, 48), activation=nn.ReLU())
    fc1 = FullyConnectedLayer(16*22*22, 120, activation=nn.ReLU())
    fc2 = FullyConnectedLayer(120, 84, activation=nn.ReLU())
    fc3 = FullyConnectedLayer(84, num_classes, activation=sigmoid)
    net = Network([conv1, conv2, fc1, fc2, fc3])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.params, lr=learning_rate)

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
                for k in range(0, n_train, batch_size))
        # for each mini batch
        for x_batch, y_batch in mini_batch:
            # forward pass
            outs = net.forward(x_batch)
            # compute loss
#            optimizer.zero_grad()
#            loss = criterion(outs, y_batch)
            loss = CrossEntropyCost.fn(outs, y_batch)
            loss /= len(x_batch)
    
            # backpropagation
            loss.backward()
#            optimizer.step()
        
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
            _, y_act = torch.max(y_batch, 1)
            correct += torch.sum((y_pred == y_act.data).int())
            
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
