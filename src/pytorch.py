#!/usr/bin/env python3
"""
pytorch.py
"""
import time
import mnist_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.conv1 = ConvPoolLayer(input_size, filter_size=(3, 6, 5),
                activation_fn=nn.ReLU())
        self.conv2 = ConvPoolLayer(input_size=(6, 48, 48), 
                filter_size=(6, 16, 5), activation_fn=nn.ReLU())
        self.fc1 = FullyConnectedLayer(16*22*22, 120)
        self.fc2 = FullyConnectedLayer(120, 84)
        self.fc3 = FullyConnectedLayer(84, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16*22*22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FullyConnectedLayer(nn.Module):
    def __init__(self, n_in, n_out, activation_fn=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = x.view(-1, self.n_in)
        out = self.fc(x)
        if self.activation_fn is not None:
            out = self.activation(out)
        return out

class ConvPoolLayer(nn.Module):
    def __init__(self, input_size, filter_size, activation_fn=None, 
            poolsize=(2,2)):
        super().__init__()
        self.input_size = input_size
        self.filter_size = filter_size
        self.activation_fn = activation_fn
        self.poolsize = poolsize
        self.conv = nn.Conv2d(*filter_size)
        self.pool = nn.MaxPool2d(poolsize)

    def forward(self, x):
        x = x.view(-1, *self.input_size)
        out = self.conv(x)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        out = self.pool(out)
        return out

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

def main():
    # check for gpu
    if torch.cuda.is_available():
        GPU = True
    else:
        GPU = False
    # hyper-parameters
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 1000

    # read data
    X = np.load('../data/X.npy')
    y = np.load('../data/y.npy')
    y_onehot = np.zeros((len(y), 2))
    y_onehot[np.arange(len(y)), y] = 1
    y = y_onehot
    # shuffle data
    X, y = shuffle_numpy(X, y)
    # remove 1-hot
    y = np.argmax(y, axis=1)
    # split data
    training_data, validation_data, _ = split_data(X, y)
    X_train, y_train = training_data
    X_valid, y_valid = validation_data
    # number of examples
    n_train = len(X_train)
    n_valid = len(X_valid)

    # start timer
    start = time.time()

    # define model
    model = Network(input_size=(3,100,100))
    if GPU:
        model = model.cuda()

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    validation_accuracies = []
    # train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        # generator to loop through mini batches
        mini_batch = ((X_train[k:k+batch_size], y_train[k:k+batch_size])
                for k in range(0, n_train, batch_size))
        # for each mini batch
        for x_batch, y_batch in mini_batch:
            if GPU:
                inputs = Variable(torch.from_numpy(x_batch).cuda())
                targets = Variable(torch.from_numpy(y_batch).cuda())
            else:
                inputs = Variable(torch.from_numpy(x_batch))
                targets = Variable(torch.from_numpy(y_batch))

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            
        # training loss
        print('Epoch {}, Training Loss {}'.format(epoch, running_loss))
        training_losses.append(running_loss)

        # test the model
        mini_batch = ((X_valid[k:k+batch_size], y_valid[k:k+batch_size])
                for k in range(0, n_valid, batch_size))
        # for each mini batch
        total = 0
        correct = 0
        for x_batch, y_batch in mini_batch:
            if GPU:
                inputs = Variable(torch.from_numpy(x_batch).cuda())
                labels = torch.from_numpy(y_batch).cuda()
            else:
                inputs = Variable(torch.from_numpy(x_batch))
                labels = torch.from_numpy(y_batch)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum((labels==predicted).int())
        accuracy = 100 * correct / total
        print('\t Validation Accuracy {}'.format(accuracy))
        validation_accuracies.append(accuracy)

    # print elapsed time
    end = time.time()
    print('Elapsed time:', end-start, 'seconds')

    # plot the graph
    plt.figure()
    plt.subplot(121), plt.plot(training_losses)
    plt.title('Training Loss'), plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.subplot(122), plt.plot(validation_accuracies)
    plt.title('Validation Accuracy'), plt.xlabel('Epoch'),
    plt.ylabel('Accuracy')
    plt.ylim(0,100)
    plt.show()

if __name__ == '__main__':
    main()
