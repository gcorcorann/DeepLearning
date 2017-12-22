#!/usr/bin/env python3
"""
pytorch.py
"""
import time
import mnist_loader
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

def main():
    # hyper-parameters
    input_size = 784
    hidden_size = 100
    output_size = 10
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 1000

    # read data
    training_data, validation_data, _ = mnist_loader.load_data()
    X_train, y_train = training_data
    X_valid, y_valid = validation_data
    print(X_train.shape)
    print(y_train.shape)
    n = len(X_train)

    # start timer
    start = time.time()

    # define model
    model = FullyConnectedLayer(input_size, hidden_size, output_size)
    if torch.cuda.is_available():
        model.cuda()

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
                for k in range(0, n, batch_size))
        # for each mini batch
        for x_batch, y_batch in mini_batch:
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(x_batch)).cuda()
                targets = Variable(torch.from_numpy(y_batch)).cuda()
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
            
        print('Epoch {}, Training Loss {}'.format(epoch, running_loss))
        training_losses.append(running_loss)
        running_loss = 0.0

        # test the model
        inputs = Variable(torch.from_numpy(X_valid)).cuda()
        labels = torch.from_numpy(y_valid).cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = torch.sum((labels==predicted).int())
        accuracy = 100 * correct / total
        print('Accuracy {}'.format(accuracy))
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
