#!/usr/bin/env python3
"""
accident_cnn.py
"""
import mnist_loader
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

    def SGD(self, training_data, epochs, batch_size, lr, validation_data):
        if validation_data:
            n_val = len(validation_data[0])
        n_train = len(training_data[0])
        validation_cost, validation_accuracy = [], []
        training_cost, training_accuracy = [], []
        # set optimizer
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            training_data = shuffle(training_data)
            mini_batches = [(training_data[0][k:k+batch_size],
                training_data[1][k:k+batch_size])
                for k in range(0, n_train, batch_size)]

            for mini_batch in mini_batches:
                x, y = mini_batch
                # wrap in Tensor
                x = torch.Tensor(x)
                y = torch.LongTensor(y)
                # wrap in Variable
                x = Variable(x)
                y = Variable(y)
                # zero the parameter gradients
                self.zero_grad()
                # forward + backward + optimize
                outputs = self.forward(x)
                loss = criterion(outputs, y)
                loss.backward()

                # update the weights
                for f in self.parameters():
                    f.data.sub_(f.grad.data * lr)
    
                # print statistics
                running_loss += loss.data[0]
    
            print('Epoch {}, loss {}'.format(epoch, running_loss))
            running_loss = 0.0
    
        print('Finished Training.')
    
        # testing
        correct = 0
        total = 0
        x, y = validation_data
        # wrap in tensor
        x = torch.Tensor(x)
        y = torch.LongTensor(y)
        outputs = self.forward(Variable(x))
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum()
        print(correct)
        exit()
            
    
        print('Accuracy {}'.format(100*correct / total))

### Miscellaneous functions
def shuffle(data):
    X, y = data
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    return X, y

def main():
    training_data, validation_data, _ = mnist_loader.load_data()
    print(training_data[0].shape, training_data[1].shape)
    net = Net()
    print(net)
    net.SGD(training_data, 5, 10, 0.1, validation_data)

    

if __name__ == '__main__':
    main()
