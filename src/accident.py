#!/usr/bin/env python3
"""
accident.py
"""
# Standard Library
import glob
import json

# Third-party Libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

# set random seet for reproducibility
#np.random.seed(1234)

### Define the quadratic and cross-entropy cost functions

class CrossEntropyCost():
    """
    Cross Entropy Cost Function and Derivative.
    """
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return a-y

class QuadraticCost():
    """
    Quadratic Cost Function and Derivative.
    """
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


### Main Network class
class Network():
    """
    Neural Network Class.
    """
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        New and improved approach to weight initializaion.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) / np.sqrt(x)
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y)
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(a, w) + b.T)
        return a

    def SGD(self, training_data, epochs, batch_size, lr, reg=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        """
        if evaluation_data:
            n_data = len(evaluation_data[0])
        n = len(training_data[0])
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for i in range(epochs):
            training_data = shuffle(training_data)
            mini_batches = [
                    (training_data[0][k:k+batch_size],
                    training_data[1][k:k+batch_size])
                    for k in range(0, n, batch_size)
                    ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr, reg, n)
            print('\tEpoch {} training complete.'.format(i))
            if monitor_training_cost:
                cost = self.total_cost(training_data, reg)
                training_cost.append(cost)
                print('\tCost on training data: {}'.format(cost)) 
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print('\tAccuracy on training data: {} / {}'.format(accuracy, 
                    n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, reg)
                evaluation_cost.append(cost)
                print('\tCost on evaluation data: {}'.format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data) / n_data * 100
                evaluation_accuracy.append(accuracy)
                print('\tAccuracy on evaluation data: {}'.format(accuracy))
            print()
        return evaluation_cost, evaluation_accuracy, \
                training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, lr, reg, n):
        x, y = mini_batch
        db, dw = self.backprop(x, y)
        self.weights = [(1-lr*(reg/n))*w - (lr/len(x))*nw
                for w, nw in zip(self.weights, dw)]
        self.biases = [b - (lr/len(x))*nb 
                for b, nb in zip(self.biases, db)]

    def backprop(self, x, y):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w) + b.T
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        db[-1] = np.reshape(np.sum(delta, axis=0), (-1, 1))
        dw[-1] = np.dot(activations[-2].T, delta)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(delta, self.weights[-l+1].T) * sp
            db[-l] = np.reshape(np.sum(delta, axis=0), (-1, 1))
            dw[-l] = np.dot(activations[-l-1].T, delta)
        return db, dw

    def accuracy(self, data):
        x, y = data
        a = self.feedforward(x)
        y_pred = np.argmax(a, axis=1)
        y = np.argmax(y, axis=1)
        return np.sum(y == y_pred)

    def total_cost(self, data, reg):
        """
        Return total cost for ``data``.
        """
        cost = 0.0
        X_data, y_data = data
        n = 3000
        for i in range(0, len(X_data), n):
            x = X_data[i:i+n]
            y = y_data[i:i+n]
            a = self.feedforward(x)
            cost += self.cost.fn(a, y) / len(X_data)
        cost += 0.5 * (reg/len(X_data)) * sum(np.linalg.norm(w)**2 
                for w in self.weights)
        return cost

    def save(self, filename):
        """
        Save the neural network to the file ``filename``.
        """
        data = {'sizes': self.sizes,
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'cost': str(self.cost.__name__)}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()

### Loading a Network
def load(filename):
    """
    Load a neural network from file.
    """
    f = open(filename, 'r')
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data['cost'])
    net = Network(data['sizes'], cost=cost)
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net

### Miscellaneous functions
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def read_data(data_path, width, height):
    """
    Read data and store in numpy arrays.
    """
    imgs_pos = glob.glob(data_path + 'positive/*.jpg')
    imgs_neg = glob.glob(data_path + 'negative/*.jpg')
    num_instances = len(imgs_pos) + len(imgs_neg)
    X = np.zeros((num_instances, height, width, 3), dtype=np.float32)
    i = 0
    for img_path in imgs_pos:
        print(i)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))
        # standardize
        img_min = img - img.min()
        img = img / img_min.max()
        X[i] = img
        i += 1
    for img_path in imgs_neg:
        print(i)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))
        # standardize
        img_min = img - img.min()
        img = img / img_min.max()
        X[i] = img
        i += 1
    y = np.zeros((num_instances, 2), dtype=np.int)
    y[:len(imgs_pos), 1] = 1
    y[len(imgs_pos):, 0] = 1
    np.save('../data/X.npy', X)
    np.save('../data/y.npy', y)
    return X, y

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
    """
    Main Function.
    """
#    datapath = '../data/dashcams/'
#    width = height = 100
#    read_data(datapath, width, height)
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
    # create neural network
    num_features = training_data[0].shape[1]
    
    # learning rates
    learning_rate = 0.1
    # number of hidden units
    num_hidden = 200
    net = Network([num_features, num_hidden, 2], cost=QuadraticCost)
#    net.large_weight_initializer()
    _, val_acc, train_cost, _ = net.SGD(training_data, 30, 10, learning_rate, 
            reg=0, evaluation_data=validation_data,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=False,
            monitor_training_accuracy=False,
            monitor_training_cost=True)

    # display plots
    plt.figure()
    plt.subplot(121)
    plt.plot(train_cost, label='num_hidden=' + str(num_hidden))
    plt.title('Training Cost for lr =' + str(learning_rate))
    plt.xlabel('Epoch'), plt.ylabel('Cost'), plt.ylim(0, 1)

    plt.subplot(122)
    plt.plot(val_acc, label='num_hidden=' + str(num_hidden))
    plt.title('Validation Accuracy for lr =' + str(learning_rate))
    plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.ylim(0, 100)
    plt.legend(loc='lower right')

    plt.savefig(str(learning_rate) + '.png')

    plt.show()

if __name__ == '__main__':
    main()
