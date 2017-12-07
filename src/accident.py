#!/usr/bin/env python3
import random
import numpy as np
import glob
import cv2

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
    y = np.zeros((num_instances, 2), dtype=np.uint8)
    y[:len(imgs_pos), 1] = 1
    y[len(imgs_pos):, 0] = 1
    np.save('X.npy', X)
    np.save('y.npy', y)
    return X, y

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(a, w) + b.T)
        return a

    def SGD(self, X_train, y_train, epochs, batch_size, lr, X_test=None,
            y_test=None):
        if X_test is not None and y_test is not None:
            n_test = len(X_test)
        n = len(X_train)
        for i in range(epochs):
            # random shuffle
            idx = np.random.permutation(n)
            X_train = X_train[idx]
            y_train = y_train[idx]
            mini_batches = [(X_train[k:k+batch_size], y_train[k:k+batch_size])
                    for k in range(0, n, batch_size)]
            c = 0.0
            for mini_batch in mini_batches:
                c += self.update_mini_batch(mini_batch, lr)
            c /= (2*len(X_train))
            if X_test is not None and y_test is not None:
                print("Epoch {0}: cost {1}, accuracy: {2}".format(i, c, 
                    self.evaluate(X_test,y_test)/n_test))
            else:
                print("Epoch {0}: cost {1}".format(i, c))

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
        c = self.cost(activation, y)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        db[-1] = np.reshape(np.sum(delta, axis=0), (-1, 1))
        dw[-1] = np.dot(activations[-2].T, delta)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(delta, self.weights[-l+1].T) * sp
            db[-l] = np.reshape(np.sum(delta, axis=0), (-1, 1))
            dw[-l] = np.dot(activations[-l-1].T, delta)
        return db, dw, c

    def update_mini_batch(self, mini_batch, lr):
        x, y = mini_batch
        db, dw, c = self.backprop(x, y)
        self.weights = [w - (lr/len(x))*nw 
                for w, nw in zip(self.weights, dw)]
        self.biases = [b - (lr/len(x))*nb 
                for b, nb in zip(self.biases, db)]
        return c

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def cost(self, output_activations, y):
        c = np.sum((y - output_activations)**2)
        return c

    def evaluate(self, X_test, y_test):
        outputs = self.feedforward(X_test)
        y_pred = np.argmax(outputs, axis=1)
        y = np.argmax(y_test, axis=1)
        total = np.sum(y_pred == y)
        return total

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def main():
    """
    Main Function.
    """
    # load data
    X = np.load('X.npy')
    y = np.load('y.npy')
    # reshape into num_instances x num_features
    X = np.reshape(X, (len(X), -1))
    # shuffle data
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    # split data
    X_train, X_val, X_test = np.split(X, [int(0.6*len(X)), int(0.8*len(X))])
    y_train, y_val, y_test = np.split(y, [int(0.6*len(y)), int(0.8*len(y))])
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_val:', X_val.shape)
    print('y_val:', y_val.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)
    # create neural network
    num_features = X_train.shape[1]
    net = Network([num_features, 50, 2])
    net.SGD(X_train, y_train, 100, 10, 0.1, X_val, y_val)


if __name__ == '__main__':
    main()
