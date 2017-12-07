#!/usr/bin/env python3
"""
Run NN modules on MNIST data.
"""
import mnist_loader
import network

def main():
    """ Main Function. """
    print(__doc__)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # create neural network object
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 2, 10000, 3.0, test_data=test_data)

if __name__ == '__main__':
    main()
