#!/usr/bin/env python3
import numpy as np
import cv2
import math

def perceptron(x):
    W = np.random.randn(x.shape[0], 1)
    b = np.random.randn(1)
    output = np.dot(x.T, W) + b
    if output <= 0:
        output = 0
    else:
        output = 1
    return output

def sigmoid(x):
    W = np.random.randn(x.shape[0], 1)
    b = np.random.randn(1)
    z = np.dot(x.T, W) + b
    # clip to remove double precision errors (i.e. overflow)
    z = np.clip(z, -500, 500)
    # compute sigmoid
    output = 1 / (1 + math.exp(-z[0,0]))
    return output

def main():
    """ Main Function. """
    # read input image
    img = cv2.imread('/home/gary/opencv/samples/data/lena.jpg', 
            cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100,100))
    img_flatten = np.reshape(img, (-1, 1))
    # standardization
    img_mean = np.mean(img_flatten)
    img_std = np.std(img_flatten)
    img_stand = (img_flatten - img_mean) / img_std
    # normalization
    img_min = img_flatten - img_flatten.min()
    img_norm = img_min / (img_min.max())
    # perceptron
    output = perceptron(img_stand)
    print('perceptron:', output)
    output = sigmoid(img_stand)
    print('sigmoid output:', output)

if __name__ == '__main__':
    main()
