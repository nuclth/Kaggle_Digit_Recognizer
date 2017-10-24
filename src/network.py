#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:31:45 2017

@author: alex
"""

## import libraries
# standard libraries
import sys
import random
import gc

# third party libraries
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import svm

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, train_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n = len(train_data)
#        tr_d = self.format_data(train_data)
#        tr_d = list(tr_d)
        
#        print (train_data[0][0])
#        print (train_data[0][1])
#        sys.exit()
        
        if test_data:
            n_test = len(test_data)
#            te_d = self.format_data(test_data)
#            te_d = list (te_d)
            print ('Formating test data')
            
        for j in range(epochs):
            random.shuffle(train_data)
            mini_batches = [
                train_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for image, value in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(image, value)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, image, value):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = image
        activation = activation.reshape(len(activation), 1)
#        print (image)
        activations = [activation] # list to store all the activations, layer by layer
#        print (activation.shape)
        zs = [] # list to store all the z vectors, layer by layer
#        print ("BEGIN B,W LOOP")
        for b, w in zip(self.biases, self.weights):
#            print (activation.shape)
            z = np.dot(w, activation) + b
#            z = z.reshape (len(z), 1) + b
#            print (z.shape)
            zs.append(z)
            activation = sigmoid(z)
#            print (activation.shape)
#            print (activations[0])
            activations.append(activation)
#        print (activations)
#        print ("END B, W LOOP")
#        sys.exit()
#        sys.exit()
        # backward pass
#        print (activations[-1].shape)
#        print (value.shape)
#        print (activations[-1])
#        print (value)
        delta = self.cost_derivative(activations[-1], value) * sigmoid_prime(zs[-1])
#        print (delta.shape)
#        print (delta)
#        print (delta[0])
#        print (delta[0][0])
#        sys.exit()
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
#        print (nabla_w[-1].shape)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
#        print ("BEGIN L LOOP")
        for l in range(2, self.num_layers):
#            print (l)
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
#            print (delta.shape)
            nabla_b[-l] = delta
            inter = np.array(activations[-l-1])
            inter = inter.reshape(len(activations[-l-1]), 1)
            inter = inter.transpose()
            nabla_w[-l] = np.dot(delta,inter)
#            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
#        print ("END L loop")
#        print (nabla_b[0].shape)
#        print (nabla_w[0].shape)
#        print (nabla_b[1].shape)
#        print (nabla_w[1].shape)
#        sys.exit()
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(image)), value)
                        for image, value in test_data]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
#        print (output_activations)
        cost = np.array(output_activations)
        cost[y] = cost[y] - 1.0
        return cost
    
    def format_data(self, data):
        """Stuff goes here"""
        inputs = [np.reshape(x, (784, 1)) for x in data[:,1:]]
        outputs = [y for y in data[:,0]] 
        formatted_data = zip(inputs, outputs)
        return formatted_data

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return (1.0/(1.0 + np.exp(-z)))
#    sig = np.zeros(len(z))
#    for a in range(0, len(z)):
#        if z[a] > 700.:
#            sig[a] = 1.
#        elif z[a] < -700.:
#            sig[a] = 0.
#        else:
#            sig[a] = (1.0/(1.0 + np.exp(-z[a])))
#    sig = sig.reshape(len(sig), 1)
#    return sig
#     return (1.0/(1.0 + np.exp(-z)))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



def svm_baseline():
    pass
#    training_data, validation_data, test_data = mnist_loader.load_data()
    # train
#    clf = svm.SVC()
#    clf.fit(training_data[0], training_data[1])
    # test
#    predictions = [int(a) for a in clf.predict(test_data[0])]
#    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
#    print "Baseline classifier using an SVM."
#    print "%s of %s values correct." % (num_correct, len(test_data[1]))

def format_data(data):
    """Stuff goes here"""
    inputs = [x for x in data[:,1:]]
    outputs = [y for y in data[:,0]] 
    formatted_data = zip(inputs, outputs)
    return formatted_data



if __name__ == '__main__':
    
#    import math
    
    training_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    
    #print(training_data.head())
    
    train_ar = np.array(training_data.iloc[:10000])
    valid_ar = np.array(training_data.iloc[10000:20000])
    test_ar  = np.array(test_data)

    train_list = format_data (train_ar)
    valid_list = format_data (valid_ar)
    
    train_list = list (train_list)
    valid_list = list (valid_list)
    
#    baseline = svm.SVC()
    
#    input_svm = [image[0] for image in train_list]
#    output_svm = [image[1] for image in train_list] 

    testing_in = [image[0] for image in valid_list]
    testing_out = [image[1] for image in valid_list]

    print (train_list[0][0])
    print (train_list[0][1])
    print (len(train_list))
      

    
#    for i in range(0,784):
#        num = input_svm[1][i]
#        print (num, end='')
#        digits = 1
#        if num != 0:
#            digits = int(math.log10(num))+1
#        if digits == 1:
#            print ("    ", end='')
#        elif digits == 2:
#            print ("   ", end='')
#        elif digits == 3:
#            print ("  ", end='')
#        if i % 28 == 0 and i != 0:
#            print ('\n')
    
#    sys.exit()
    
#    print (output_svm[0])

#    print (len(input_svm))
#    print (len(output_svm))

#    print ("Performing SVM Fit")

#    baseline.fit (input_svm, output_svm)
    
#    print ("Fit complete")
    
#    predictions = [int(a) for a in baseline.predict(testing_in)]
    
#    print ("Predictions complete")
    
#    correct = sum(int(a==y) for a, y in zip(predictions, testing_out))
    
#    print ("Baseline SVM fit.")
#    print ("{} of {} values correct".format(correct, len(testing_out)))
    
#    sys.exit()
    net = Network([784, 30, 10])
    net.SGD(train_list, 30, 10, 3.0, test_data = valid_list)
    #print (train_ar.shape)
    #print (valid_ar.shape)
    #print (test_ar.shape)
