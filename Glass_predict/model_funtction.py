import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from nn_functions import *


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(dtype=tf.float32, shape=[None, n_H0, n_W0, n_C0], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name="Y")

    return X, Y


def compute_cost(Z, Y, regularizer=None):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z, labels=Y))
    # Regularize
    if regularizer is not None:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    else:
        reg_term = 0
    cost += reg_term
    return cost


def initialize_parameters_L5(beta=0):

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    # Regularization
    if beta != 0:
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
    else:
        regularizer = None

    W1 = tf.get_variable("W1", [4, 4, 3, 4], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    W2 = tf.get_variable("W2", [4, 4, 4, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    W3 = tf.get_variable("W3", [2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    W4 = tf.get_variable("W4", [4, 4, 32, 40], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    b1 = tf.get_variable("b1", [1, 4], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    b2 = tf.get_variable("b2", [1, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    b3 = tf.get_variable("b3", [1, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    b4 = tf.get_variable("b4", [1, 40], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "b1": b1,
                  "b2": b2,
                  "b3": b3,
                  "b4": b4}

    return parameters, regularizer


def forward_propagation_L_5(X, parameters, regularizer=None):

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    b4 = parameters["b4"]

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    Z1 += b1
    # RELU
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.avg_pool(A1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    Z2 += b2
    # RELU
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    Z3 += b3
    # RELU
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.avg_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='SAME')

    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
    Z4 += b4
    # RELU
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.avg_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    P4 = tf.contrib.layers.flatten(P4)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 3 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z5 = tf.contrib.layers.fully_connected(P4, num_outputs=3, activation_fn=None, weights_regularizer=regularizer)
    print(str(Z5))
    return Z5


def initialize_parameters_l5_2fc(beta=0):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    # Regularization
    if beta != 0:
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
    else:
        regularizer = None

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    W2 = tf.get_variable("W2", [4, 4, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    W3 = tf.get_variable("W3", [2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    W4 = tf.get_variable("W4", [512, 160], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    b1 = tf.get_variable("b1", [1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    b2 = tf.get_variable("b2", [1, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    b3 = tf.get_variable("b3", [1, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)
    b4 = tf.get_variable("b4", [1, 160], initializer=tf.contrib.layers.xavier_initializer(seed=0),
                         regularizer=regularizer)

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "b1": b1,
                  "b2": b2,
                  "b3": b3,
                  "b4": b4}

    return parameters, regularizer


def forward_propagation_l5_2fc(X, parameters, regularizer=None):

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    b4 = parameters["b4"]
    # -------------------------------------------------------------------------------------
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    Z1 += b1
    # RELU
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.avg_pool(A1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')
    # -------------------------------------------------------------------------------------
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 2, 2, 1], padding='SAME')
    Z2 += b2
    # RELU
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # -------------------------------------------------------------------------------------
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    Z3 += b3
    # RELU
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.avg_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='VALID')
    # -------------------------------------------------------------------------------------
    # FLATTEN
    print(str(P3))
    P3 = tf.contrib.layers.flatten(P3)
    print(str(P3))
    # -------------------------------------------------------------------------------------
    Z4 = tf.matmul(P3, W4) + b4
    A4 = tf.nn.relu(Z4)
    # -------------------------------------------------------------------------------------
    Z5 = tf.contrib.layers.fully_connected(A4, num_outputs=3, activation_fn=None, weights_regularizer=regularizer)
    return Z5
