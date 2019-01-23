import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


def load_dataset():
    train_dataset = h5py.File('D:/git/datasets/acc_train_set_256.hdf5', "r")

    # train_set_x_orig = np.array(train_dataset["X_train"][:6136])  # train set features
    # train_set_y_orig = np.array(train_dataset["Y_train"][:6136])  # train set features

    # test_set_x_orig = np.array(train_dataset["X_train"][6137:])  # train set features
    # test_set_y_orig = np.array(train_dataset["Y_train"][6137:])  # train set features

    train_set_x_orig = np.array(train_dataset["X_train"][:2048])  # train set features
    train_set_y_orig = np.array(train_dataset["Y_train"][:2048])  # train set features

    test_set_x_orig = np.array(train_dataset["X_train"][2048:2300])  # train set features
    test_set_y_orig = np.array(train_dataset["Y_train"][2048:2300])  # train set features

    classes = {
        "glasses": np.array([1, 0, 0]),
        "sunglasses": np.array([0, 1, 0]),
        "background": np.array([0, 0, 1])
    }

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], train_set_y_orig.shape[1]))
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], test_set_y_orig.shape[1]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.avg_pool(A1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.avg_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='SAME')

    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.avg_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    P4 = tf.contrib.layers.flatten(P4)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 3 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z5 = tf.contrib.layers.fully_connected(P4, num_outputs=3, activation_fn=None)
    return Z5


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    params = {"W1": W1,
              "W2": W2,
              "W3": W3,
              "W4": W4}

    x = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))

    z3 = forward_propagation_for_predict(x, params)
    print(str(z3))
    p = tf.argmax(z3)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    prediction = sess.run(p, feed_dict={x: X})

    return prediction
