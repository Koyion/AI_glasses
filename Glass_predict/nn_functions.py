import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


def load_dataset():
    train_dataset = h5py.File('train_data/inst_train_set_256.hdf5', "r")

    train_set_x_orig = np.array(train_dataset["X_train"][:2560])  # train set features
    train_set_y_orig = np.array(train_dataset["Y_train"][:2560])  # train set features

    test_set_x_orig = np.array(train_dataset["X_train"][2561:2813])  # test set features
    test_set_y_orig = np.array(train_dataset["Y_train"][2561:2813])  # test set features

    classes = {
        "glasses": np.array([1, 0, 0]),
        "sunglasses": np.array([0, 1, 0]),
        "background": np.array([0, 0, 1])
    }

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], train_set_y_orig.shape[1]))
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], test_set_y_orig.shape[1]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_dataset_parts(tr_begin, tr_end, te_begin, te_size=200):
    train_dataset = h5py.File('train_data/inst_train_set_256.hdf5', "r")
    train_set_x_orig = np.array(train_dataset["X_train"][tr_begin:tr_end])  # train set features
    train_set_y_orig = np.array(train_dataset["Y_train"][tr_begin:tr_end])  # train set features

    test_set_x_orig = np.array(train_dataset["X_train"][te_begin:te_begin + te_size])  # test set features
    test_set_y_orig = np.array(train_dataset["Y_train"][te_begin:te_begin + te_size])  # test set features

    classes = {
        "glasses": np.array([1, 0, 0]),
        "sunglasses": np.array([0, 1, 0]),
        "background": np.array([0, 0, 1])
    }

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], train_set_y_orig.shape[1]))
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], test_set_y_orig.shape[1]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_dataset_eval(tr_begin, tr_end, te_begin, te_size=200):
    train_dataset = h5py.File('train_data/inst_train_set_256.hdf5', "r")
    train_set_x_orig = np.array(train_dataset["X_train"][tr_begin:tr_end])  # train set features
    train_set_y_orig = np.array(train_dataset["Y_train"][tr_begin:tr_end])  # train set features

    test_set_x_orig = np.array(train_dataset["X_train"][te_begin:te_begin + te_size])  # test set features
    test_set_y_orig = np.array(train_dataset["Y_train"][te_begin:te_begin + te_size])  # test set features

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], train_set_y_orig.shape[1]))
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], test_set_y_orig.shape[1]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size
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
