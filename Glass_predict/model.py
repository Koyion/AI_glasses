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
from model_funtction import *


X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.01,
          num_epochs=120, minibatch_size=16, print_cost=True, beta=0.0):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters, regularizer = initialize_parameters_l5_2fc_nb(beta=beta)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z_L = forward_propagation_l5_2fc_nb(X, parameters, regularizer=regularizer)

    y_pred = tf.nn.softmax(Z_L, name='y_pred')

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z_L, Y, regularizer=regularizer)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    # Configure CPU usage
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8
    # Start the session to compute the tensorflow graph
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        # Run the initialization
        sess.run(init)
        print(str("----------------------------------------"))
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed=0)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 2 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()
        # lets save the parameters in a variable
        # parameters = sess.run(parameters)
        print("Parameters have been trained!")
        # Calculate the correct predictions
        predict_op = tf.argmax(Z_L, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train[:300], Y: Y_train[:300]})
        test_accuracy = accuracy.eval({X: X_test[:], Y: Y_test[:]})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        save_path = saver.save(sess, "datasets/model_test")
        print("Model saved in path: %s" % save_path)
        return train_accuracy, test_accuracy, parameters


_, _, parameters = model(X_train, Y_train, X_test, Y_test,
                         learning_rate=0.01, num_epochs=20, minibatch_size=64, beta=0.001)
