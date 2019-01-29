import tensorflow as tf


def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(dtype=tf.float32, shape=[None, n_H0, n_W0, n_C0], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name="Y")

    return X, Y


def compute_cost(Z, Y, regularizer=None):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z, labels=Y))
    # Regularize
    if regularizer is not None:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    else:
        reg_term = 0
    cost += reg_term
    return cost


def initialize_parameters_l5_2fc_nb(beta=0.0):

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

    parameters = {
                  "W1": W1,
                  "W2": W2,
                  "W3": W3
                  }

    return parameters, regularizer


def forward_propagation_l5_2fc_nb(X, parameters, regularizer=None):

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    # -------------------------------------------------------------------------------------
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.avg_pool(A1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')
    # -------------------------------------------------------------------------------------
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 2, 2, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # -------------------------------------------------------------------------------------
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.avg_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='VALID')
    # -------------------------------------------------------------------------------------
    # FLATTEN
    P3 = tf.contrib.layers.flatten(P3)
    # -------------------------------------------------------------------------------------
    Z4 = tf.contrib.layers.fully_connected(P3, num_outputs=160, activation_fn=tf.nn.relu,
                                           biases_initializer=tf.zeros_initializer(), biases_regularizer=regularizer,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                           weights_regularizer=regularizer)
    # -------------------------------------------------------------------------------------
    Z5 = tf.contrib.layers.fully_connected(Z4, num_outputs=3, activation_fn=None)
    return Z5
