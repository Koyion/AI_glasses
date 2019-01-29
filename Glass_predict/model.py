from model_functions import *
from nn_functions import *
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

def model(num_packs=5, tr_dataset_part=512, learning_rate=0.01, num_epochs=120,
          minibatch_size=16, print_cost=True, beta=0.0):

    X_train_beg, Y_train_beg, X_test_beg, Y_test_beg, classes = load_dataset_parts(0, 1, 2, 1)

    print("number of training examples in begin = " + str(X_train_beg.shape[0]))
    print("number of test examples in begin = " + str(X_test_beg.shape[0]))
    print("X_train shape: " + str(X_train_beg.shape))
    print("Y_train shape: " + str(Y_train_beg.shape))
    print("X_test shape: " + str(X_test_beg.shape))
    print("Y_test shape: " + str(Y_test_beg.shape))

    ops.reset_default_graph()
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train_beg.shape
    n_y = Y_train_beg.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    # Initialize parameters
    parameters, regularizer = initialize_parameters_l5_2fc_nb(beta=beta)
    # Forward propagation
    Z_L = forward_propagation_l5_2fc_nb(X, parameters, regularizer=regularizer)
    y_pred = tf.nn.softmax(Z_L, name='y_pred')
    # Cost function
    cost = compute_cost(Z_L, Y, regularizer=regularizer)
    # Backpropagation
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
        print(str("--------------------------------------"))
        # Do the training loop
        for epoch in range(num_epochs):
            te_begin = 0  # begin of the test set
            pack = 0  # to start set from begin
            minibatch_cost = 0.
            for i in range(num_packs):
                X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset_parts(pack, pack + tr_dataset_part,
                                                                                         te_begin)
                # print("SET IS: " + str(pack) + " to: " + str(pack + tr_dataset_part))
                pack += tr_dataset_part
                X_train = X_train_orig / 255.
                (m, n_H0, n_W0, n_C0) = X_train.shape
                # print("--------------------------------------")
                num_minibatches = int(num_packs * tr_dataset_part / minibatch_size)
                # number of minibatches of size minibatch_size in the train set
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed=0)
                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                    _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                    minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
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

        # Calculate accuracy on the train and test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        X_train_eval, Y_train_eval, X_test_orig_eval, Y_test_eval = load_dataset_eval(0, 500, 8200)
        X_train = X_train_eval / 255.
        X_test = X_test_orig_eval / 255.
        train_accuracy = accuracy.eval({X: X_train[:], Y: Y_train_eval[:]})
        test_accuracy = accuracy.eval({X: X_test[:], Y: Y_test_eval[:]})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        save_path = saver.save(sess, "models/glass_model")
        print("Model saved in path: %s" % save_path)
        return train_accuracy, test_accuracy, parameters


# num_packs = 5 tr_dataset_part = 512 for 2560 examples
_, _, parameters = model(num_packs=2, tr_dataset_part=512,
                         learning_rate=0.015, num_epochs=40, minibatch_size=64, beta=0.01)
