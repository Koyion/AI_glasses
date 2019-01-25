import time
import numpy as np
import h5py


def load_dataset_parts(tr_begin, tr_end, te_begin, te_size=200):
    train_dataset = h5py.File('datasets/acc_train_set_256.hdf5', "r")
    train_set_x_orig = np.array(train_dataset["X_train"][tr_begin:tr_end])  # train set features
    train_set_y_orig = np.array(train_dataset["Y_train"][tr_begin:tr_end])  # train set features

    test_set_x_orig = np.array(train_dataset["X_train"][te_begin:te_begin + te_size])  # train set features
    test_set_y_orig = np.array(train_dataset["Y_train"][te_begin:te_begin + te_size])  # train set features

    classes = {
        "glasses": np.array([1, 0, 0]),
        "sunglasses": np.array([0, 1, 0]),
        "background": np.array([0, 0, 1])
    }

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], train_set_y_orig.shape[1]))
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], test_set_y_orig.shape[1]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


tr_dataset_part = 512
te_begin = 3000
pack = 0
for i in range(5):
    X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset_parts(pack, pack + tr_dataset_part, te_begin)
    print("SET IS: " + str(pack) + " to: " + str(pack + tr_dataset_part))
    pack += tr_dataset_part
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    print("--------------------------------------")
    time.sleep(2)


