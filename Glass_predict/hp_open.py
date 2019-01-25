import cv2 as cv2
import numpy as np
import h5py


train_dataset = h5py.File('D:/git/datasets/acc_train_set_128.hdf5', "r")
train_set_x_orig = np.array(train_dataset["X_train"][:])  # train set features
train_set_y = np.array(train_dataset["Y_train"][:])  # train set features

test_set_x_orig = np.array(train_dataset["X_train"][:])  # train set features
test_set_y = np.array(train_dataset["Y_train"][:])  # train set features

print(str(train_set_x_orig.shape))
print(str(train_set_y.shape))

print(str(test_set_x_orig.shape))
print(str(test_set_y.shape))

train_set_y = train_set_y.reshape((train_set_y.shape[0], train_set_y.shape[1]))
test_set_y = test_set_y.reshape((test_set_y.shape[0], test_set_y.shape[1]))
print("-------------------")

print(str(train_set_x_orig.shape))
print(str(train_set_y.shape))

print(str(test_set_x_orig.shape))
print(str(test_set_y.shape))

print(str(train_set_y[12]))
print(str(test_set_y[0]))

for i in range(0, train_set_x_orig.shape[0]):
    index = i
    img = train_set_x_orig[index, :]
    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 32:
            print(str(train_set_y[index]))
            break

