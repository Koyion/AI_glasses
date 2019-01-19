import cv2 as cv2
import numpy as np
import h5py

# OPEN IMAGE TEST
train_dataset = h5py.File('train_set_512.hdf5', "r")
train_set_x_orig = np.array(train_dataset["X_train"][:])  # train set features
train_set_y = np.array(train_dataset["Y_train"][:])  # train set features

print(str(train_set_x_orig.shape))
print(str(train_set_y.shape))


for i in range(0, train_set_x_orig.shape[0]):
    index = i
    img = train_set_x_orig[index, :]
    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 32:
            print("good")
            break
