import cv2 as cv2
import numpy as np
import h5py
import time

train_dataset = h5py.File('datasets/acc_train_set_512.hdf5', "r")
train_set_x_orig = np.array(train_dataset["X_train"][:])  # train set features
train_set_y = np.array(train_dataset["Y_train"][:])  # train set features

print(str(train_set_x_orig.shape))
print(str(train_set_y.shape))
print("-------------------")
print(str(train_set_x_orig.shape))
print(str(train_set_y.shape))

assert train_set_x_orig.shape[0] == train_set_y.shape[0]

gl = 1
sgl = 1
bak = 1
for i in range(0, train_set_x_orig.shape[0]):

    img = train_set_x_orig[i, :]
    cls = train_set_y[i]
    if (cls == [[1], [0], [0]]).all():
        cv2.imwrite('datasets/1/' + str(gl) + '.jpg', img)
        print("Glasses")
        gl += 1
    elif (cls == [[0], [1], [0]]).all():
        cv2.imwrite('datasets/2/' + str(sgl) + '.jpg', img)
        print("Sunglasses")
        sgl += 1
    else:
        cv2.imwrite('datasets/3/' + str(bak) + '.jpg', img)
        print("Background")
        bak += 1

print("Glasses: " + str(gl))
print("Sunglasses: " + str(sgl))
print("Background: " + str(bak))
