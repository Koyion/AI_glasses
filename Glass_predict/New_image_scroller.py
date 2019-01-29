import cv2 as cv2
import os
import sys
import numpy as np
import h5py
import scipy.misc
from PIL import Image
# from imgaug import augmenters as iaa # for augumentation

# TRAIN SET COMPLETING
photoGlassesPath = sys.argv[1]
photoSunglassesPath = sys.argv[2]
photoPersonPath = sys.argv[3]
y_set = []
train_set = []
num_px = 256

for i in os.listdir(photoGlassesPath):
    img = cv2.imread(photoGlassesPath + i, 1)
    tmp_img = scipy.misc.np.array(Image.fromarray(img).resize([num_px, num_px]))
    train_set.append(np.array(tmp_img))
    cls = np.array([1, 0, 0])  # glasses
    y_set.append(cls)

for i in os.listdir(photoSunglassesPath):
    img = cv2.imread(photoSunglassesPath + i, 1)
    tmp_img = scipy.misc.np.array(Image.fromarray(img).resize([num_px, num_px]))
    train_set.append(np.array(tmp_img))
    cls = np.array([0, 1, 0])  # sunglasses
    y_set.append(cls)

for i in os.listdir(photoPersonPath):
    img = cv2.imread(photoPersonPath + i, 1)
    tmp_img = scipy.misc.np.array(Image.fromarray(img).resize([num_px, num_px]))
    train_set.append(np.array(tmp_img))
    cls = np.array([0, 0, 1])  # person
    y_set.append(cls)

# Reformatting
train_set = np.array(train_set)
y_set = np.array(y_set)

# Shuffeling
m = train_set.shape[0]  # number of training examples
assert train_set.shape[0] == y_set.shape[0]
mini_batches = []
np.random.seed(1)

# Shuffle (X, Y)
permutation = list(np.random.permutation(m))
shuffled_X = train_set[permutation, :, :, :]
shuffled_Y = y_set[permutation, :]
# Print shapes
print("Shape of x_set_0 now: " + str(shuffled_X.shape))
print("Shape of y_set now: " + str(shuffled_Y.shape))
# Print total guesses
counter = shuffled_Y.sum(axis=0)
print("Glasses in set: " + str(counter[0]))
print("Sunglasses in set: " + str(counter[1]))
print("Backgrounds in set: " + str(counter[2]))
# Preprocess for h5
train_set = {
    "X_train": shuffled_X,
    "Y_train": shuffled_Y
}
# Generate hp files
with h5py.File('test/train_set_256.hdf5', 'w') as f:
    d_set = f.create_dataset("X_train", data=train_set["X_train"])
    d_set = f.create_dataset("Y_train", data=train_set["Y_train"])
