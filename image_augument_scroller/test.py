import cv2 as cv2
import os
import sys
import numpy as np
import h5py
import scipy.misc
from PIL import Image
from imgaug import augmenters as iaa
from Class_selcet import image_class


# ---------------------------------------------------------------------------------------------
augumenter = iaa.Sequential([
    iaa.Fliplr(1.),  # horizontally flip 100% of the images
    iaa.Crop(px=(20, 55)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.5, 1.8))  # blur images with a sigma of 0.5 to 1.75
])

augumenter_two = iaa.Sequential([
    iaa.Add((-10, 10), per_channel=True),  # change brightness of images (by -10 to 10 of original value)
    iaa.Fliplr(0.5),  # horizontally flip 100% of the images
    iaa.Crop(px=(3, 7)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.4, 1.45))  # blur images with a sigma of 0.5 to 1.75
])

augumenter_test = iaa.Sequential {
    [
        iaa.Affine([
                scale = {"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
                translate_percent = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate = (-45, 45),  # rotate by -45 to +45 degrees
                shear = (-16, 16),  # shear by -16 to +16 degrees
                order = [0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval = (0, 255),  # if mode is constant, use a cval be              tween 0 and 255
                mode = ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ])
    ]
}

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

    tmp_aug = augumenter.augment_image(tmp_img)
    train_set.append(np.array(tmp_aug))

    cls = np.array([1, 0, 0])  # glasses
    y_set.append(cls)

for i in os.listdir(photoSunglassesPath):
    img = cv2.imread(photoSunglassesPath + i, 1)
    tmp_img = scipy.misc.np.array(Image.fromarray(img).resize([num_px, num_px]))
    train_set.append(np.array(tmp_img))

    tmp_aug = augumenter.augment_image(tmp_img)
    train_set.append(np.array(tmp_aug))

    cls = np.array([0, 1, 0])  # sunglasses
    y_set.append(cls)

for i in os.listdir(photoPersonPath):
    img = cv2.imread(photoPersonPath + i, 1)
    tmp_img = scipy.misc.np.array(Image.fromarray(img).resize([num_px, num_px]))
    train_set.append(np.array(tmp_img))

    tmp_aug = augumenter.augment_image(tmp_img)
    train_set.append(np.array(tmp_aug))

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
with h5py.File('train_data/inst_aug_train_set_256.hdf5', 'w') as f:
    d_set = f.create_dataset("X_train", data=train_set["X_train"])
    d_set = f.create_dataset("Y_train", data=train_set["Y_train"])
