import cv2 as cv2
import os
import sys
import numpy as np
import h5py
import scipy.misc
from PIL import Image
from imgaug import augmenters as iaa
from Class_selcet import select_class

# TRAIN SET COMPLETING
photoDumpPath = sys.argv[1]  # PATH WITH IMAGE FOLDER
train_image_x_set = []
y_set = []
num_px = 128
num_aug = 2

augumenter_one = iaa.Sequential([
    iaa.Fliplr(1.),  # horizontally flip 100% of the images
    iaa.Crop(px=(20, 60)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.5, 1.75))  # blur images with a sigma of 0.5 to 1.75
])

augumenter_two = iaa.Sequential([
    iaa.Add((-10, 10), per_channel=True),  # change brightness of images (by -10 to 10 of original value)
    iaa.Fliplr(0.5),  # horizontally flip 100% of the images
    iaa.Crop(px=(30, 40)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.4, 1.45))  # blur images with a sigma of 0.5 to 1.75
])


# 113 - Q - glasses; 119 - W 0 sunglasses; 101 - E - background; 32 - SPACE - skip
for i in os.listdir(photoDumpPath):
    img = cv2.imread(photoDumpPath + i, 1)
    # cv2.imshow('image', img)

    img = scipy.misc.np.array(Image.fromarray(img).resize([num_px, num_px]))

    train_image_x_set.append(np.array(img))

    aug_one_img = augumenter_one.augment_image(img)
    train_image_x_set.append(np.array(aug_one_img))
    aug_two_img = augumenter_two.augment_image(img)
    train_image_x_set.append(np.array(aug_two_img))

    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 113:
            y_class = select_class("glasses")
            for j in range(0, num_aug + 1):
                y_set.append(y_class)
            print("Shape of x_set now: " + str(np.array(train_image_x_set).shape))
            print("Shape of y_set now: " + str(np.array(y_set).shape))
            break
        elif k == 119:
            y_class = select_class("sunglasses")
            for j in range(0, num_aug + 1):
                y_set.append(y_class)
            print("Shape of x_set now: " + str(np.array(train_image_x_set).shape))
            print("Shape of y_set now: " + str(np.array(y_set).shape))
            break
        elif k == 101 or k == 32:
            y_class = select_class("background")
            for j in range(0, num_aug + 1):
                y_set.append(y_class)
            print("Shape of x_set now: " + str(np.array(train_image_x_set).shape))
            print("Shape of y_set now: " + str(np.array(y_set).shape))
            break
        else:
            y_class = select_class("background")
            for j in range(0, num_aug + 1):
                y_set.append(y_class)
            print("Shape of x_set now: " + str(np.array(train_image_x_set).shape))
            print("Shape of y_set now: " + str(np.array(y_set).shape))
            break

cv2.destroyAllWindows()
train_image_x_set = np.array(train_image_x_set)
y_set = np.array(y_set)

train_set = {
   "X_train": train_image_x_set,
   "Y_train": y_set
}

with h5py.File('data_test.hdf5', 'w') as f:
    d_set = f.create_dataset("X_train", data=train_set["X_train"])
    d_set = f.create_dataset("Y_train", data=train_set["Y_train"])

#"""#OPEN IMAGE TEST
#
#train_dataset = h5py.File('data_test.hdf5', "r")
#train_set_x_orig = np.array(train_dataset["X_train"][:])  # train set features
#train_set_y = np.array(train_dataset["Y_train"][:])  # train set features
#
#print(str(train_set_x_orig.shape))
#print(str(train_set_y.shape))
#
#index = 1
#img = train_set_x_orig[index, :]
#while True:
#    cv2.imshow('image', img)
#    k = cv2.waitKey(0)
#    if k == 113:
#        print("good")
#        break
#"""