import cv2 as cv2
import os
import sys
import numpy as np
import h5py
import scipy.misc
from PIL import Image
from imgaug import augmenters as iaa
from Class_selcet import image_class, transcript_class

# TRAIN SET COMPLETING
photoDumpPath = sys.argv[1]  # PATH WITH IMAGE FOLDER
train_image_x_set_0 = []
train_image_x_set_1 = []
train_image_x_set_2 = []
train_image_x_set_3 = []
y_set = []
num_px = [64, 128, 256, 512]
tmp_img = []
num_aug = 2

# ---------------------------------------------------------------------------------------------
augumenter_one_0 = iaa.Sequential([
    iaa.Fliplr(1.),  # horizontally flip 100% of the images
    iaa.Crop(px=(2, 6)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.5, 1.75))  # blur images with a sigma of 0.5 to 1.75
])

augumenter_two_0 = iaa.Sequential([
    iaa.Add((-10, 10), per_channel=True),  # change brightness of images (by -10 to 10 of original value)
    iaa.Fliplr(0.5),  # horizontally flip 100% of the images
    iaa.Crop(px=(3, 7)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.4, 1.45))  # blur images with a sigma of 0.5 to 1.75
])
# ---------------------------------------------------------------------------------------------
augumenter_one_1 = iaa.Sequential([
    iaa.Fliplr(1.),  # horizontally flip 100% of the images
    iaa.Crop(px=(6, 15)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.5, 1.75))  # blur images with a sigma of 0.5 to 1.75
])

augumenter_two_1 = iaa.Sequential([
    iaa.Add((-10, 10), per_channel=True),  # change brightness of images (by -10 to 10 of original value)
    iaa.Fliplr(0.5),  # horizontally flip 100% of the images
    iaa.Crop(px=(7, 17)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.4, 1.45))  # blur images with a sigma of 0.5 to 1.75
])
# ---------------------------------------------------------------------------------------------
augumenter_one_2 = iaa.Sequential([
    iaa.Fliplr(1.),  # horizontally flip 100% of the images
    iaa.Crop(px=(23, 32)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.5, 1.75))  # blur images with a sigma of 0.5 to 1.75
])

augumenter_two_2 = iaa.Sequential([
    iaa.Add((-10, 10), per_channel=True),  # change brightness of images (by -10 to 10 of original value)
    iaa.Fliplr(0.5),  # horizontally flip 100% of the images
    iaa.Crop(px=(20, 30)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.4, 1.45))  # blur images with a sigma of 0.5 to 1.75
])
# ---------------------------------------------------------------------------------------------
augumenter_one_3 = iaa.Sequential([
    iaa.Fliplr(1.),  # horizontally flip 100% of the images
    iaa.Crop(px=(40, 55)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.5, 1.75))  # blur images with a sigma of 0.5 to 1.75
])

augumenter_two_3 = iaa.Sequential([
    iaa.Add((-10, 10), per_channel=True),  # change brightness of images (by -10 to 10 of original value)
    iaa.Fliplr(0.5),  # horizontally flip 100% of the images
    iaa.Crop(px=(35, 45)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.GaussianBlur(sigma=(0.4, 1.45))  # blur images with a sigma of 0.5 to 1.75
])
# ---------------------------------------------------------------------------------------------

# 113 - Q - glasses; 119 - W 0 sunglasses; 101 - E - background; 32 - SPACE - skip
for i in os.listdir(photoDumpPath):
    # Open image as RGB array
    img = cv2.imread(photoDumpPath + i, 1)
    # Reshaping for 4 datasets
    tmp_img_0 = scipy.misc.np.array(Image.fromarray(img).resize([num_px[0], num_px[0]]))
    tmp_img_1 = scipy.misc.np.array(Image.fromarray(img).resize([num_px[1], num_px[1]]))
    tmp_img_2 = scipy.misc.np.array(Image.fromarray(img).resize([num_px[2], num_px[2]]))
    tmp_img_3 = scipy.misc.np.array(Image.fromarray(img).resize([num_px[3], num_px[3]]))
    y_class = image_class(tmp_img_3)
    # IF SKIPPING IMAGE
    if (y_class == False).all():
        print("Skippinmg")
        continue
    # Append to Y data set
    for j in range(0, num_aug + 1):
        y_set.append(y_class)
    # Appending in 4 datasets
    train_image_x_set_0.append(np.array(tmp_img_0))
    train_image_x_set_1.append(np.array(tmp_img_1))
    train_image_x_set_2.append(np.array(tmp_img_2))
    train_image_x_set_3.append(np.array(tmp_img_3))
    # First agumenters
    first_aug_image_0 = augumenter_one_0.augment_image(tmp_img_0)
    first_aug_image_1 = augumenter_one_1.augment_image(tmp_img_1)
    first_aug_image_2 = augumenter_one_2.augment_image(tmp_img_2)
    first_aug_image_3 = augumenter_one_3.augment_image(tmp_img_3)
    # Append augumented
    train_image_x_set_0.append(np.array(first_aug_image_0))
    train_image_x_set_1.append(np.array(first_aug_image_1))
    train_image_x_set_2.append(np.array(first_aug_image_2))
    train_image_x_set_3.append(np.array(first_aug_image_3))
    # Second augumenter
    second_aug_image_0 = augumenter_two_0.augment_image(tmp_img_0)
    second_aug_image_1 = augumenter_two_1.augment_image(tmp_img_1)
    second_aug_image_2 = augumenter_two_2.augment_image(tmp_img_2)
    second_aug_image_3 = augumenter_two_3.augment_image(tmp_img_3)
    # Append augumented
    train_image_x_set_0.append(np.array(second_aug_image_0))
    train_image_x_set_1.append(np.array(second_aug_image_1))
    train_image_x_set_2.append(np.array(second_aug_image_2))
    train_image_x_set_3.append(np.array(second_aug_image_3))
    # Choose image class


cv2.destroyAllWindows()
# Reformatting
train_image_x_set_0 = np.array(train_image_x_set_0)
train_image_x_set_1 = np.array(train_image_x_set_1)
train_image_x_set_2 = np.array(train_image_x_set_2)
train_image_x_set_3 = np.array(train_image_x_set_3)
y_set = np.array(y_set)

# Print shapes
print("Shape of x_set_0 now: " + str(train_image_x_set_0.shape))
print("Shape of x_set_1 now: " + str(train_image_x_set_1.shape))
print("Shape of x_set_2 now: " + str(train_image_x_set_2.shape))
print("Shape of x_set_3 now: " + str(train_image_x_set_3.shape))
print("Shape of y_set now: " + str(y_set.shape))

# Print total guesses
counter = y_set.sum(axis=0)
print("Glasses in set: " + str(counter[0][0]))
print("Sunglasses in set: " + str(counter[1][0]))
print("Backgrounds in set: " + str(counter[2][0]))
# Preprocess for h5
train_set = {
    "X_train_64": train_image_x_set_0,
    "X_train_128": train_image_x_set_1,
    "X_train_256": train_image_x_set_2,
    "X_train_512": train_image_x_set_3,
    "Y_train": y_set
}
# Generate hp files
with h5py.File('train_set_64.hdf5', 'w') as f:
    d_set = f.create_dataset("X_train", data=train_set["X_train_64"])
    d_set = f.create_dataset("Y_train", data=train_set["Y_train"])

with h5py.File('train_set_128.hdf5', 'w') as f:
    d_set = f.create_dataset("X_train", data=train_set["X_train_128"])
    d_set = f.create_dataset("Y_train", data=train_set["Y_train"])

with h5py.File('train_set_256.hdf5', 'w') as f:
    d_set = f.create_dataset("X_train", data=train_set["X_train_256"])
    d_set = f.create_dataset("Y_train", data=train_set["Y_train"])

with h5py.File('train_set_512.hdf5', 'w') as f:
    d_set = f.create_dataset("X_train", data=train_set["X_train_512"])
    d_set = f.create_dataset("Y_train", data=train_set["Y_train"])

