import cv2 as cv2
import os
import sys
import numpy as np
import h5py
import scipy.misc
from PIL import Image
from imgaug import augmenters as iaa
from Class_selcet import image_class

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
image_counter = 0

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

# 113 - Q - glasses; 119 - W 0 sunglasses; 101 - E - background; 32 - SPACE - skip
for i in os.listdir(photoDumpPath):
    # Open image as RGB array
    img = cv2.imread(photoDumpPath + i, 1)
    image_counter += 1
    print("Image number: " + str(image_counter) + "/2600")
    # Reshaping for 4 datasets
    tmp_img_0 = scipy.misc.np.array(Image.fromarray(img).resize([num_px[0], num_px[0]]))
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
    # First agumenters
    first_aug_image_0 = augumenter_one_0.augment_image(tmp_img_0)
    # Append augumented
    train_image_x_set_0.append(np.array(first_aug_image_0))
    # Second augumenter
    second_aug_image_0 = augumenter_two_0.augment_image(tmp_img_0)
    # Append augumented
    train_image_x_set_0.append(np.array(second_aug_image_0))
    # Choose image class


cv2.destroyAllWindows()
# Reformatting
train_image_x_set_0 = np.array(train_image_x_set_0)
y_set = np.array(y_set)

# Print shapes
print("Shape of x_set_0 now: " + str(train_image_x_set_0.shape))
print("Shape of y_set now: " + str(y_set.shape))

# Print total guesses
counter = y_set.sum(axis=0)
print("Glasses in set: " + str(counter[0][0]))
print("Sunglasses in set: " + str(counter[1][0]))
print("Backgrounds in set: " + str(counter[2][0]))
# Preprocess for h5
train_set = {
    "X_train_64": train_image_x_set_0
    "Y_train": y_set
}
# Generate hp files
with h5py.File('train_set_64.hdf5', 'w') as f:
    d_set = f.create_dataset("X_train", data=train_set["X_train_64"])
    d_set = f.create_dataset("Y_train", data=train_set["Y_train"])

