import tensorflow as tf
import numpy as np
import os, glob
import cv2 as cv2
import sys, argparse


dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
filename = dir_path + '/' + image_path
image_size = 256
num_channels = 3
images = []
image = cv2.imread(filename)
image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
x_batch = images.reshape(1, image_size, image_size, num_channels)

# Restore the model
sess = tf.Session()
# Restore network graph
saver = tf.train.import_meta_graph('datasets/b003la01ep30tra94tea765/glass_model.meta')
# Load weights
saver.restore(sess, tf.train.latest_checkpoint('datasets/b003la01ep30tra94tea765/'))

# Accessing the default graph
graph = tf.get_default_graph()

# Check on predict step of net
y_pred = graph.get_tensor_by_name("y_pred:0")

# Feed the image
x = graph.get_tensor_by_name("X:0")
y_true = graph.get_tensor_by_name("Y:0")
y_test_images = np.zeros((1, 3))

# Calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)
# Result shows probability of ["glasses", "sunglasees", "person without glasses, or maybe no person at all"]
print(result)
print(np.sum(result)) # sum must be ~1
result_class = np.argmax(result, 1)
if result_class == 0:
    print("GLASSES ON PERSON")
elif result_class == 1:
    print("SUNGLASSES ON PERSON")
else:
    print("NO GLASSES OR SUNGLASSES ON PERSON")
