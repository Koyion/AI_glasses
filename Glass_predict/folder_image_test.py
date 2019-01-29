import cv2 as cv2
import os
import sys
import numpy as np
import tensorflow as tf

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
y_predict = np.zeros((1, 3))

TestPhotos = sys.argv[1]
resultPath = sys.argv[2]

image_size = 256
num_channels = 3
for i in os.listdir(TestPhotos):
    image = []
    img = cv2.imread(TestPhotos + i, 1)
    tmp_img = cv2.resize(img, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    image.append(tmp_img)
    image = np.array(image, dtype=np.uint8)
    image = image.astype('float32')
    image = np.multiply(image, 1.0 / 255.0)
    X_img = image.reshape(1, image_size, image_size, num_channels)
    # Calculate y_pred
    feed_dict_testing = {x: X_img, y_true: y_predict}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # Result shows probability of ["glasses", "sunglasees", "person without glasses, or maybe no person at all"]
    # print(result)
    # print(np.sum(result))  # sum must be ~1
    result_class = np.argmax(result, 1)
    if result_class == 0:
        result_class = "Glasses"
        print("GLASSES ON PERSON: " + str(i))
    elif result_class == 1:
        result_class = "Sunglasses"
        print("SUNGLASSES ON PERSON: " + str(i))
    else:
        result_class = "Person or background"
        print("NO GLASSES OR SUNGLASSES ON PERSON: " + str(i))
    f = open(resultPath, "a")
    f.write(i + ": " + result_class + " - " + str(result) + "\n")
    f.close()
