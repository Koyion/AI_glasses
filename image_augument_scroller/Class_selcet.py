import numpy as np
import cv2 as cv2


def select_class(selected):
    if selected == "glasses":
        cls = np.array([[1], [0], [0]])
        print("Glasses")
    elif selected == "sunglasses":
        cls = np.array([[0], [1], [0]])
        print("Sunglasses")
    elif selected == "background":
        cls = np.array([[0], [0], [1]])  # background
        print("Background")
    else:
        cls = np.array([[0], [0], [0]])
    return cls


def image_class(img):
    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if key == 113:
            y_class = select_class("glasses")
            break
        elif key == 119:
            y_class = select_class("sunglasses")
            break
        elif key == 101 or key == 32:
            y_class = select_class("background")
            break
        else:
            y_class = select_class("skip")
            break

    return y_class


def transcript_class(y_class):
    classes = {
        "glasses": np.array([[1], [0], [0]]),
        "sunglasses": np.array([[0], [1], [0]]),
        "background": np.array([[0], [0], [1]])
    }
    for i in classes:
        if (y_class == classes[str(i)]).all():
            return i
