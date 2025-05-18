import cv2 as cv
import numpy as np
import os

def viola_jones(image, image_size):
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    haar_classifiers_path = os.path.join(os.path.dirname(__file__), '../assets/haarcascade_frontalface_default.xml')
    face_cascade = cv.CascadeClassifier(haar_classifiers_path)

    faces = face_cascade.detectMultiScale(
        grayscale_image,
        scaleFactor=1.1,  # smaller step for image scale pyramid
        minNeighbors=3,  # higher value = fewer false positives
        minSize=(30, 30),  # minimum object size to detect
        flags=cv.CASCADE_SCALE_IMAGE
    )

    face_windows = []

    for (x, y, w, h) in faces:
        face_windows.append(cv.resize(np.array(grayscale_image[y:y + h, x:x + w]), (image_size, image_size)))

    return face_windows, faces