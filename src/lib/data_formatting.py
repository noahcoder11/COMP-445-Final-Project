import numpy as np
import cv2 as cv
from lib.viola_jones import viola_jones
import os

def load_training_set():
    training_set = []
    directory = os.path.join(os.path.dirname(__file__), '../assets/images/training/')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image = cv.imread(os.path.join(directory, filename), cv.IMREAD_GRAYSCALE)
            training_set.append(image)

    flattened_images = np.array([image.flatten() for image in training_set])
    normalized_images = flattened_images / 255.0

    return normalized_images.T

def load_testing_set():
    testing_set = []
    directory = os.path.join(os.path.dirname(__file__), '../assets/images/testing/')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image = cv.imread(os.path.join(directory, filename), cv.IMREAD_GRAYSCALE)
            testing_set.append(image)

    flattened_images = np.array([image.flatten() for image in testing_set])
    normalized_images = flattened_images / 255.0

    return normalized_images.T

def read_image_into_set(file_name, image_size):
    file_path = os.path.join(os.path.dirname(__file__), '../assets/images/originals/' + file_name)
    new_file_path = os.path.join(os.path.dirname(__file__), '../assets/images/training/')
 
    original = cv.imread(file_path)

    face_windows = viola_jones(original, image_size)

    index = 0
    for win in face_windows:
        cv.imwrite(f"{new_file_path}{index}{file_name}", win)
        index += 1

def format_dataset():
    directory = os.path.join(os.path.dirname(__file__), '../assets/images/originals/')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            read_image_into_set(filename)