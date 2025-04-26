import numpy as np
import cv2 as cv
from lib.data_processing import read_image_into_set
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

def format_dataset(IMAGE_SIZE):
    directory = os.path.join(os.path.dirname(__file__), '../assets/images/originals/')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            read_image_into_set(filename, IMAGE_SIZE)