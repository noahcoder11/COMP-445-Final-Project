import numpy as np
import cv2 as cv
from lib.data_processing import read_image_into_set
import os

def load_training_set():
    training_set = []
    classes = []
    directory = os.path.join(os.path.dirname(__file__), '../assets/images/training/')

    for subdir in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, subdir)):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                image = cv.imread(os.path.join(directory, subdir, filename), cv.IMREAD_GRAYSCALE)
                training_set.append(image)
                classes.append(subdir)

    flattened_images = np.array([image.flatten() for image in training_set])
    normalized_images = flattened_images / 255.0

    return (normalized_images.T, classes)

def load_testing_set():
    training_set = []
    classes = []
    directory = os.path.join(os.path.dirname(__file__), '../assets/images/testing/')

    for subdir in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, subdir)):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                image = cv.imread(os.path.join(directory, subdir, filename), cv.IMREAD_GRAYSCALE)
                training_set.append(image)
                classes.append(subdir)

    flattened_images = np.array([image.flatten() for image in training_set])
    normalized_images = flattened_images / 255.0

    return (normalized_images.T, classes)

def format_dataset(IMAGE_SIZE):
    directory = os.path.join(os.path.dirname(__file__), '../assets/images/originals/')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            read_image_into_set(filename, IMAGE_SIZE)