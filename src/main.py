import cv2 as cv
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from lib.data_formatting import load_training_set, load_testing_set
from lib.pca_method import load_config, recognize_faces
from lib.neural import initialize_model, test_model

# Hyperparameters
IMAGE_SIZE = 128
MAX_PRINCIPLE_COMPONENTS = 50

#format_dataset(IMAGE_SIZE)

training_set = load_training_set()
testing_set = load_testing_set()

def runPCA(testing_set, training_set):
    config = load_config(training_set, MAX_PRINCIPLE_COMPONENTS)

    mean_image = config['training_mean'].reshape(IMAGE_SIZE, IMAGE_SIZE)
    cv.imshow("Mean Image", mean_image)

    indices = recognize_faces(training_set, testing_set, config)

    for i, j in zip(testing_set.T, indices):
        print(i.shape)
        cv.imshow("Test Image", i.reshape(IMAGE_SIZE, IMAGE_SIZE))
        cv.imshow("Matched Image", training_set[:,j].reshape(IMAGE_SIZE, IMAGE_SIZE))
        cv.waitKey(0)

    cv.destroyAllWindows()

def runModel(testing_set, training_set):
    config = load_config(training_set, MAX_PRINCIPLE_COMPONENTS)

    model = initialize_model(testing_set, training_set, config)

if __name__ == "__main__":
    # runPCA(testing_set, training_set)
    runModel(testing_set, training_set)