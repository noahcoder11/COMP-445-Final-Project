import cv2 as cv
import numpy as np
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from lib.data_formatting import load_training_set, load_testing_set
from lib.pca_method import load_config, recognize_faces
from lib.neural import initialize_model

# Hyperparameters
IMAGE_SIZE = 128
MAX_PRINCIPLE_COMPONENTS = 50

training_set, target_classes = load_training_set()
testing_set = load_testing_set()

print(target_classes)

def runPCA(testing_set, training_set, classes):
    config = load_config(training_set, MAX_PRINCIPLE_COMPONENTS)

    mean_image = config['training_mean'].reshape(IMAGE_SIZE, IMAGE_SIZE)
    cv.imshow("Mean Image", mean_image)

    index = recognize_faces(training_set, testing_set, config)

    for i, j in zip(testing_set.T, index):
        print(i.shape)
        img = i.reshape(IMAGE_SIZE, IMAGE_SIZE)
        bgr_image = cv.cvtColor((img * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
        cv.putText(bgr_image, text=classes[j], org=(0, 20), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
        cv.imshow("Test Image", bgr_image)
        cv.imshow("Matched Image", training_set[:,j].reshape(IMAGE_SIZE, IMAGE_SIZE))
        cv.waitKey(0)

    cv.destroyAllWindows()

def runModel(testing_set, training_set, classes):
    config = load_config(training_set, MAX_PRINCIPLE_COMPONENTS)

    model = initialize_model(testing_set, training_set, classes)

if __name__ == "__main__":
    runPCA(testing_set, training_set, target_classes)
    #runModel(testing_set, training_set, target_classes)