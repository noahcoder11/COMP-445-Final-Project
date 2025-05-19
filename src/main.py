import cv2 as cv
import numpy as np
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from lib.data_formatting import load_training_set, load_testing_set
from lib.pca_method import load_config, recognize_faces
from lib.neural import load_model, test_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Hyperparameters
IMAGE_SIZE = 128
MAX_PRINCIPLE_COMPONENTS = 50

training = load_training_set()
testing = load_testing_set()

def testPCA(testing, training):
    testing_set, test_classes = testing
    training_set, train_classes = training
    config = load_config(training_set, MAX_PRINCIPLE_COMPONENTS)

    mean_image = config['training_mean'].reshape(IMAGE_SIZE, IMAGE_SIZE)
    cv.imshow("Mean Image", mean_image)

    predicted_indices = recognize_faces(training_set, testing_set, config)
    predicted_labels = [train_classes[i] for i in predicted_indices]

    for i, j in zip(testing_set.T, predicted_indices):
        print(i.shape)
        img = i.reshape(IMAGE_SIZE, IMAGE_SIZE)
        bgr_image = cv.cvtColor((img * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
        cv.putText(bgr_image, text=train_classes[j], org=(0, 20), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
        cv.imshow("Test Image", bgr_image)
        cv.imshow("Matched Image", training_set[:,j].reshape(IMAGE_SIZE, IMAGE_SIZE))
        cv.waitKey(0)

    cv.destroyAllWindows()

    cm = confusion_matrix(test_classes, predicted_labels, labels=["Omar", "Christian", "Ethan", "Noah", "Sammy", "Unknown"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Omar", "Christian", "Ethan", "Noah", "Sammy", "Unknown"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("PCA Confusion Matrix")
    plt.show()

def testModel(testing, training):
    testing_set, test_classes = testing
    training_set, train_classes = training
    model = load_model(training_set, train_classes)
    predicted_labels = test_model(model, testing_set)

    cm = confusion_matrix(test_classes, predicted_labels, labels=["Omar", "Christian", "Ethan", "Noah", "Sammy", "Unknown"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Omar", "Christian", "Ethan", "Noah", "Sammy", "Unknown"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Neural Network Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    #testPCA(testing, training)
    testModel(testing, training)