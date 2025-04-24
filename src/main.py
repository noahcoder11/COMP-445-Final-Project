import cv2 as cv
from lib.data_formatting import format_dataset, load_training_set, load_testing_set
from lib.pca_method import load_config, recognize_faces

# Hyperparameters
IMAGE_SIZE = 128
MAX_PRINCIPLE_COMPONENTS = 100

#format_dataset(IMAGE_SIZE)

training_set = load_training_set()
testing_set = load_testing_set()

config = load_config(training_set, MAX_PRINCIPLE_COMPONENTS)

training_mean = config['training_mean']
pca_projection_matrix = config['pca_projection_matrix'].reshape(config['pca_projection_shape'])

indices = recognize_faces(training_set, testing_set, config)
print(indices)

for i, j in zip(testing_set.T, indices):
    print(i.shape)
    cv.imshow("Test Image", i.reshape(IMAGE_SIZE, IMAGE_SIZE))
    cv.imshow("Matched Image", training_set[:,j].reshape(IMAGE_SIZE, IMAGE_SIZE))
    cv.waitKey(0)

cv.destroyAllWindows()