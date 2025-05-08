import tensorflow as tf
import numpy as np
from lib.pca_method import compute_eigenfaces
import cv2 as cv
from lib.data_formatting import load_training_set, load_testing_set

IMAGE_SIZE = 128

def initialize_model(testing_set, training_set, classes):
    mapping = {
        "Omar":      np.array([1, 0, 0, 0, 0]),
        "Christian": np.array([0, 1, 0, 0, 0]),
        "Ethan":     np.array([0, 0, 1, 0, 0]),
        "Noah":      np.array([0, 0, 0, 1, 0]),
        "Unknown":   np.array([0, 0, 0, 0, 1]),
    }
    mapping2 = ["Omar", "Christian", "Ethan", "Noah", "Unknown"]

    print(training_set.shape)

    class_mapping = np.array([mapping[cs] for cs in classes])
    print(class_mapping.shape)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(128**2,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(training_set.T, class_mapping, epochs=200)

    # ✅ 🎯 MAKE PREDICTIONS
    predictions = model.predict(testing_set.T)  # shape: (num_test_samples, num_samples)
    
    # ✅ 🎯 GET PREDICTED CLASS INDICES
    predicted_classes = np.argmax(predictions, axis=1)

    for img, pred in zip(testing_set.T, predicted_classes):
        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE)
        bgr_image = cv.cvtColor((img * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
        cv.putText(bgr_image, text=mapping2[pred], org=(0, 20), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
        cv.imshow("Test Image", bgr_image)
        cv.waitKey(0)
    
    return model

def test_model(model, testing_set, config):
  test_vectors = (testing_set / 255.0).T  # Transpose test data too
  return model.predict(test_vectors)
