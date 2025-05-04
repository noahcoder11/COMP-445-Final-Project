import tensorflow as tf
import numpy as np
from lib.pca_method import compute_eigenfaces
import cv2 as cv
from lib.data_formatting import load_training_set, load_testing_set

IMAGE_SIZE = 128

def initialize_model(testing_set, training_set, config):
    train_vectors, test_vectors = compute_eigenfaces(training_set, testing_set, config)
    
    num_samples = train_vectors.shape[1]
    print(f"Number of samples: {num_samples}")
    
    y_train = np.eye(num_samples)  # One-hot encoding for training
    print(f"y_train shape: {y_train.shape}")
    
    # Transpose to (samples, features)
    train_vectors = (train_vectors / 255.0).T
    print(f"train_vectors shape: {train_vectors.shape}")
    
    test_vectors = (test_vectors / 255.0).T
    print(f"test_vectors shape: {test_vectors.shape}")
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(train_vectors.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_samples, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_vectors, y_train, epochs=500)

    # ✅ 🎯 MAKE PREDICTIONS
    predictions = model.predict(test_vectors)  # shape: (num_test_samples, num_samples)
    
    # ✅ 🎯 GET PREDICTED CLASS INDICES
    predicted_classes = np.argmax(predictions, axis=1)
    
    # ✅ 🎯 Define expected class indices manually (or load them from somewhere)
    expected_labels = [2, 5]  # update as needed for your test images
    
    print("\n🔍 EVALUATION RESULTS:")
    for i in range(len(predicted_classes)):
        pred = predicted_classes[i]
        expected = expected_labels[i]
        correct = (pred == expected)
        print("predicted_classes:", predicted_classes)
        print("predicted_classes.shape:", predicted_classes.shape)
        print(f"Test sample {i}: predicted class {pred}, expected {expected}, correct? {correct}")
        print("Config keys:", config.keys())
        train_vec = train_vectors[5]
        test_vec = test_vectors[1]
        probs = model.predict(test_vectors)

        diff = np.linalg.norm(train_vec - test_vec)
        print(f"Distance in PCA space between training[5] and test[1]: {diff}")

        #coefficients = train_vectors[pred]  # shape: (50,)
        coefficients = train_vectors[pred] * 255.0
        reconstructed = config['training_mean'] + config['pca_projection_matrix'] @ coefficients
        cv.imshow(f"Predicted Match {i}", reconstructed.reshape(IMAGE_SIZE, IMAGE_SIZE))
        cv.waitKey(0)
    
    return model

def test_model(model, testing_set, config):
  test_vectors = (testing_set / 255.0).T  # Transpose test data too
  return model.predict(test_vectors)
