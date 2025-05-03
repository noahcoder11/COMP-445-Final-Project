import tensorflow as tf
import numpy as np
from lib.pca_method import compute_eigenfaces

def initialize_model(testing_set, training_set, config):
  train_vectors, test_vectors = compute_eigenfaces(training_set, testing_set, config)
  y_test = np.zeros(3, len(training_set))

  # Transpose to get shape (samples, features)
  train_vectors = (train_vectors / 255.0).T  # Now (197, 50)
  test_vectors = (test_vectors / 255.0).T    # Now (197, 50) or whatever test size

  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),  # 50 features
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(197, activation='softmax')  # 197 output classes
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # One-hot labels for 197 samples
  y_train = np.eye(197)  # shape (197, 197)

  print(f"train_vectors shape: {train_vectors.shape}, y_train shape: {y_train.shape}")
  model.fit(train_vectors, y_train, epochs=500)
  model.evaluate(test_vectors, y_train)

  return model

def test_model(model, testing_set, config):
  test_vectors = (testing_set / 255.0).T  # Transpose test data too
  return model.predict(test_vectors)
