import tensorflow as tf
import numpy as np
import os
import cv2 as cv

IMAGE_SIZE = 128

checkpoint_path = "assets/model/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

class_to_vector_map = {
    "Omar":      np.array([1, 0, 0, 0, 0, 0]),
    "Christian": np.array([0, 1, 0, 0, 0, 0]),
    "Ethan":     np.array([0, 0, 1, 0, 0, 0]),
    "Noah":      np.array([0, 0, 0, 1, 0, 0]),
    "Sammy":     np.array([0, 0, 0, 0, 1, 0]),
    "Unknown":   np.array([0, 0, 0, 0, 0, 1]),
}

index_to_class_map = ["Omar", "Christian", "Ethan", "Noah", "Sammy", "Unknown"]

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE**2,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model, training_set, classes, epochs=500):
    class_vectors = np.array([class_to_vector_map[cs] for cs in classes])
    model.fit(training_set.T, class_vectors, epochs=epochs, callbacks=[cp_callback])

def load_model(training_set, target_classes):
    model = create_model()
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    else:
        train_model(model, training_set, target_classes)
    return model


def test_model(model, testing_set):
    # ✅ 🎯 MAKE PREDICTIONS
    predictions = model.predict(testing_set.T)  # shape: (num_test_samples, num_classes)

    # ✅ 🎯 GET PREDICTED CLASS INDICES
    predicted_indices = np.argmax(predictions, axis=1)

    # ✅ 🎯 CONVERT INDICES TO CLASS LABELS
    predicted_labels = [index_to_class_map[idx] for idx in predicted_indices]

    # ✅ 🎯 VISUALIZATION (Optional)
    for img, label in zip(testing_set.T, predicted_labels):
        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE)
        bgr_image = cv.cvtColor((img * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
        cv.putText(bgr_image, text=label, org=(0, 20),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                   color=(0, 0, 255), thickness=2)
        cv.imshow("Test Image", bgr_image)
        cv.waitKey(0)

    cv.destroyAllWindows()

    return predicted_labels

