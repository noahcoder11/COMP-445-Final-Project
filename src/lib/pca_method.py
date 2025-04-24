import cv2
import numpy as np
import json
import os

def load_config(training_set, MAX_PRINCIPLE_COMPONENTS):
    config = {}

    path = os.path.join(os.path.dirname(__file__), '../pca_config.json')

    try:
        with open(path, 'r') as f:
            try:
                config = json.load(f)
                config['training_mean'] = np.array(config['training_mean'])
                config['pca_projection_matrix'] = np.array(config['pca_projection_matrix']).reshape(config['pca_projection_shape'])
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in pca_config.json")
                os.remove(path)
                exit(1)
            f.close()
        return config
    except FileNotFoundError:
        with open(path, 'w') as f:
            # Calculate mean of training set:
            training_mean = compute_training_mean(training_set)
            # Save the mean to the config:
            config['training_mean'] = training_mean.tolist()
            # Compute PCA projection matrix:
            pca_projection_matrix = compute_pca_projection_matrix(training_set, training_mean, MAX_PRINCIPLE_COMPONENTS)
            # Save the PCA projection matrix to the config:
            config['pca_projection_matrix'] = pca_projection_matrix.tolist()
            config['pca_projection_shape']  = pca_projection_matrix.shape

            json.dump(config, f)
            f.close()

            config['training_mean'] = np.array(config['training_mean'])
            config['pca_projection_matrix'] = np.array(config['pca_projection_matrix']).reshape(config['pca_projection_shape'])

        return config

def project_images(image_set, config):
    centered_images = np.array([i - config['training_mean'] for i in image_set.T]).T
    projected = config['pca_projection_matrix'].T @ centered_images

    return projected

def compute_eigenfaces(training_set, testing_set, config):
    train_eigenfaces = project_images(training_set, config)
    test_eigenfaces = project_images(testing_set, config)
    return train_eigenfaces, test_eigenfaces

def calculate_euclidean_distance(train_eig, test_eig):
    return np.linalg.norm(train_eig - test_eig)

def recognize_faces(training_set, testing_set, config):
    print("Computing eigenfaces...")
    train_eigenfaces, test_eigenfaces = compute_eigenfaces(training_set, testing_set, config)
    print("Done.")

    print("Recognizing faces...")

    # Calculate the Euclidean distance between the training and testing eigenfaces:
    distances = []

    for test_eig in test_eigenfaces.T:
        dist_array = []
        for train_eig in train_eigenfaces.T:
            dist_array.append(calculate_euclidean_distance(train_eig, test_eig))
        distances.append(dist_array)

    # Find the index of the minimum distance:
    min_index = np.argmin(np.array(distances), axis=1)

    print("Done.")

    return min_index

def compute_training_mean(training_set):
    print("Computing training mean...")

    # Compute the mean image:
    mean_image = np.mean(training_set, axis=1)

    print("Done.")

    return mean_image

def compute_pca_projection_matrix(training_set, mean_image, numPCAs):
    print("Computing PCA projection matrix...")

    # Center the images:
    centered_images = np.array([i - mean_image for i in training_set.T]).T

    # Indirectly compute eigenvectors for speed
    reverse_order_covariance = centered_images.T @ centered_images
    reverse_eigenvalues, reverse_eigenvectors = np.linalg.eigh(reverse_order_covariance)

    # Compute the eigenvalues and eigenvectors:
    eigenvectors = centered_images @ reverse_eigenvectors
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    print("Done.")

    return eigenvectors[:, -numPCAs:]