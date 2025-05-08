import cv2 as cv
import os
import sys
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from lib.viola_jones import viola_jones
from lib.data_formatting import load_training_set
from lib.pca_method import load_config, recognize_faces

training_set, target_classes = load_training_set()
config = load_config(training_set, 50)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection (optional)
    face_windows, face_boxes = viola_jones(frame, 128)

    face_classifications = []

    # Perform face recognition (optional)
    for face_window in face_windows:
        flattened_face = face_window.flatten() / 255.0
        face_classification = recognize_faces(training_set, np.array([flattened_face]), config)[0]
        face_classifications.append(training_set[face_classification])

    for face_box, face_classification in zip(face_boxes, face_classifications):
        print(face_box)
        x, y, w, h = face_box
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(frame, target_classes[face_classification], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv.imshow('Video Feed', frame)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break