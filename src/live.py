import cv2 as cv
import os
import sys
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from lib.viola_jones import viola_jones
from lib.data_formatting import load_training_set
from lib.neural import load_model, index_to_class_map
from PIL import Image, ImageDraw, ImageFont

training_set, target_classes = load_training_set()

cap = cv.VideoCapture(0)

def lerp(a, b, t):
    """
    Linear interpolation between two points a and b.
    """
    return a + (b - a) * t

def dashed_line(img, start, end, color=(255, 0, 0), dash_length=5):
    """
    Draws a dashed line on the image.
    """
    x1, y1 = start
    x2, y2 = end
    length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    for i in range(0, length, dash_length * 3):
        cv.line(img, (int(lerp(x1, x2, i / length)), int(lerp(y1, y2, i / length))), (int(lerp(x1, x2, (i+10) / length)), int(lerp(y1, y2, (i+10) / length))), color, 2)

model = load_model(training_set, target_classes)
face_box = cv.imread("assets/images/gui/admin_focus.png", cv.IMREAD_UNCHANGED)

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
        class_vector = model.predict(np.array([flattened_face]))
        face_classification = index_to_class_map[class_vector.argmax()]
        face_classifications.append(face_classification)

    for face_classification, face_box in zip(face_classifications, face_boxes):
        x, y, w, h = face_box
        dashed_line(frame, (x, y), (x + w, y), color=(0, 255, 255), dash_length=10)
        dashed_line(frame, (x, y), (x, y + h), color=(0, 255, 255), dash_length=10)
        dashed_line(frame, (x + w, y), (x + w, y + h), color=(0, 255, 255), dash_length=10)
        dashed_line(frame, (x, y + h), (x + w, y + h), color=(0, 255, 255), dash_length=10)
        cv.putText(frame, face_classification, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv.imshow('Video Feed', frame)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break