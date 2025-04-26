import os
import cv2 as cv
from lib.viola_jones import viola_jones

def read_image_into_set(file_name, image_size):
    file_path = os.path.join(os.path.dirname(__file__), '../assets/images/originals/' + file_name)
    new_file_path = os.path.join(os.path.dirname(__file__), '../assets/images/training/')

    original = cv.imread(file_path)

    face_windows = viola_jones(original, image_size)

    index = 0
    for win in face_windows:
        cv.imwrite(f"{new_file_path}{index}{file_name}", win)
        index += 1