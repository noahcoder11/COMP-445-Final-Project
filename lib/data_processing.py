import os
import cv2 as cv
from viola_jones import viola_jones

def read_image_into_set(file_name, image_size):
    file_path = os.path.join(os.path.dirname(__file__), '../assets/images/originals/' + file_name)
    new_file_path = os.path.join(os.path.dirname(__file__), '../assets/images/training/')

    original = cv.imread(file_path)

    face_windows = viola_jones(original, image_size)[0]
    index = 0
    for win in face_windows:
        cv.imwrite(f"{new_file_path}{index}{file_name}", win)
        index += 1

def preprocess_and_save_images(input_folder, output_folder, image_size=128):
    """
    Reads all images from input_folder, converts to grayscale,
    center-crops them to square, resizes to image_size,
    saves to output_folder, and deletes the original.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    i = 0

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            img = cv.imread(input_path)

            if img is None:
                print(f"⚠️ Could not read {filename}, skipping.")
                continue

            processed = viola_jones(img, image_size)[0]
            output_path = os.path.join(output_folder)
            j = 0

            for img in processed:
                cv.imwrite(f"{output_path}/Sammy{i}{j}.png", img)
                j += 1

            os.remove(input_path)
            print(f"🗑️ Deleted original: {input_path}")
            i += 1


if __name__ == "__main__":
    path = os.getcwd()
    input_folder = r"C:\Users\dream\OneDrive\Desktop\python3.12\COMP-445-Final-Project\assets\images\originals"
    output_folder = r"C:\Users\dream\OneDrive\Desktop\python3.12\COMP-445-Final-Project\assets\images\training\Sammy"

    preprocess_and_save_images(input_folder, output_folder)