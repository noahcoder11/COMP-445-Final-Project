import cv2
import os

def preprocess_and_save_images(input_folder, output_folder, image_size=(128, 128)):
    """
    Reads all images from input_folder, converts to grayscale,
    center-crops them to square, resizes to image_size,
    saves to output_folder, and deletes the original.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            img = cv2.imread(input_path)

            if img is None:
                print(f"⚠️ Could not read {filename}, skipping.")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            crop_size = min(h, w)

            # Compute center crop coordinates
            start_x = w // 2 - crop_size // 2
            start_y = h // 2 - crop_size // 2
            cropped = gray[start_y:start_y+crop_size, start_x:start_x+crop_size]

            # Resize to target size
            processed = cv2.resize(cropped, image_size)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed)
            print(f"✅ Saved {output_path}")

            os.remove(input_path)
            print(f"🗑️ Deleted original: {input_path}")

if __name__ == "__main__":
    input_folder = r"C:\Users\dream\OneDrive\Desktop\python3.12\COMP-445-Final-Project\assets\images\selfies_to_be_processed"
    output_folder = r"C:\Users\dream\OneDrive\Desktop\python3.12\COMP-445-Final-Project\assets\images\images_of_christian"

    preprocess_and_save_images(input_folder, output_folder)