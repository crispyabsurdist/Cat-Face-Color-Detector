import cv2
import os
from tqdm import tqdm
import shutil

# Define the target size (64x64 pixels)
TARGET_SIZE = (64, 64)
# Define the pixel intensity threshold to classify faces as black or white
INTENSITY_THRESHOLD = 50  # Adjust this value as needed

def is_black_cat_face(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average pixel intensity (0-255, lower values indicate darker regions)
    avg_intensity = cv2.mean(gray)[0]

    return avg_intensity < INTENSITY_THRESHOLD

def main():
    input_folder = "dataset-part3"  # Replace with your input folder path

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

    image_files = [filename for filename in os.listdir(input_folder) if filename.endswith(('.jpg', '.jpeg', '.png'))]

    with tqdm(total=len(image_files)) as pbar:
        for filename in image_files:
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Resize the image to the target size
            image = cv2.resize(image, TARGET_SIZE)

            # Detect cat faces in the resized image
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):
                    # Crop the face region
                    face = image[y:y+h, x:x+w]

                    if is_black_cat_face(face):
                        dest_folder = 'black'
                    else:
                        dest_folder = 'white'

                    # Append face index to filename if multiple faces are detected
                    if len(faces) > 1:
                        dest_filename = f"{os.path.splitext(filename)[0]}_face{i+1}{os.path.splitext(filename)[1]}"
                    else:
                        dest_filename = filename

                    shutil.move(image_path, os.path.join(dest_folder, dest_filename))

            pbar.update(1)
            pbar.set_description(f"Processed: {filename}")

if __name__ == "__main__":
    main()
