import math

import cv2
import dlib
import os

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")

# Path to the input image
# image_path = r"C:\Users\bokch\PyCharm\Ergonomics\pictures\frames\0999_front\frame_1082.jpg"
image_path = r"C:\Users\bokch\PyCharm\Ergonomics\pictures\frames\0999_front\frame_2917.jpg"
# Directory to save the image with landmarks
output_folder = r"C:\Users\bokch\PyCharm\Ergonomics\pictures\final"
os.makedirs(output_folder, exist_ok=True)

# Read the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error loading image file at {image_path}")
    exit()

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Draw circles for facial landmarks if faces are detected
if faces:
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Draw a green circle for each landmark point
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        x1, y1 = landmarks.part(0).x, landmarks.part(0).y
        x17, y17 = landmarks.part(16).x, landmarks.part(16).y
        distance = math.sqrt((x17 - x1) ** 2 + (y17 - y1) ** 2)
        print(f"Distance between landmarks 1 and 17: {distance:.2f}")


output_path = os.path.join(output_folder, "frame_29187.jpg")
cv2.imwrite(output_path, image)

print(f"Image with landmark points saved to {output_path}")
