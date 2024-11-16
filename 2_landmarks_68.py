import cv2
import dlib
import pandas as pd

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")

# Path to the input video
video_path = r"C:\Users\bokch\PyCharm\Ergonomics\videos\front\bad\1001.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error opening video file at {video_path}")
    exit()

# Initialize a list to store the coordinates
landmark_coordinates = []

# Process each frame in the video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Initialize a dictionary to store the coordinates for the current frame
    frame_landmarks = {'Frame': frame_count}

    # Draw lines for facial landmarks if faces are detected
    if faces:
        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(gray, face)

            # Get the coordinates of all 68 facial landmarks
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                frame_landmarks[f'Landmark_{i}'] = (x, y)

    # Append the coordinates to the list
    landmark_coordinates.append(frame_landmarks)
    frame_count += 1

# Release the video capture
cap.release()

# Convert the list of coordinates to a DataFrame
df = pd.DataFrame(landmark_coordinates)

# Save the DataFrame to a CSV file
csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\front\bad\1001_68.csv"
df.to_csv(csv_path, index=False)

print(f"Landmark coordinates saved to {csv_path}")
