import cv2
import dlib
import mediapipe as mp
import pandas as pd

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Path to the input video
video_path = r"C:\Users\bokch\Videos\Research\Front\Front_1.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error opening video file at {video_path}")
    exit()

# Initialize a list to store the distances
distances = []

# Process each frame in the video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert the frame to RGB for pose detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Process the frame and find the pose
    results = pose.process(frame_rgb)

    # Initialize variables to store the y-coordinates of the eyes and shoulders
    eyes_y_avg = None
    shoulders_y_avg = None

    # Draw lines for facial landmarks if faces are detected
    if faces:
        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(gray, face)

            # Get the coordinates of the eyes
            left_eye_x = landmarks.part(36).x
            left_eye_y = landmarks.part(36).y
            right_eye_x = landmarks.part(45).x
            right_eye_y = landmarks.part(45).y

            # Calculate the average y-coordinate of the eyes
            eyes_y_avg = (left_eye_y + right_eye_y) / 2

    # Draw lines for pose landmarks if pose is detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates of the shoulders
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Convert normalized coordinates to pixel coordinates
        frame_height, frame_width, _ = frame.shape
        left_shoulder_x = left_shoulder.x * frame_width
        left_shoulder_y = left_shoulder.y * frame_height
        right_shoulder_x = right_shoulder.x * frame_width
        right_shoulder_y = right_shoulder.y * frame_height

        # Calculate the average y-coordinate of the shoulders
        shoulders_y_avg = (left_shoulder_y + right_shoulder_y) / 2

    # Calculate the vertical distance between the lines if both are detected
    if eyes_y_avg is not None and shoulders_y_avg is not None:
        vertical_distance = abs(shoulders_y_avg - eyes_y_avg)
    else:
        vertical_distance = None

    print(vertical_distance)
    # Append the distance to the list
    distances.append({'Frame': frame_count, 'Shoulder-Eye Distance': vertical_distance})
    frame_count += 1

# Release the video capture and pose objects
cap.release()
pose.close()

# Convert the list of distances to a DataFrame
df = pd.DataFrame(distances)

# Save the DataFrame to a CSV file
csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\shoulder-eye_distance.csv"
df.to_csv(csv_path, index=False)

print(f"Distances saved to {csv_path}")
