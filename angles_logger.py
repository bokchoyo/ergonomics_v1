import cv2
import dlib
import mediapipe as mp
import pandas as pd
import math

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

# Initialize a list to store the angles
angles = []

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

    # Initialize variables to store the coordinates of the eyes, ears, and shoulders
    left_eye = right_eye = left_ear = right_ear = None
    left_shoulder = right_shoulder = None

    # Draw lines for facial landmarks if faces are detected
    if faces:
        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(gray, face)

            # Get the coordinates of the eyes
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)

            # Get the coordinates of the ears
            left_ear = (landmarks.part(0).x, landmarks.part(0).y)  # LEFT_EAR
            right_ear = (landmarks.part(16).x, landmarks.part(16).y)  # RIGHT_EAR

    # Draw lines for pose landmarks if pose is detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates of the shoulders
        frame_height, frame_width, _ = frame.shape
        left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
        right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)

    # Calculate the angle between the eye line and the shoulder line
    if left_eye and right_eye and left_shoulder and right_shoulder:
        # Calculate the slopes of the eye line and the shoulder line
        eye_slope = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
        shoulder_slope = (right_shoulder[1] - left_shoulder[1]) / (right_shoulder[0] - left_shoulder[0])

        # Calculate the angles of the lines with respect to the horizontal
        eye_angle = math.degrees(math.atan(eye_slope))
        shoulder_angle = math.degrees(math.atan(shoulder_slope))

        # Calculate the angle between the eye line and the shoulder line
        eye_shoulder_angle = shoulder_angle - eye_angle

        # Adjust angle to be within the range -90 to 90
        if eye_shoulder_angle > 90:
            eye_shoulder_angle -= 180
        elif eye_shoulder_angle < -90:
            eye_shoulder_angle += 180
    else:
        eye_shoulder_angle = None

    # Calculate the angle between the shoulder line and the ear line
    if left_ear and right_ear and left_shoulder and right_shoulder:
        # Calculate the slope of the ear line
        ear_slope = (right_ear[1] - left_ear[1]) / (right_ear[0] - left_ear[0])

        # Calculate the angle of the ear line with respect to the horizontal
        ear_angle = math.degrees(math.atan(ear_slope))

        # Calculate the angle between the shoulder line and the ear line
        shoulder_ear_angle = shoulder_angle - ear_angle

        # Adjust angle to be within the range -90 to 90
        if shoulder_ear_angle > 90:
            shoulder_ear_angle -= 180
        elif shoulder_ear_angle < -90:
            shoulder_ear_angle += 180
    else:
        shoulder_ear_angle = None

    print(f"Frame: {frame_count}, Eye-Shoulder Angle: {eye_shoulder_angle}, Shoulder-Ear Angle: {shoulder_ear_angle}")

    # Append the angles to the list
    angles.append(
        {'Frame': frame_count, 'Eye-Shoulder Angle': eye_shoulder_angle, 'Shoulder-Ear Angle': shoulder_ear_angle})
    frame_count += 1

# Release the video capture and pose objects
cap.release()
pose.close()

# Convert the list of angles to a DataFrame
df = pd.DataFrame(angles)

# Save the DataFrame to a CSV file
csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\angles.csv"
df.to_csv(csv_path, index=False)

print(f"Angles saved to {csv_path}")
