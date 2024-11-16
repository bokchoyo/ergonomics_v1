import cv2
import dlib
import mediapipe as mp
import os

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Path to the input video
video_path = r"C:\Users\bokch\Videos\Research\Front\Front_1.mp4"

# Output folder for saving annotated frames
output_folder = r"C:\Users\bokch\PyCharm\Ergonomics\Pictures\Front"
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error opening video file at {video_path}")
    exit()

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

            # Draw eye lines
            cv2.line(frame, (int(left_eye_x), int(left_eye_y)), (int(right_eye_x), int(right_eye_y)), (0, 255, 0), 2)

    # Draw lines for pose landmarks if pose is detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates of the shoulders
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        # Convert normalized coordinates to pixel coordinates
        frame_height, frame_width, _ = frame.shape
        left_shoulder_x = int(left_shoulder.x * frame_width)
        left_shoulder_y = int(left_shoulder.y * frame_height)
        right_shoulder_x = int(right_shoulder.x * frame_width)
        right_shoulder_y = int(right_shoulder.y * frame_height)

        # Calculate the average y-coordinate of the shoulders
        shoulders_y_avg = (left_shoulder_y + right_shoulder_y) / 2

        # Draw shoulder lines
        cv2.line(frame, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (0, 0, 255), 2)

    # Save annotated frame to the output folder
    annotated_frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(annotated_frame_path, frame)

    frame_count += 1

# Release the video capture and pose objects
cap.release()
pose.close()

print(f"Annotated frames saved to {output_folder}")
