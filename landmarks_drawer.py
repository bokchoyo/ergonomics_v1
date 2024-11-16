import cv2
import dlib
import os
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")

# Path to the input video
video_path = r"C:\Users\bokch\PyCharm\Ergonomics\videos\front\good\0999.mp4"

# Directory to save frames with landmarks
output_folder = r"C:\Users\bokch\PyCharm\Ergonomics\pictures\combo\front"
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

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Draw circles for facial landmarks if faces are detected
    if faces:
        for face in faces:
            # Detect facial landmarks
            landmarks_dlib = predictor(gray, face)

            # Draw a red circle for each landmark point (68 landmarks)
            for i in range(68):
                x = landmarks_dlib.part(i).x
                y = landmarks_dlib.part(i).y
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw a red circle

    # Convert the frame to RGB for pose detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find the pose landmarks
    results = pose.process(frame_rgb)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        landmarks_pose = results.pose_landmarks.landmark

        # Define the desired landmarks (right and left shoulders)
        desired_landmarks = [
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        # Iterate through desired landmarks and draw their coordinates
        for landmark in desired_landmarks:
            landmark_index = landmark.value  # Get the index of the landmark
            landmark_pos = landmarks_pose[landmark_index]  # Get the landmark position

            # Convert normalized coordinates to pixel coordinates
            image_height, image_width, _ = frame.shape
            landmark_x = int(landmark_pos.x * image_width)
            landmark_y = int(landmark_pos.y * image_height)

            # Draw the landmark on the frame
            cv2.circle(frame, (landmark_x, landmark_y), 5, (255, 0, 0), -1)  # Draw a blue circle

    # Save the frame with landmarks to the output folder
    output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, frame)

    frame_count += 1

# Release the video capture and pose object
cap.release()
pose.close()

print(f"Frames with combined landmarks saved to {output_folder}")
