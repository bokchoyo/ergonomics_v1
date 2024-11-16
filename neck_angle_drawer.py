import cv2
import mediapipe as mp
import math
import os

# Function to calculate angle between two vectors
def angle(v1, v2):
    dot_product = sum((a * b) for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum((a * a) for a in v1))
    magnitude_v2 = math.sqrt(sum((a * a) for a in v2))
    return math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Path to the video file
video_path = r"C:\Users\bokch\PyCharm\Ergonomics\videos\side\continuous\test_4.MOV"

# Output folder for saving annotated frames
output_folder = r"C:\Users\bokch\PyCharm\Ergonomics\pictures\neck_angles\continuous\test_4"
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error opening video file at {video_path}")
    exit()

# Loop through each frame of the video
frame_count = 0  # Calculate angle every 2 frames (every other frame)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame and get the pose landmarks
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    landmarks = results.pose_landmarks.landmark

    # Calculate neck vector
    neck_vector_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x
    neck_vector_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y
    neck_vector = [neck_vector_x, neck_vector_y]

    # Calculate torso vector
    torso_vector = [
        (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2 -
        (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
        (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 -
        (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    ]

    # Calculate neck angle
    neck_angle = angle(neck_vector, torso_vector)

    # Draw neck line
    cv2.line(frame, (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])),
             (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * frame.shape[1]),
              int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y * frame.shape[0])),
             (0, 255, 0), 2)

    # Draw torso line
    cv2.line(frame, (int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2 * frame.shape[1]),
                     int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 * frame.shape[0])),
             (int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2 * frame.shape[1]),
              int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2 * frame.shape[0])),
             (0, 0, 255), 2)

# Save annotated frame to the output folder
    annotated_frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(annotated_frame_path, frame)

    frame_count += 1

# Release the video capture and pose objects
cap.release()
cv2.destroyAllWindows()

print(f"Annotated frames saved to {output_folder}")
