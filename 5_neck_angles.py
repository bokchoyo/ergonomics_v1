import cv2
import mediapipe as mp
import math
import csv


# Function to calculate angle between two vectors
def angle(v1, v2):
    dot_product = sum((a * b) for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum((a * a) for a in v1))
    magnitude_v2 = math.sqrt(sum((a * a) for a in v2))
    return math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))


# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Path to the video file

video_path = rf"C:\Users\bokch\PyCharm\Ergonomics\videos\side\bad\1001.MOV"

# Output CSV file
output_csv = rf"C:\Users\bokch\PyCharm\Ergonomics\data\side\bad\1001_angles"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Write headers to CSV file
with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Neck Angle"])

# Loop through each frame of the video
frame_count = 0
angle_calculation_interval = 1  # Calculate angle every 10 frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame and get the pose landmarks
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark

        if frame_count % angle_calculation_interval == 0:  # Calculate angle only for every 10 frames
            # Calculate neck angle
            neck_vector_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - landmarks[
                mp_pose.PoseLandmark.RIGHT_EAR].x
            neck_vector_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - landmarks[
                mp_pose.PoseLandmark.RIGHT_EAR].y
            neck_vector = [neck_vector_x, neck_vector_y]
            torso_vector = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2 -
                            (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[
                                mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
                            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[
                                mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 -
                            (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[
                                mp_pose.PoseLandmark.RIGHT_HIP].y) / 2]

            neck_angle = angle(neck_vector, torso_vector)

            # Write angle to CSV file
            with open(output_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count, neck_angle])

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
