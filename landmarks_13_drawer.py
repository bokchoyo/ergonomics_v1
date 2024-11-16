import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Path to the input image
input_image_path = r"C:\Users\bokch\PyCharm\Ergonomics\pictures\final\frame_29187.jpg"
# Path to the output image
output_image_path = r"C:\Users\bokch\PyCharm\Ergonomics\pictures\final\front_landmarks_2.jpg"

# Read the image
image = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error loading image at {input_image_path}")
    exit()

# Convert the image to RGB for pose detection
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and find the pose landmarks
results = pose.process(image_rgb)

# Check if pose landmarks are detected
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # Define the desired landmarks (indices) from mp_pose.PoseLandmark
    desired_landmarks = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_SHOULDER
    ]

    # Iterate through desired landmarks and draw their coordinates
    for landmark in desired_landmarks:
        landmark_index = landmark.value  # Get the index of the landmark
        landmark_pos = landmarks[landmark_index]  # Get the landmark position

        # Convert normalized coordinates to pixel coordinates
        image_height, image_width, _ = image.shape
        landmark_x = int(landmark_pos.x * image_width)
        landmark_y = int(landmark_pos.y * image_height)

        # Draw the landmark on the image
        cv2.circle(image, (landmark_x, landmark_y), 25, (0, 255, 0), -1)

# Save the image with landmarks drawn
cv2.imwrite(output_image_path, image)

# Release the pose object
pose.close()

print(f"Image with landmarks saved to {output_image_path}")
