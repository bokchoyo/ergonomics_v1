import cv2
import os

# Path to the input video file
video_path = r"C:\Users\bokch\PyCharm\Ergonomics\videos\side\continuous\test_4.MOV"

# Directory to save the frames
output_folder = r"C:\Users\bokch\PyCharm\Ergonomics\pictures\frames\continuous\test_4"
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error opening video file at {video_path}")
    exit()

frame_count = 0

# Process each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Construct the output frame file name
    frame_filename = f"frame_{frame_count:04d}.jpg"
    frame_filepath = os.path.join(output_folder, frame_filename)

    # Save the frame
    cv2.imwrite(frame_filepath, frame)

    frame_count += 1

# Release the video capture
cap.release()

print(f"Total {frame_count} frames saved to {output_folder}")
