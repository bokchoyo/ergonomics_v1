import os

import cv2
import mediapipe as mp
import pandas as pd


def record_shoulder_landmarks(file_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    video_path = file_path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file at {video_path}")
        exit()

    landmarks_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = {}

            for landmark in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER]:
                landmark_index = landmark.value
                landmark_name = mp_pose.PoseLandmark(landmark_index).name
                landmark_pos = landmarks[landmark_index]
                image_height, image_width, _ = frame.shape
                landmark_x = int(landmark_pos.x * image_width)
                landmark_y = int(landmark_pos.y * image_height)
                frame_landmarks[landmark_name] = (landmark_x, landmark_y)

            landmarks_data.append({'Frame': frame_count, **frame_landmarks})

        frame_count += 1

    cap.release()
    pose.close()
    df = pd.DataFrame(landmarks_data)
    file_path_without_extension = os.path.splitext(file_path)[0]
    path_parts = file_path_without_extension.split(os.sep)
    index = path_parts.index("videos")
    csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(path_parts[index + 1:]) + "_shoulders.csv"
    df.to_csv(csv_path, index=False)

    print(f"Shoulder landmark coordinates saved to {csv_path}")


record_shoulder_landmarks(r"C:\Users\bokch\PyCharm\Ergonomics\videos\front\good\0002.mp4")
