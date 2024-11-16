import cv2
import dlib
import mediapipe as mp
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


def compute_distance(point1, point2):
    try:
        x1, y1 = map(float, str(point1).strip('()').split(','))
        x2, y2 = map(float, str(point2).strip('()').split(','))
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    except Exception as e:
        print(f"Error computing distance: {e}")
        return np.nan


def load_and_train_model(data_path):
    data = pd.read_csv(data_path)

    selected_features = ['Right_Shoulder_To_Landmark_1_Distance',
                         'Left_Shoulder_To_Landmark_1_Distance',
                         'Right_Shoulder_To_Landmark_7_Distance',
                         'Right_Shoulder_To_Landmark_8_Distance',
                         'Right_Shoulder_To_Landmark_9_Distance',
                         'Left_Shoulder_To_Landmark_9_Distance',
                         'Left_Shoulder_To_Landmark_10_Distance',
                         'Left_Shoulder_To_Landmark_11_Distance',
                         'Left_Shoulder_To_Landmark_12_Distance',
                         'Left_Shoulder_To_Landmark_49_Distance']


    X = data[selected_features]
    y = data.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier()
    model.fit(X_scaled, y)

    return model, scaler


def main():
    video_path = 0  # Use 0 for webcam, or provide the path to a video file
    data_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\continuous_data_labeled.csv'

    model, scaler = load_and_train_model(data_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    frame_count = 0
    distances_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        frame_landmarks = {}

        if faces:
            for face in faces:
                landmarks = predictor(gray, face)
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    frame_landmarks[f'Landmark_{i}'] = (x, y)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for landmark in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER]:
                landmark_index = landmark.value
                landmark_pos = landmarks[landmark_index]
                image_height, image_width, _ = frame.shape
                landmark_x = int(landmark_pos.x * image_width)
                landmark_y = int(landmark_pos.y * image_height)
                frame_landmarks[mp_pose.PoseLandmark(landmark_index).name] = (landmark_x, landmark_y)
                cv2.circle(frame, (landmark_x, landmark_y), 5, (255, 0, 0), -1)

        if 'Landmark_1' in frame_landmarks and 'RIGHT_SHOULDER' in frame_landmarks and 'LEFT_SHOULDER' in frame_landmarks:
            distances = [
                compute_distance(frame_landmarks['Landmark_1'], frame_landmarks['RIGHT_SHOULDER']),
                compute_distance(frame_landmarks['Landmark_1'], frame_landmarks['LEFT_SHOULDER']),
                compute_distance(frame_landmarks.get('Landmark_7', (0, 0)), frame_landmarks['RIGHT_SHOULDER']),
                compute_distance(frame_landmarks.get('Landmark_8', (0, 0)), frame_landmarks['RIGHT_SHOULDER']),
                compute_distance(frame_landmarks.get('Landmark_9', (0, 0)), frame_landmarks['RIGHT_SHOULDER']),
                compute_distance(frame_landmarks.get('Landmark_10', (0, 0)), frame_landmarks['LEFT_SHOULDER']),
                compute_distance(frame_landmarks.get('Landmark_11', (0, 0)), frame_landmarks['LEFT_SHOULDER']),
                compute_distance(frame_landmarks.get('Landmark_12', (0, 0)), frame_landmarks['LEFT_SHOULDER'])
            ]

            distances_buffer.append(distances)
            frame_count += 1

            if frame_count % 10 == 0 and distances_buffer:
                distances_np = np.array(distances_buffer)
                distances_scaled = scaler.transform(distances_np)
                classifications = model.predict(distances_scaled)

                for classification in classifications:
                    print(f"Classification: {classification}")

                distances_buffer = []

        cv2.imshow('Real-Time Landmark Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
