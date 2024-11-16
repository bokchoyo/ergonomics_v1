import csv
import math
import os
import cv2
import dlib
import mediapipe as mp
import numpy as np
import pandas as pd


def compute_angle(vector1, vector2):
    dot_product = sum((a * b) for a, b in zip(vector1, vector2))
    magnitude_v1 = math.sqrt(sum((a * a) for a in vector1))
    magnitude_v2 = math.sqrt(sum((a * a) for a in vector2))
    return math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))


def compute_distance(point1, point2):
    try:
        point1 = str(point1).strip()
        point2 = str(point2).strip()
        x1, y1 = map(float, point1.strip('()').split(','))
        x2, y2 = map(float, point2.strip('()').split(','))

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    except Exception as e:
        print(f"Error computing distance: {e}")
        return np.nan


def record_combined_landmarks(video_path):
    # Initialize mediapipe pose and dlib face detector
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file at {video_path}")
        return

    landmarks_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Convert frame to grayscale for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        frame_landmarks = {'Frame': frame_count}

        # Check and record 68 facial landmarks using dlib
        face_landmarks_detected = False
        if faces:
            for face in faces:
                landmarks = predictor(gray, face)
                face_landmarks_detected = True

                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    frame_landmarks[f'Landmark_{i}'] = (x, y)
                    print(f"Saved landmark {i}: ({x}, {y})")

            if not face_landmarks_detected:
                print(f"Skipping frame {frame_count}: facial landmarks not detected.")
                frame_count += 1
                continue
        else:
            print(f"Skipping frame {frame_count}: no faces detected.")
            frame_count += 1
            continue

        # Check and record shoulder landmarks using mediapipe
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulders_detected = True

            for landmark in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER]:
                landmark_index = landmark.value
                landmark_pos = landmarks[landmark_index]
                image_height, image_width, _ = frame.shape
                landmark_x = int(landmark_pos.x * image_width)
                landmark_y = int(landmark_pos.y * image_height)
                if landmark_x == 0 and landmark_y == 0:
                    shoulders_detected = False
                    break
                frame_landmarks[mp_pose.PoseLandmark(landmark_index).name] = (landmark_x, landmark_y)
                print(f"Saved shoulder landmark: ({landmark_x}, {landmark_y})")

            if not shoulders_detected:
                print(f"Skipping frame {frame_count}: shoulder landmarks not detected.")
                frame_count += 1
                continue
        else:
            print(f"Skipping frame {frame_count}: no pose landmarks detected.")
            frame_count += 1
            continue

        landmarks_data.append(frame_landmarks)
        frame_count += 1

    cap.release()
    pose.close()

    # Save combined landmarks data to CSV
    df = pd.DataFrame(landmarks_data)
    file_path_without_extension = os.path.splitext(video_path)[0]
    path_parts = file_path_without_extension.split(os.sep)
    index = path_parts.index("videos")
    csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(path_parts[index + 1:]) + "_combined_landmarks.csv"
    df.to_csv(csv_path, index=False)

    print(f"Combined landmark coordinates saved to {csv_path}")
    return csv_path


def record_shoulder_landmarks(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
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
                print(f"Saved shoulder landmark: ({landmark_x}, {landmark_y})")
                frame_landmarks[landmark_name] = (landmark_x, landmark_y)

            landmarks_data.append({'Frame': frame_count, **frame_landmarks})

        frame_count += 1

    cap.release()
    pose.close()
    df = pd.DataFrame(landmarks_data)
    file_path_without_extension = os.path.splitext(video_path)[0]
    path_parts = file_path_without_extension.split(os.sep)
    index = path_parts.index("videos")
    csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(path_parts[index + 1:]) + "_shoulders.csv"
    df.to_csv(csv_path, index=False)

    print(f"Shoulder landmark coordinates saved to {csv_path}")
    return csv_path


def record_68_landmarks(video_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file at {video_path}")
        exit()

    landmark_coordinates = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        frame_landmarks = {'Frame': frame_count}

        if faces:
            for face in faces:
                landmarks = predictor(gray, face)

                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    frame_landmarks[f'Landmark_{i}'] = (x, y)
                    print(f"Saved landmark {i}: ({x}, {y})")

        landmark_coordinates.append(frame_landmarks)
        frame_count += 1

    cap.release()
    df = pd.DataFrame(landmark_coordinates)
    file_path_without_extension = os.path.splitext(video_path)[0]
    path_parts = file_path_without_extension.split(os.sep)
    index = path_parts.index("videos")
    csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(path_parts[index + 1:]) + "_68_landmarks.csv"
    df.to_csv(csv_path, index=False)

    print(f"68 landmark coordinates saved to {csv_path}")
    return csv_path


def combine_horizontally(csv_path_1, csv_path_2, output_name):
    csv_path_output = fr'{csv_path_1[:csv_path_1.rfind('\\') + 5]}{output_name}.csv'

    with open(csv_path_1, 'r', newline='') as csv1, \
            open(csv_path_2, 'r', newline='') as csv2, \
            open(csv_path_output, 'w', newline='') as output:
        reader1 = csv.reader(csv1)
        reader2 = csv.reader(csv2)
        writer = csv.writer(output)

        for row1, row2 in zip(reader1, reader2):
            appended_row = row1 + row2[1:]
            writer.writerow(appended_row)

    print(f"Appended data from '{csv_path_2}' to '{csv_path_1}' and saved to '{csv_path_output}'.")
    return csv_path_output


def combine_vertically(csv_path_1, csv_path_2):
    csv_path_output = fr'{csv_path_1[:csv_path_1.rfind('\\') + 5]}_training_data.csv'
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)
    combined_df = pd.concat([df1.iloc[:, 0:], df2.iloc[:, 0:]], axis=0, ignore_index=True)
    combined_df.to_csv(csv_path_output, index=False)

    print(f"Vertically combined data saved to {csv_path_output}")


def record_distances(csv_path):
    df = pd.read_csv(csv_path)

    # Extract the first column (frame number) and the last two columns (for distance computation)
    frame_col = df.columns[0]
    second_to_last_col = df.columns[-2]
    last_col = df.columns[-1]

    new_columns = []

    for i in range(1, 69):
        col_name1 = f'Right_Shoulder_To_Landmark_{i}_Distance'
        col_name2 = f'Left_Shoulder_To_Landmark_{i}_Distance'
        distances1 = []
        distances2 = []

        for index, row in df.iterrows():
            try:
                distance1 = compute_distance(row[second_to_last_col], row[df.columns[i]])
                distance2 = compute_distance(row[last_col], row[df.columns[i]])
                distances1.append(distance1)
                distances2.append(distance2)
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                distances1.append(np.nan)
                distances2.append(np.nan)

        print(f"Processed {len(distances1)} distances for {col_name1}")
        print(f"Processed {len(distances2)} distances for {col_name2}")

        new_columns.append(pd.Series(distances1, name=col_name1))
        new_columns.append(pd.Series(distances2, name=col_name2))

    # Concatenate the new distance columns with the original DataFrame
    df_distances = pd.concat(new_columns, axis=1)

    # Insert the frame number column at the beginning
    df_distances.insert(0, frame_col, df[frame_col])

    # Construct the output CSV file path
    csv_path_output = f"{csv_path[:-4]}_distances.csv"

    # Save the DataFrame to the output CSV file
    df_distances.to_csv(csv_path_output, index=False)

    print(f"Distances saved to {csv_path_output}")
    return csv_path_output


def record_neck_angles(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    file_path_without_extension = os.path.splitext(video_path)[0]
    path_parts = file_path_without_extension.split(os.sep)
    index = path_parts.index("videos")
    csv_path_output = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(
        path_parts[index + 1:]) + "_neck_angles.csv"
    cap = cv2.VideoCapture(video_path)

    with open(csv_path_output, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Neck Angle"])

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
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

            neck_angle = compute_angle(neck_vector, torso_vector)
            print(f"Saved neck angle: {neck_angle}")

            with open(csv_path_output, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count, neck_angle])

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return csv_path_output


def label(csv_path):
    path_parts = csv_path.split(os.sep)

    if path_parts[-2] == 'good':
        value = 1
    else:
        value = 0

    df = pd.read_csv(csv_path)
    df['Posture'] = value
    df.to_csv(csv_path, index=False)

    print(f"New column 'Posture' added to {csv_path}")
    return csv_path


def process(video_path):
    return label(combine_horizontally(record_distances(record_combined_landmarks(video_path)), record_neck_angles(video_path), "_training"))


def process_both(video_number):
    good_video_path = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\front\good\0{video_number}.mp4"
    bad_video_path = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\front\bad\1{video_number}.mp4"

    combine_vertically(process(good_video_path), process(bad_video_path))


combine_horizontally()
