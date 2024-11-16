import ast
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
        # print(f"Error computing distance: {e}")
        return np.nan


def record_combined_landmarks(video_path):
    # Initialize mediapipe pose and dlib face detector
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")
    cap = cv2.VideoCapture(video_path)
    interval = 1

    if not cap.isOpened():
        print(f"Error opening video file at {video_path}")
        return

    landmarks_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval != 0:
            frame_count += 1
            continue

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
                    # print(f"Saved landmark {i}: ({x}, {y})")

            if not face_landmarks_detected:
                # print(f"Skipping frame {frame_count}: facial landmarks not detected.")
                frame_count += 1
                continue
        else:
            # print(f"Skipping frame {frame_count}: no faces detected.")
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
                # print(f"Saved shoulder landmark: ({landmark_x}, {landmark_y})")

            # print(f"Processed Frame {frame_count}")

            if not shoulders_detected:
                # print(f"Skipping frame {frame_count}: shoulder landmarks not detected.")
                frame_count += 1
                continue
        else:
            # print(f"Skipping frame {frame_count}: no pose landmarks detected.")
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
    csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(
        path_parts[index + 1:]) + "_combined_landmarks.csv"

    if os.path.exists(csv_path):
        csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(
        path_parts[index + 1:]) + "_combined_landmarks_v2.csv"

    df.to_csv(csv_path, index=False)

    print(f"Combined landmark coordinates saved to {csv_path}")
    return csv_path


# def record_shoulder_landmarks(video_path):
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         print(f"Error opening video file at {video_path}")
#         exit()
#
#     landmarks_data = []
#     frame_count = 0
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
#
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             frame_landmarks = {}
#
#             for landmark in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER]:
#                 landmark_index = landmark.value
#                 landmark_name = mp_pose.PoseLandmark(landmark_index).name
#                 landmark_pos = landmarks[landmark_index]
#                 image_height, image_width, _ = frame.shape
#                 landmark_x = int(landmark_pos.x * image_width)
#                 landmark_y = int(landmark_pos.y * image_height)
#                 print(f"Saved shoulder landmark: ({landmark_x}, {landmark_y})")
#                 frame_landmarks[landmark_name] = (landmark_x, landmark_y)
#
#             landmarks_data.append({'Frame': frame_count, **frame_landmarks})
#
#         frame_count += 1
#
#     cap.release()
#     pose.close()
#     df = pd.DataFrame(landmarks_data)
#     file_path_without_extension = os.path.splitext(video_path)[0]
#     path_parts = file_path_without_extension.split(os.sep)
#     index = path_parts.index("videos")
#     csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(path_parts[index + 1:]) + "_shoulders.csv"
#     df.to_csv(csv_path, index=False)
#
#     print(f"Shoulder landmark coordinates saved to {csv_path}")
#     return csv_path
#
#
# def record_68_landmarks(video_path):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         print(f"Error opening video file at {video_path}")
#         exit()
#
#     landmark_coordinates = []
#     frame_count = 0
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)
#         frame_landmarks = {'Frame': frame_count}
#
#         if faces:
#             for face in faces:
#                 landmarks = predictor(gray, face)
#
#                 for i in range(68):
#                     x = landmarks.part(i).x
#                     y = landmarks.part(i).y
#                     frame_landmarks[f'Landmark_{i}'] = (x, y)
#                     print(f"Saved landmark {i}: ({x}, {y})")
#
#         landmark_coordinates.append(frame_landmarks)
#         frame_count += 1
#
#     cap.release()
#     df = pd.DataFrame(landmark_coordinates)
#     file_path_without_extension = os.path.splitext(video_path)[0]
#     path_parts = file_path_without_extension.split(os.sep)
#     index = path_parts.index("videos")
#     csv_path = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(path_parts[index + 1:]) + "_68_landmarks.csv"
#     df.to_csv(csv_path, index=False)
#
#     print(f"68 landmark coordinates saved to {csv_path}")
#     return csv_path


def combine_horizontally(csv_path_1, csv_path_2, output_name):
    csv_path_output = fr'{csv_path_1[:csv_path_1.rfind('\\') + 7]}{output_name}.csv'

    if os.path.exists(csv_path_output):
        csv_path_output = fr'{csv_path_1[:csv_path_1.rfind('\\') + 7]}{output_name}_v2.csv'

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
    csv_path_output = fr'{csv_path_1[:csv_path_1.rfind('\\') + 7]}_training_data.csv'

    if os.path.exists(csv_path_output):
        csv_path_output = fr'{csv_path_1[:csv_path_1.rfind('\\') + 7]}_training_data_v2.csv'


    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)
    combined_df = pd.concat([df1.iloc[:, 0:], df2.iloc[:, 0:]], axis=0, ignore_index=True)
    combined_df.to_csv(csv_path_output, index=False)

    print(f"Vertically combined data saved to {csv_path_output}")


def combine_vertically_2(csv_path_1, csv_path_2, name):
    csv_path_output = fr'{csv_path_1[:csv_path_1.rfind('\\') + 1]}{name}.csv'

    if os.path.exists(csv_path_output):
        csv_path_output = fr'{csv_path_1[:csv_path_1.rfind("\\") + 1]}{name}_v2.csv'

    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)
    combined_df = pd.concat([df1.iloc[:, 0:], df2.iloc[:, 0:]], axis=0, ignore_index=True)
    combined_df.to_csv(csv_path_output, index=False)

    print(f"Vertically combined data saved to {csv_path_output}")


def combine_all_vertically(file_paths, output_name):
    output_dir = os.path.dirname(file_paths[0])
    csv_path_output = os.path.join(output_dir, f"{output_name}.csv")

    # Check if the output file already exists and adjust the name if needed
    if os.path.exists(csv_path_output):
        base_name, ext = os.path.splitext(csv_path_output)
        csv_path_output = f"{base_name}_v2{ext}"

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Loop through each file and append its data to the combined DataFrame
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)

    # Save the combined DataFrame to the output file
    combined_df.to_csv(csv_path_output, index=False)

    print(f"Vertically combined data saved to {csv_path_output}")

def record_distances(csv_path):
    df = pd.read_csv(csv_path)
    second_to_last_col = df.columns[-2]
    last_col = df.columns[-1]
    frames_col = df.columns[0]
    new_columns = [df[frames_col]]

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
                # print(f"Error processing row {index}: {e}")
                distances1.append(np.nan)
                distances2.append(np.nan)

        # print(f"Processed {len(distances1)} distances for {col_name1}")
        # print(f"Processed {len(distances2)} distances for {col_name2}")

        new_columns.append(pd.Series(distances1, name=col_name1))
        new_columns.append(pd.Series(distances2, name=col_name2))

    df_distances = pd.concat(new_columns, axis=1)
    csv_path_output = fr'{csv_path[:csv_path.rfind('\\') + 7]}_distances.csv'

    if os.path.exists(csv_path_output):
        csv_path_output = fr'{csv_path[:csv_path.rfind('\\') + 7]}_distances_v2.csv'

    df_distances.to_csv(csv_path_output, index=False)

    print(f"Distances saved to {csv_path_output}")
    return csv_path_output


import numpy as np
import pandas as pd
import os


def compute_front_angle(shoulder_landmark_coordinates, facial_landmark_coordinates):
    """
    Compute the angle (in degrees) between the vector from the shoulder to the facial landmark
    and the horizontal vector (x-axis), including negative angles.

    Arguments:
    - shoulder: (x, y) coordinates of the shoulder.
    - landmark: (x, y) coordinates of the facial landmark.

    Returns:
    - angle: The angle in degrees, ranging from -180 to 180.
    """
    # Vector from shoulder to landmark
    vector = (facial_landmark_coordinates[0] - shoulder_landmark_coordinates[0], facial_landmark_coordinates[1] - shoulder_landmark_coordinates[1])

    # Horizontal vector is (1, 0) along the x-axis
    horizontal_vector = (1, 0)

    # Calculate the angle in radians between the two vectors
    angle_radians = np.arctan2(vector[1], vector[0])  # This returns the angle in radians (-π to π)

    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def record_front_angles(csv_path):
    df = pd.read_csv(csv_path)
    second_to_last_col = df.columns[-2]  # Right shoulder column
    last_col = df.columns[-1]  # Left shoulder column
    frames_col = df.columns[0]  # Frame index column
    new_columns = [df[frames_col]]  # List to hold new angle columns

    for i in range(1, 69):  # Assuming landmarks are in columns 1 to 68
        col_name1 = f'Right_Shoulder_To_Landmark_{i}_Angle'
        col_name2 = f'Left_Shoulder_To_Landmark_{i}_Angle'
        angles1 = []
        angles2 = []

        for index, row in df.iterrows():
            try:
                # Parse the (x, y) string coordinates to tuples
                right_shoulder = ast.literal_eval(row[second_to_last_col])
                left_shoulder = ast.literal_eval(row[last_col])
                landmark = ast.literal_eval(row[df.columns[i]])

                # Compute angles between the shoulder and the landmark
                angle1 = compute_angle(right_shoulder, landmark)
                angle2 = compute_angle(left_shoulder, landmark)

                angles1.append(angle1)
                angles2.append(angle2)

            except Exception as e:
                # If there is an error in computation, store NaN and print error for debugging
                print(f"Error processing row {index}: {e}")
                angles1.append(np.nan)
                angles2.append(np.nan)

        # Add the new angle columns for this landmark
        new_columns.append(pd.Series(angles1, name=col_name1))
        new_columns.append(pd.Series(angles2, name=col_name2))

    # Concatenate the new columns to the original dataframe
    df_angles = pd.concat(new_columns, axis=1)

    # Save the new dataframe to a CSV file
    csv_path_output = fr'{csv_path[:csv_path.rfind("\\") + 7]}_angles.csv'

    if os.path.exists(csv_path_output):
        csv_path_output = fr'{csv_path[:csv_path.rfind("\\") + 7]}_angles_v2.csv'

    df_angles.to_csv(csv_path_output, index=False)

    print(f"Angles saved to {csv_path_output}")
    return csv_path_output

def record_angles(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    file_path_without_extension = os.path.splitext(video_path)[0]
    path_parts = file_path_without_extension.split(os.sep)
    index = path_parts.index("videos")
    csv_path_output = r"C:\Users\bokch\PyCharm\Ergonomics\data\\" + os.sep.join(
        path_parts[index + 1:]) + "_three_angles.csv"
    cap = cv2.VideoCapture(video_path)

    with open(csv_path_output, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Neck_Torso_Angle", "Neck_Vertical_Angle", "Torso_Vertical_Angle"])

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
            # torso_vector = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[
            #     mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2 -
            #                 (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[
            #                     mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
            #                 (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[
            #                     mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 -
            #                 (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[
            #                     mp_pose.PoseLandmark.RIGHT_HIP].y) / 2]
            shoulder_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2

            hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[
                                mp_pose.PoseLandmark.RIGHT_HIP].x) / 2

            shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[
                                mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2

            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[
                                mp_pose.PoseLandmark.RIGHT_HIP].y) / 2

            torso_vector = [shoulder_x - hip_x,
                            shoulder_y - hip_y]

            # Vertical Vector
            vertical_vector = [0, shoulder_y - hip_y]

            neck_torso_angle = compute_front_angle(neck_vector, torso_vector)
            neck_vertical_angle = compute_front_angle(neck_vector, vertical_vector)
            torso_vertical_angle = compute_front_angle(torso_vector, vertical_vector)
            # print(f"Saved neck angle: {neck_angle}")

            with open(csv_path_output, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count, neck_torso_angle, neck_vertical_angle, torso_vertical_angle])

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


def create_testing_set(video_number):
    front_video = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\front\continuous\{video_number}.mp4"
    side_video = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\side\continuous\{video_number}.MOV"

    return combine_horizontally(record_distances(record_combined_landmarks(front_video)),
                                record_angles(side_video),
                                "_testing_v2")


def process(front_path, side_path):
    return label(combine_horizontally(record_distances(record_combined_landmarks(front_path)),
                                      record_angles(side_path), "_training_v2"))


def create_training_set(video_number):
    front_good_path = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\front\good\0{video_number}.mp4"
    front_bad_path = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\front\bad\1{video_number}.mp4"
    side_good_path = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\side\good\0{video_number}.MOV"
    side_bad_path = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\side\bad\1{video_number}.MOV"
    combine_vertically_2(process(front_good_path, side_good_path), process(front_bad_path, side_bad_path), 'set_training_1-2')


# create_testing_set('test_0')

# for n in ['001', '002', '003', '004', '006', '007', '008', '009', '010', '011']:
#     create_training_set(n)

# video_path = r"/videos/front/Long_Video.mp4"
#
# for video_name in ['test_0', 'test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 'test_8', 'test_9']:
#     create_testing_set(video_name)
#
def create_train_set(test_id_list):
    id_list = [id for id in all_ids if id not in test_id_list]
    fp = r'C:\Users\bokch\PyCharm\Ergonomics\data\final'
    combine_all_vertically([fp + f'\\test_{id_list[0]}_testing_v2_v2.csv',
                            fp + f'\\test_{id_list[1]}_testing_v2_v2.csv',
                            fp + f'\\test_{id_list[2]}_testing_v2_v2.csv',
                            fp + f'\\test_{id_list[3]}_testing_v2_v2.csv',
                            fp + f'\\test_{id_list[4]}_testing_v2_v2.csv',
                            fp + f'\\test_{id_list[5]}_testing_v2_v2.csv',
                            fp + f'\\test_{id_list[6]}_testing_v2_v2.csv'],
                           f'training_continuous_{test_id_list[0]}-{test_id_list[1]}-{test_id_list[2]}')

def create_test_set(id_list):
    fp = r'C:\Users\bokch\PyCharm\Ergonomics\data\final'
    combine_all_vertically([fp + f'\\test_{id_list[0]}_testing_v2_v2.csv',
                            fp + f'\\test_{id_list[1]}_testing_v2_v2.csv',
                            fp + f'\\test_{id_list[2]}_testing_v2_v2.csv'],
                           f'testing_continuous_{id_list[0]}-{id_list[1]}-{id_list[2]}')

all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# test_ids = [3, 5, 9]
# train_ids = [id for id in all_ids if id not in test_ids]
# # create_test_set([3, 5, 9])
# print(train_ids)
# create_train_set(test_ids)

# combine_vertically_2(r'C:\Users\bokch\PyCharm\Ergonomics\data\final\testing_continuous_3-5-9.csv', r'C:\Users\bokch\PyCharm\Ergonomics\data\final\testing_continuous_3-5-9.csv', 'all_continuous')
#

record_front_angles(r'C:\Users\bokch\PyCharm\Ergonomics\data\front\continuous\test_1_combined_landmarks_v2.csv')

# fp = r'C:\Users\bokch\PyCharm\Ergonomics\data\front\continuous'
# combine_all_vertically([fp + '\\test_4_testing_v2.csv',
#                         fp + '\\test_5_testing_v2.csv',
#                         fp + '\\test_6_testing_v2.csv',
#                         fp + '\\test_7_testing_v2.csv',
#                         fp + '\\test_8_testing_v2.csv',
#                         fp + '\\test_9_testing_v2.csv', r'C:\Users\bokch\PyCharm\Ergonomics\data\set_testing_1-2.csv'],
#                        '7_people_train_data')

# combine_vertically_2(r'C:\Users\bokch\PyCharm\Ergonomics\data\training_set.csv', r'C:\Users\bokch\PyCharm\Ergonomics\data\testing_set.csv', 'training_testing_set')


#
# fp = r'C:\Users\bokch\PyCharm\Ergonomics\data\front\continuous'
# combine_all_vertically([fp + '\\test_1_testing_v2.csv',
#                         fp + '\\test_2_testing_v2.csv',
#                         fp + '\\test_3_testing_v2.csv'],
#                        '3_people_test_data')

# for video_name in ['test_8', 'test_9']:
#     create_testing_set(video_name)
