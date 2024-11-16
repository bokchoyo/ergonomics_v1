import random
import mediapipe as mp
import dlib
import cv2
import os

def draw_landmarks_on_random_frame(video_path, output_folder):
    # Initialize mediapipe pose and dlib face detector
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\bokch\PyCharm\Ergonomics\shape_predictor.dat")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file at {video_path}")
        return

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Choose a random frame number
    random_frame_number = random.randint(0, total_frames - 1)

    # Set the video capture to the random frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

    ret, frame = cap.read()
    if not ret:
        print(f"Error reading frame {random_frame_number} from video file at {video_path}")
        return

    # Convert frame to RGB for mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Convert frame to grayscale for dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Draw facial landmarks
    green_landmarks = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 32, 49]
    if faces:
        for face in faces:
            landmarks = predictor(gray, face)
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                if i in green_landmarks:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 0, 255)  # Red
                cv2.circle(frame, (x, y), 5, color, -1)

    # Draw shoulder landmarks
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for landmark in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER]:
            landmark_index = landmark.value
            landmark_pos = landmarks[landmark_index]
            image_height, image_width, _ = frame.shape
            landmark_x = int(landmark_pos.x * image_width)
            landmark_y = int(landmark_pos.y * image_height)
            cv2.circle(frame, (landmark_x, landmark_y), 10, (0, 255, 0), -1)  # Green

    # Save the frame with landmarks
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'random_frame_{random_frame_number}.jpg')
    cv2.imwrite(output_path, frame)

    print(f"Frame with landmarks saved to {output_path}")

    cap.release()
    pose.close()

# Example usage
draw_landmarks_on_random_frame(r"C:\Users\bokch\PyCharm\Ergonomics\videos\front\Front_1.mp4", r"C:\Users\bokch\PyCharm\Ergonomics\pictures\front\landmarks")
