import cv2


def count_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return -1

    # Initialize frame count
    frame_count = 0

    # Loop through frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

    # Release the video capture object
    video.release()

    return frame_count


# Example usage
video_path = fr"C:\Users\bokch\PyCharm\Ergonomics\videos\front\good\0{999}.mp4"
frame_count = count_frames(video_path)
print(f"Number of frames in the video: {frame_count}")
