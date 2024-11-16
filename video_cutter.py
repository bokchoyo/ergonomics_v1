from moviepy.editor import VideoFileClip


def split_video_at_4_minutes(input_video_path, output_video_path1, output_video_path2):
    # Load the video
    video = VideoFileClip(input_video_path)

    # Define the 4-minute mark in seconds
    four_minute_mark = 4 * 60

    # Check the duration of the video
    video_duration = video.duration

    if video_duration > four_minute_mark:
        # Split the video at the 4-minute mark
        first_part = video.subclip(0, four_minute_mark)
        second_part = video.subclip(four_minute_mark, video_duration)

        # Save the two parts
        first_part.write_videofile(output_video_path1, codec="libx264", preset="medium",
                                   ffmpeg_params=["-vf", "scale=iw:ih"])
        second_part.write_videofile(output_video_path2, codec="libx264", preset="medium",
                                    ffmpeg_params=["-vf", "scale=iw:ih"])

        print(f"Video successfully split and saved as {output_video_path1} and {output_video_path2}.")
    else:
        print("The video is shorter than 4 minutes. No splitting is performed.")


# Example usage
input_video_path = r"C:\Users\bokch\PyCharm\Ergonomics\videos\side\combo.MOV"
output_video_part1_path = r"C:\Users\bokch\PyCharm\Ergonomics\videos\side\good\0999.MOV"
output_video_part2_path = r"C:\Users\bokch\PyCharm\Ergonomics\videos\side\bad\0999.MOV"

split_video_at_4_minutes(input_video_path, output_video_part1_path, output_video_part2_path)
