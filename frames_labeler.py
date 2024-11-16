import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

# Path to the CSV file
csv_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\combo_test_labeled.csv'

# Path to the folder containing frames
frames_folder = r'C:\Users\bokch\PyCharm\Ergonomics\pictures\combo\side'

# Read the CSV into a pandas DataFrame
df = pd.read_csv(csv_file)

# Print the number of rows in the DataFrame
print(f"Number of rows in the DataFrame: {len(df)}")

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    frame_number = int(row['Frame'])
    binary_value = int(row['Posture'])


    # Determine the text to write based on binary_value
    if binary_value == 1:
        text = "Good"
    elif binary_value == 0:
        text = "Bad"
    else:
        print(f"Binary value neither 1 or 0: {binary_value}")
        continue  # Handle other cases if necessary

    # Construct the filename of the frame
    frame_filename = f"frame_{frame_number:04d}.jpg"  # Adjust extension if different

    # Load the image from frames_folder
    image_path = os.path.join(frames_folder, frame_filename)
    if not os.path.exists(image_path):
        continue  # Skip if image file does not exist

    # Open the image
    image = Image.open(image_path)

    # Get the dimensions of the image
    width, height = image.size

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the font (adjust the font size and style as needed)
    font = ImageFont.truetype("arial.ttf", size=100)

    # Calculate text size and position
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[:2]
    text_position = (width - text_width - 500, 0)  # Top right corner with a small margin

    # Add the text to the image
    draw.text(text_position, text, font=font, fill="black")

    # Save the modified image back to the frames_folder
    modified_image_path = os.path.join(frames_folder, f"frame_{frame_number:04d}.jpg")
    image.save(modified_image_path)

    image.save(image_path)

    # Close the image to free up resources
    image.close()

    print(f"Processed frame {frame_number:04d}: {text}")

print("Processing complete.")
