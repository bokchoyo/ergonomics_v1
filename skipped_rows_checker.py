import pandas as pd


def find_missing_numbers(csv_path):
    df = pd.read_csv(csv_path, skiprows=1)  # Skip the first row
    frame_numbers = df.iloc[:, 0]  # Assuming the first column contains frame numbers

    # Check if the frame numbers are integers
    if not pd.api.types.is_integer_dtype(frame_numbers):
        raise ValueError("The first column does not contain integer values.")

    missing_numbers = []
    for expected in range(frame_numbers.iloc[0], frame_numbers.iloc[-1] + 1):
        if expected not in frame_numbers.values:
            missing_numbers.append(expected)

    if missing_numbers:
        print("Missing numbers:", missing_numbers)
    else:
        print("No missing numbers found.")


# Example usage
csv_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\front\good\0999_neck_angles.csv'
find_missing_numbers(csv_path)
