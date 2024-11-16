import pandas as pd

def add_frame_column(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Create a new column "Frame" starting from 7137 and incrementing by 1
    df.insert(0, 'Frame', range(7137, 7137 + len(df)))

    # Save the updated DataFrame back to a new CSV file
    df.to_csv(output_csv, index=False)

# Example usage
input_csv = r'C:\Users\bokch\PyCharm\Ergonomics\data\combo_test_labeled.csv'  # replace with your input CSV file path
output_csv = r'C:\Users\bokch\PyCharm\Ergonomics\data\combo_test_labeled.csv'  # replace with your desired output CSV file path
add_frame_column(input_csv, output_csv)
