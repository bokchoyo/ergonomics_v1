import os
import pandas as pd


def combine_csv_files(folder_path, output_file):
    # List to hold dataframes
    dataframes = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file and append to the list of dataframes
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all dataframes in the list
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"All CSV files have been combined and saved to {output_file}")


# Example usage
folder_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\training'
output_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\set_training_1-2'
combine_csv_files(folder_path, output_file)
