import pandas as pd

def combine_csv_vertically(file1, file2, output_file):
    # Read both CSV files into pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Get column names for both DataFrames
    columns1 = df1.columns
    columns2 = df2.columns

    # Combine data starting from the 72nd column onward
    combined_df = pd.concat([df1.iloc[:, 0:], df2.iloc[:, 0:]], axis=0, ignore_index=True)

    # Write combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print(f"Combined data written to {output_file}")


# Example usage:
file1 = r'C:\Users\bokch\PyCharm\Ergonomics\data\testing_data.csv'
file2 = r'C:\Users\bokch\PyCharm\Ergonomics\data\training_data.csv'
output_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\all_train_data.csv'

combine_csv_vertically(file1, file2, output_file)
