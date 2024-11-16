import pandas as pd

# Paths to the two CSV files
file_path1 = r'C:\Users\bokch\PyCharm\Ergonomics\data\continuous_data_labeled.csv'
file_path2 = r'C:\Users\bokch\PyCharm\Ergonomics\data\combo_test_labeled.csv'

# Read the CSV files into pandas DataFrames
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Combine the two DataFrames vertically
combined_df = pd.concat([df1, df2], ignore_index=True)

# Optionally, remove any duplicate rows
combined_df = combined_df.drop_duplicates()

# Save the combined DataFrame to a new CSV file
output_file_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\data_set_2.csv'
combined_df.to_csv(output_file_path, index=False)

print(f"The combined CSV file has been saved to {output_file_path}.")
