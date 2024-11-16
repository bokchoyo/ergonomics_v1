import pandas as pd

# Load the first CSV file (all frame numbers and "Neck Angle")
first_csv = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\front\good\0999_neck_angles.csv')

# Load the second CSV file (some frame numbers missing and 136 columns)
second_csv = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\front\good\0999_distances.csv')

# Merge the two CSV files on the "Frame Number" column
merged_csv = second_csv.merge(first_csv[['Frame', 'Neck Angle']], on='Frame', how='left')

# Save the merged CSV file
merged_csv.to_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\front\good\0999_data.csv', index=False)

print("Merged file saved as '0999_data.csv'")
