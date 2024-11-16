import pandas as pd

# Path to your input CSV file
input_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\data_set_1.csv'

# Path to the output CSV file
output_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\front\bad\1001_training.csv'

# Read the existing CSV file
df = pd.read_csv(input_file)

# Add a new column 'Posture' with a value of 1 for each row
df['Posture'] = 0

# Save the updated DataFrame back to a CSV file
df.to_csv(output_file, index=False)

print(f"New column 'Posture' added to {output_file}")
