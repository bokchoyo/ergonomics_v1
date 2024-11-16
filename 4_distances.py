import pandas as pd
import numpy as np


def compute_distance(point1, point2):
    try:
        # Convert to string and strip leading/trailing whitespace
        point1 = str(point1).strip()
        point2 = str(point2).strip()

        # Extract coordinates
        x1, y1 = map(float, point1.strip('()').split(','))
        x2, y2 = map(float, point2.strip('()').split(','))

        # Calculate Euclidean distance
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    except Exception as e:
        print(f"Error computing distance: {e}")
        return np.nan  # Return NaN if there's an error


# Load the CSV file into a pandas DataFrame
csv_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\front\bad\1001_70.csv'
df = pd.read_csv(csv_file)


second_to_last_col = df.columns[-2]  # Second to last column (LEFT_SHOULDER)
last_col = df.columns[-1]  # Last column (LEFT_EYE_INNER)

# List to store new columns
new_columns = []

for i in range(1, 69):
    col_name1 = f'Right_Shoulder_To_Landmark_{i}_Distance'
    col_name2 = f'Left_Shoulder_To_Landmark_{i}_Distance'

    distances1 = []
    distances2 = []

    for index, row in df.iterrows():
        try:
            distance1 = compute_distance(row[second_to_last_col], row[df.columns[i]])
            distance2 = compute_distance(row[last_col], row[df.columns[i]])
            distances1.append(distance1)
            distances2.append(distance2)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            distances1.append(np.nan)  # Handle error by appending NaN
            distances2.append(np.nan)  # Handle error by appending NaN

    print(f"Processed {len(distances1)} distances for {col_name1}")
    print(f"Processed {len(distances2)} distances for {col_name2}")

    # Store new columns in a list
    new_columns.append(pd.Series(distances1, name=col_name1))
    new_columns.append(pd.Series(distances2, name=col_name2))

# Concatenate all new columns to the original DataFrame
df = pd.concat([df] + new_columns, axis=1)

# Save the updated DataFrame back to a new CSV file with headers
output_csv_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\front\bad\1001_distances.csv'
df.to_csv(output_csv_file, index=False)

print(f"CSV file '{output_csv_file}' has been created with headers in the top row.")
