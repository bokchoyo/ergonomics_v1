import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
file_path = 'data.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Step 2: Extract the second and third columns
# Note: pandas columns are 0-indexed, so column 2 is index 1, and column 3 is index 2
x = data.iloc[:, 2]  # Third column
y = data.iloc[:, 1]  # Second column

# Extract the column names for labels
x_label = data.columns[2]
y_label = data.columns[1]

# Step 3: Plot the graph
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=10, color='b')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f"{y_label} vs. {x_label}")
plt.grid(True)
plt.show()
