import pandas as pd
import matplotlib.pyplot as plt


# Function to read data from CSV and plot histogram
def plot_histogram_from_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Ensure there are at least two columns
    if df.shape[1] < 2:
        raise ValueError("The CSV file does not have enough columns")

    # Extract the relevant columns
    angle_column = df.columns[-2]
    classification_column = df.columns[-1]

    # Check if the classification column is binary
    if not set(df[classification_column]).issubset({0, 1}):
        raise ValueError("The classification column must be binary (0 or 1)")

    # Separate the data based on the binary classification
    data_class_0 = df[df[classification_column] == 0][angle_column]
    data_class_1 = df[df[classification_column] == 1][angle_column]

    # Plot the histogram
    plt.hist([data_class_0, data_class_1], bins=30, color=['blue', 'orange'], label=['Class 0', 'Class 1'], alpha=0.7)
    plt.xlabel('Angle')
    plt.ylabel('Frequency')
    plt.title('Histogram of Angles by Binary Classification')
    plt.legend(loc='upper right')
    plt.show()


# Example usage
file_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\all_train_data.csv'  # Replace with your CSV file path
plot_histogram_from_csv(file_path)
