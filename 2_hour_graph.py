import pandas as pd

# Load the labeled test data from the CSV file
labeled_test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\2_hour_video_labeled.csv')

# Calculate the percentage of rows labeled 0 and 1
posture_counts = labeled_test_data['Posture'].value_counts()
total_rows = len(labeled_test_data)
percent_labeled_0 = (posture_counts[0] / total_rows) * 100
percent_labeled_1 = (posture_counts[1] / total_rows) * 100

# Print the percentages
print(f'Percentage labeled 0: {percent_labeled_0:.2f}%')
print(f'Percentage labeled 1: {percent_labeled_1:.2f}%')
