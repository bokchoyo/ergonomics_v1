import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

name1 = "data/neck_angles"  # Two columns
name2 = "data/neck_angles"

# Read the CSV files
data1 = pd.read_csv(f'{name1}.csv')
data2 = pd.read_csv(f'{name2}.csv')

# Extract x and y values from the dataframes
id_column_frame_number = 0
id_column_neck_angle = 1
frames_per_second = 30
number_of_seconds_in_a_minute = 60
factor_to_convert_from_frame_to_minute = frames_per_second * number_of_seconds_in_a_minute
frame_numbers_by_second = data1.iloc[:, id_column_frame_number] / factor_to_convert_from_frame_to_minute  # Assuming the first column is x
# print(len(frame_numbers_by_second))
neck_angles = data1.iloc[:, id_column_neck_angle]  # Second Column
# print(len(neck_angles))
frame_numbers = data2.iloc[:, id_column_frame_number]  # Second Column
# print(len(frame_numbers))
# Initialize variables for iteration
degree = 4
tolerance = 1000
removed_points = True

# Process for the first dataset
while removed_points:
    # Fit a polynomial curve
    p1 = np.polyfit(frame_numbers_by_second, neck_angles, degree)
    f1 = np.poly1d(p1)

    # Calculate residuals
    residuals1 = neck_angles - f1(frame_numbers_by_second)

    # Filter out points with residuals greater than tolerance
    filtered_x1 = frame_numbers_by_second[abs(residuals1) <= tolerance]
    filtered_y1 = neck_angles[abs(residuals1) <= tolerance]

    # Check if any points were removed
    if len(filtered_x1) == len(frame_numbers_by_second):
        removed_points = False
    else:
        frame_numbers_by_second = filtered_x1
        neck_angles = filtered_y1

# Process for the second dataset
removed_points = True
while removed_points:
    print(len(frame_numbers))
    print(len(neck_angles))
    # Fit a polynomial curve
    p2 = np.polyfit(frame_numbers, neck_angles, degree)
    f2 = np.poly1d(p2)

    # Calculate residuals
    residuals2 = neck_angles - f2(frame_numbers)

    # Filter out points with residuals greater than tolerance
    filtered_x2 = frame_numbers[abs(residuals2) <= tolerance]
    filtered_y2 = neck_angles[abs(residuals2) <= tolerance]

    # Check if any points were removed
    if len(filtered_x2) == len(frame_numbers):
        removed_points = False
    else:
        frame_numbers = filtered_x2
        neck_angles = filtered_y2

# Calculate lowest and average angles for both graphs
y_predicted1 = f1(frame_numbers_by_second)
lowest_value1 = min(y_predicted1)
average_value1 = np.mean(filtered_y1)

y_predicted2 = f2(frame_numbers)
lowest_value2 = min(y_predicted2)
average_value2 = np.mean(filtered_y2)

calc_x = 0.05
calc_y1 = 0.2
calc_y2 = 0.15
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(filtered_x1, filtered_y1, label=f'Filtered Data (Tolerance {tolerance})')
plt.plot(frame_numbers_by_second, f1(frame_numbers_by_second), color='red', label=f'Curve of Best Fit (Degree {degree})')
plt.xlabel('Time (Minutes)')
plt.ylabel('Neck Angle')
plt.title(f'Neck Angle Over Time {name1}')
plt.grid(True)
plt.legend(loc='lower left')
plt.text(calc_x, calc_y1, f'Lowest Angle: {lowest_value1:.2f}', transform=plt.gca().transAxes)
plt.text(calc_x, calc_y2, f'Average Angle: {average_value1:.2f}', transform=plt.gca().transAxes)
plt.ylim(150, 180)  # Set y-axis limits
plt.gca().xaxis.set_major_locator(MultipleLocator(10))  # Set major ticks every 10 minutes

# Plot for the second dataset
plt.subplot(1, 2, 2)
plt.plot(filtered_x2, filtered_y2, label=f'Filtered Data (Tolerance {tolerance})')
plt.plot(frame_numbers, f2(frame_numbers), color='red', label=f'Curve of Best Fit (Degree {degree})')
plt.xlabel('Time (Minutes)')
plt.ylabel('Neck Angle')
plt.title(f'Neck Angle Over Time {name2}')
plt.grid(True)
plt.legend(loc='lower left')
plt.text(calc_x, calc_y1, f'Lowest Angle: {lowest_value2:.2f}', transform=plt.gca().transAxes)
plt.text(calc_x, calc_y2, f'Average Angle: {average_value2:.2f}', transform=plt.gca().transAxes)
plt.ylim(150, 180)  # Set y-axis limits
plt.gca().xaxis.set_major_locator(MultipleLocator(10))  # Set major ticks every 10 minutes

plt.tight_layout()
plt.show()
