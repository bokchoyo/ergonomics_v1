import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Example usage
discrete_training_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\set_training_1-2.csv'
continuous_training_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\final\training_continuous_3-5-9.csv'
continuous_testing_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\final\testing_continuous_3-5-9.csv'
all_continuous_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\final\all_continuous.csv'
file_path = all_continuous_path
# file_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\continuous_data_labeled.csv'

# Read the CSV file
df = pd.read_csv(file_path)


def plot_correlation_heatmap(df):
    front_data = df.iloc[:, 1:-4]
    font_size = 7
    correlation_matrix = front_data.corr().abs()  # Exclude the second to last column

    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.8
    threshold = 0.985
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # Drop highly correlated features
    selected_features = [column for column in front_data if
                         column not in to_drop]  # Exclude the second to last column

    # Print the names of the columns that are kept
    print(f"Selected {len(selected_features)} features:", selected_features)
    selected_correlation_matrix = front_data[selected_features].corr().abs()
    new_column_names = [col[:-9] for col in selected_correlation_matrix.columns]
    selected_correlation_matrix.columns = new_column_names
    selected_correlation_matrix.index = new_column_names
    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(selected_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                square=True, annot_kws={"fontsize": font_size})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('Correlation Heatmap of Selected Features')
    plt.tight_layout()  # Adjust the layout to ensure everything fits
    plt.show()


# plot_correlation_heatmap(df)


def calculate_neck_angle_statistics(df):
    # Check the structure of the dataframe to ensure the second to last column is "Neck Angle" and the last column is "Posture"
    if df.columns[-2] != "Neck Angle" or df.columns[-1] != "Posture":
        raise ValueError("CSV file does not have the expected structure with 'Neck Angle' as the second to last column and 'Posture' as the last column.")

    # Separate the data based on the classification
    class_0 = df[df["Posture"] == 0]
    class_1 = df[df["Posture"] == 1]

    # Calculate the average neck angle for each classification
    average_neck_angle_0 = class_0["Neck Angle"].mean()
    average_neck_angle_1 = class_1["Neck Angle"].mean()

    # Calculate the standard deviation of the neck angle for each classification
    std_dev_neck_angle_0 = class_0["Neck Angle"].std()
    std_dev_neck_angle_1 = class_1["Neck Angle"].std()

    return (average_neck_angle_0, std_dev_neck_angle_0), (average_neck_angle_1, std_dev_neck_angle_1)


def calculate_average_difference(df):
    # Ensure the "Posture" column is present
    if "Posture" not in df.columns:
        raise ValueError("CSV file does not have the expected 'Posture' column.")

    # Separate the data based on the classification
    class_0 = df[df["Posture"] == 0]
    class_1 = df[df["Posture"] == 1]

    differences = []
    for col in df.columns[:-1]:  # Exclude the "Posture" column
        avg_0 = class_0[col].mean()
        avg_1 = class_1[col].mean()
        differences.append(avg_1 - avg_0)

    average_difference = sum(differences) / len(differences)
    return average_difference


mm_per_pixel = 0.48404222097


def calculate_column_statistics(df):
    # Ensure the "Posture" column is present
    if "Posture" not in df.columns:
        raise ValueError("CSV file does not have the expected 'Posture' column.")

    # Check if there are enough columns
    if len(df.columns) < 137:
        raise ValueError("CSV file does not have enough columns to calculate averages for columns 2 to 137.")

    # Select columns 2 through 137 (assuming 0-indexing, these are columns 1 to 136)
    columns_to_average = df.iloc[:, 1:137]

    # Separate the data based on the classification
    class_0 = columns_to_average[df["Posture"] == 0] * mm_per_pixel
    class_1 = columns_to_average[df["Posture"] == 1] * mm_per_pixel

    # Calculate the average and standard deviation for each classification
    average_class_0 = class_0.mean().mean()
    std_dev_class_0 = class_0.stack().std()

    average_class_1 = class_1.mean().mean()
    std_dev_class_1 = class_1.stack().std()

    return (average_class_0, std_dev_class_0), (average_class_1, std_dev_class_1)


def plot_distance_boxplot(df):
    if "Posture" not in df.columns or len(df.columns) < 137:
        raise ValueError("CSV file does not have the expected structure.")

    # Select columns 2 through 137 (assuming 0-indexing, these are columns 1 to 136)
    columns_to_plot = df.iloc[:, 1:137] * mm_per_pixel

    # Create a melted DataFrame for easier plotting
    df_melted = columns_to_plot.copy()
    df_melted['Posture'] = df['Posture']

    # Convert to a long format DataFrame
    df_long = df_melted.melt(id_vars='Posture', var_name='Landmark', value_name='Distance')
    # Calculate and print 25th percentile, 75th percentile, and IQR for each posture
    percentiles = df_long.groupby('Posture')['Distance'].quantile([0.25, 0.75]).unstack()
    percentiles.columns = ['25th Percentile', '75th Percentile']
    percentiles['IQR'] = percentiles['75th Percentile'] - percentiles['25th Percentile']

    print("Landmark Distance Percentiles and IQR by Posture:")
    print(percentiles)
    # Plot the box plot
    plt.figure(figsize=(4.5, 6))
    sns.boxplot(x='Posture', y='Distance', data=df_long)
    plt.title('Landmark Distances by Posture')
    plt.xlabel('Posture')
    plt.ylabel('Millimeters')
    plt.yticks(np.arange(0, 451, 50))
    plt.ylim(0, 455)
    plt.show()

    stats_0, stats_1 = calculate_column_statistics(df)
    average_0, std_dev_0 = stats_0
    average_1, std_dev_1 = stats_1

    # Print results
    print(f"Average for classification 0: {average_0}, Standard Deviation: {std_dev_0}")
    print(f"Average for classification 1: {average_1}, Standard Deviation: {std_dev_1}")


#


def graph_distances():
    if "Posture" not in df.columns:
        raise ValueError("CSV file does not have the expected 'Posture' column.")

    # Check if there are enough columns
    if len(df.columns) < 137:
        raise ValueError("CSV file does not have enough columns to calculate averages for columns 2 to 137.")

    # Select columns 2 through 137 (assuming 0-indexing, these are columns 1 to 136)
    columns_to_plot = df.iloc[:, 1:137]

    # Separate the data based on the classification
    class_0 = columns_to_plot[df["Posture"] == 0]
    class_1 = columns_to_plot[df["Posture"] == 1]

    # Stack the data for each class to plot in a single histogram
    class_0_values = class_0.stack()
    class_1_values = class_1.stack()

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(class_0_values, color='red', label='0', kde=False, bins=50, alpha=0.5)
    sns.histplot(class_1_values, color='green', label='1', kde=False, bins=50, alpha=0.5)
    plt.title('Distribution of Landmark Distances by Posture')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend(title='Posture')
    plt.show()


# # Calculate statistics
# stats_0, stats_1 = calculate_neck_angle_statistics(df)
# average_0, std_dev_0 = stats_0
# average_1, std_dev_1 = stats_1
#
# # Print results
# print(f"Average Neck Angle for classification 0: {average_0}, Standard Deviation: {std_dev_0}")
# print(f"Average Neck Angle for classification 1: {average_1}, Standard Deviation: {std_dev_1}")
#
# graph_distances()

def plot_neck_angle_boxplot(df):
    if "Posture" not in df.columns or "Neck Angle" not in df.columns:
        raise ValueError("CSV file does not have the expected structure.")

    # Create a melted DataFrame for easier plotting (if needed)
    # Since 'Neck Angle' is already in the DataFrame, no need to melt in this case.

    # Calculate and print 25th percentile, 75th percentile, and IQR for each posture
    percentiles = df.groupby('Posture')['Neck Angle'].quantile([0.25, 0.75]).unstack()
    percentiles.columns = ['25th Percentile', '75th Percentile']
    percentiles['IQR'] = percentiles['75th Percentile'] - percentiles['25th Percentile']

    print("Neck Angle Percentiles and IQR by Posture:")
    print(percentiles)

    # Plot the box plot
    plt.figure(figsize=(4.5, 6))
    sns.boxplot(x='Posture', y='Neck Angle', data=df)
    plt.title('Neck Angles by Posture')
    plt.xlabel('Posture')
    plt.ylabel('Neck Angle (Degrees)')
    plt.show()


def plot_angle_boxplot(df, col_name, angle_name):
    if "Posture" not in df.columns or col_name not in df.columns:
        raise ValueError("CSV file does not have the expected structure.")

    # Calculate 25th percentile, 75th percentile, IQR, mean, and standard deviation for each posture
    percentiles = df.groupby('Posture')[col_name].quantile([0.25, 0.75]).unstack()
    percentiles.columns = ['25th Percentile', '75th Percentile']
    # percentiles['IQR'] = percentiles['75th Percentile'] - percentiles['25th Percentile']

    # Calculate the mean for each posture
    mean_values = df.groupby('Posture')[col_name].mean()
    percentiles['Mean'] = mean_values

    # Calculate the standard deviation for each posture
    std_dev = df.groupby('Posture')[col_name].std()
    percentiles['Standard Deviation'] = std_dev

    print(f"{angle_name} Statistics by Posture:")
    print(percentiles)

    # Plot the box plot
    plt.figure(figsize=(4.5, 6))
    sns.boxplot(x='Posture', y=col_name, data=df)
    plt.title(f'{angle_name}s by Posture')
    plt.xlabel('Posture')
    plt.ylabel(f'{angle_name} (Degrees)')
    plt.ylim(104, 181)
    plt.show()


# # plot_neck_angle_boxplot(df)
#
plot_distance_boxplot(df)
# plot_angle_boxplot(df, 'Torso_Vertical_Angle', 'Back to Vertical Vector Angle')
# plot_angle_boxplot(df, 'Neck_Vertical_Angle', 'Neck to Vertical Vector Angle')
# plot_angle_boxplot(df, 'Neck_Torso_Angle', 'Neck to Torso Vector Angle')
#
# # plot_correlation_heatmap(df)
#
# fpr, tpr, thresholds = roc_curve(df['Posture'], -df['Neck_Torso_Angle'])
# roc_auc = auc(fpr, tpr)
# # Plot ROC curve
# plt.figure(figsize=(6, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
#
# youden_index = tpr - fpr
#
# # Find the index of the maximum Youden Index
# max_index = np.argmax(youden_index)
#
# # Find the best threshold
# best_threshold = thresholds[max_index]
#
# # Sensitivity (TPR) and Specificity (1 - FPR) at the best threshold
# sensitivity = tpr[max_index]
# specificity = 1 - fpr[max_index]
#
# print(f"Best Threshold: {best_threshold}")
# print(f"Maximum Youden Index: {youden_index[max_index]}")
# print(f"Sensitivity at Best Threshold: {sensitivity}")
# print(f"Specificity at Best Threshold: {specificity}")
#
# plt.show()