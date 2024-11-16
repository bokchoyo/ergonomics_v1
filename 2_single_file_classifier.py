import math
import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier  # Import XGBClassifier from XGBoost
from lightgbm import LGBMClassifier

# Load data from a single CSV file
data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\cont_test_labeled.csv')


# Drop highly correlated features
correlation_matrix = data.iloc[:, 1:-2].corr().abs()  # Exclude the first and second to last columns

# Select upper triangle of correlation matrix
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

upper_triangle.to_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\correlation_matrix.csv', index=True)
# Find features with correlation greater than 0.8
threshold = 0.99
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

# Drop highly correlated features
selected_features = [column for column in data.columns[1:-2] if column not in to_drop]  # Exclude the first and second to last columns
# 8 selected_features = ['Right_Shoulder_To_Landmark_1_Distance', 'Left_Shoulder_To_Landmark_1_Distance',
# 'Right_Shoulder_To_Landmark_7_Distance', 'Right_Shoulder_To_Landmark_8_Distance',
# 'Right_Shoulder_To_Landmark_9_Distance', 'Left_Shoulder_To_Landmark_10_Distance',
# 'Left_Shoulder_To_Landmark_11_Distance', 'Left_Shoulder_To_Landmark_12_Distance']


# Print the names of the columns that are kept
print(f"Selected {len(selected_features)} features:", selected_features)

selected_correlation_matrix = data[selected_features].corr().abs()
new_column_names = [col[:-9] for col in selected_correlation_matrix.columns]
selected_correlation_matrix.columns = new_column_names
selected_correlation_matrix.index = new_column_names
# Plot the heatmap
plt.figure(figsize=(8.15, 8.15))
sns.heatmap(selected_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            square=True, annot_kws={"fontsize": 6.4})
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Correlation Heatmap of Selected Features')
plt.tight_layout()  # Adjust the layout to ensure everything fits
plt.show()
# Separate the features (X) and the target variable (y)
X = data[selected_features]
y = data.iloc[:, -1]
neck_angles = data["Neck Angle"]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the XGBoost classifier model
model = LogisticRegression()

start_train_time = time.time()
model.fit(X_train, y_train)
end_train_time = time.time()
training_time = (end_train_time - start_train_time) * 1000
print(f'Training time: {training_time:.3f} milliseconds')

start_test_time = time.time()
y_pred = model.predict(X_test)
end_test_time = time.time()
testing_time = (end_test_time - start_test_time) * 1000
print(f'Prediction time: {testing_time:.3f} milliseconds')

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.8f}')

# Predict probabilities for the test set
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc:.8f}')

# conf_matrix = confusion_matrix(y_test, y_pred)
# conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum()
#
# # Labels for annotating cells
# labels = np.array([
#     [f"True Neg\n{conf_matrix[0, 0]}\n{conf_matrix_percent[0, 0]:.2%}",
#      f"False Pos\n{conf_matrix[0, 1]}\n{conf_matrix_percent[0, 1]:.2%}"],
#     [f"False Neg\n{conf_matrix[1, 0]}\n{conf_matrix_percent[1, 0]:.2%}",
#      f"True Pos\n{conf_matrix[1, 1]}\n{conf_matrix_percent[1, 1]:.2%}"]
# ])
#
# # Define colors for heatmap
# colors = ['darkblue', 'lightblue']
#
# # Plot confusion matrix with custom colors
# plt.figure(figsize=(8, 8))
# sns.heatmap(conf_matrix, annot=labels, fmt='', cmap=colors, cbar=False, annot_kws={"size": 16})
# plt.xlabel('Predicted Posture')
# plt.ylabel('Actual Posture')
# plt.title('Confusion Matrix')
# plt.show()

# # Plot ROC curve
# plt.figure(figsize=(8, 8))  # Square dimensions
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.8f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.show()
#
# df_for_plot = pd.DataFrame()
# df_for_plot['neck_angles']= data['Neck Angle']
# df_for_plot['posture'] = data['Posture']
# df_for_plot['frame'] = data['Frame']
# # Plot neck angles (if applicable)
# test_neck_angles = neck_angles  # Assuming the neck angles are the last column of X_test
# plt.figure(figsize=(10, 10))  # Square dimensions
# #plt.plot(test_neck_angles)
# #import seaborn as sns
# sns.scatterplot(data = df_for_plot, x = 'frame', y= 'neck_angle', hue = 'posture')
# plt.title('Neck Angles')
# plt.show()
