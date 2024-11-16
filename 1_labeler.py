import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Load data from CSV files
unlabeled_data_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\final\training_continuous_all.csv'
train_data = pd.read_csv(r"C:\Users\bokch\PyCharm\Ergonomics\data\final\training_discrete.csv")
# # unlabeled_test_data = pd.read_csv(r"C:\Users\bokch\PyCharm\Ergonomics\data\set_testing_2-2.csv")
# unlabeled_test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\3_people_test_data.csv')
unlabeled_test_data = pd.read_csv(unlabeled_data_path)
# Exclude the first column for training and labeling
# Calculate the correlation matrix
train_data_without_frame_and_posture = train_data.iloc[:, 1:-1]
correlation_matrix = train_data_without_frame_and_posture.corr().abs()  # Exclude the first and columns

# Select upper triangle of correlation matrix
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.8
threshold = 1
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

# Drop highly correlated features
selected_features = [column for column in train_data.columns[1:-1] if column not in to_drop]  # Exclude only the first column

# Print the names of the columns that are kept
print(f"Selected {len(selected_features)} features:", selected_features)

# Separate the features (X) and the target variable (y) in the training data
X_train = train_data[selected_features]
y_train = train_data.iloc[:, -1]  # Assuming the first column is the target variable

# Separate the features (X) in the unlabeled test data
X_unlabeled_test = unlabeled_test_data[selected_features]

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_unlabeled_test_scaled = scaler.transform(X_unlabeled_test)

print("Initializing Model...")
# Initialize the random forest classifier model
model = XGBClassifier()
print("Initialized Model.")

print("Training Model...")
# Train the model
model.fit(X_train_scaled, y_train)
print("Trained Model.")

print("Making Predictions...")
# Make predictions on the unlabeled test set
y_unlabeled_pred = model.predict(X_unlabeled_test_scaled)
print("Made Predictions.")

print("Writing Posture Column")
# Add the "Posture" column to the unlabeled test data
unlabeled_test_data['Posture'] = y_unlabeled_pred

# Save the labeled test data to a new CSV file
unlabeled_test_data.to_csv(unlabeled_data_path, index=False)

print("Labeled test data saved with 'Posture' column.")

#
# y_prob = model.predict_proba(X_train_scaled)[:, 1]  # Probability of the positive class
#
# # Compute ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(train_data['Posture'], y_prob)
# roc_auc = auc(fpr, tpr)
#
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
