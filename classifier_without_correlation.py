import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
import time

# Load data from CSV files
train_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\final\training_continuous_3-5-9.csv')
test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\final\testing_continuous_3-5-9.csv')

# Feature selection
id_to_exclude_side_data_for_training_testing = -4
train_data_front = train_data.iloc[:, 1:id_to_exclude_side_data_for_training_testing]

# Calculate the correlation matrix
correlation_matrix = train_data_front.corr().abs()

# Select upper triangle of correlation matrix
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.99
threshold = 0.988
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

# Drop highly correlated features
selected_features = [column for column in train_data_front.columns if column not in to_drop]

print(f"Selected {len(selected_features)} features:", selected_features)

# Separate the features (X) and the target variable (y) in the training data
X_train = train_data[selected_features]
y_train = train_data.iloc[:, -1]

# Separate the features (X) and the target variable (y) in the testing data
X_test = test_data[selected_features]
y_test = test_data.iloc[:, -1]

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVC': SVC(probability=True),
    'Random Forest': RandomForestClassifier(max_depth=3),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Create a figure for plotting
plt.figure(figsize=(8, 8))

# Loop through models
for name, model in models.items():
    # Record the start time for training
    start_train_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # Record the training time
    train_time = time.time() - start_train_time

    # Record the start time for prediction
    start_pred_time = time.time()

    # Predict probabilities for the test set
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Record the prediction time
    pred_time = time.time() - start_pred_time

    # Make class predictions
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Print the metrics
    print(f'{name}:')
    print(f'  Training time: {train_time:.4f} seconds')
    print(f'  Prediction time: {pred_time:.4f} seconds')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  AUC: {roc_auc:.4f}\n')

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot diagonal line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Configure plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
