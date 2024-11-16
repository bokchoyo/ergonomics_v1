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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load data from a single CSV file
data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\continuous_data_labeled.csv')

# Calculate the correlation matrix
correlation_matrix = data.iloc[:, 1:-2].corr().abs()  # Exclude the first and second to last columns

# Select upper triangle of correlation matrix
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.8
threshold = 0.99
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

# Drop highly correlated features
selected_features = [column for column in data.columns[1:-2] if column not in to_drop]  # Exclude the first and second to last columns

# Print the names of the columns that are kept
print(f"Selected {len(selected_features)} features:", selected_features)

# Separate the features (X) and the target variable (y)
X = data[selected_features]
y = data.iloc[:, -1]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the classifiers with their respective colors
classifiers = {
    'Logistic Regression': {'model': LogisticRegression(max_iter=1000), 'color': '#4285F4'},
    'Support Vector Machine': {'model': SVC(probability=True), 'color': '#EA4335'},
    'Random Forest': {'model': RandomForestClassifier(), 'color': '#FBBC05'},
    'XGBoost': {'model': XGBClassifier(), 'color': '#34A853'}
}

# Plot ROC curves
plt.figure(figsize=(8, 10))  # Square dimensions
for name, clf_info in classifiers.items():
    model = clf_info['model']
    color = clf_info['color']

    # Measure the training time in milliseconds
    start_train_time = time.time()
    model.fit(X_train, y_train)
    end_train_time = time.time()
    training_time = (end_train_time - start_train_time) * 1000
    print(f'{name} Training time: {training_time:.8f} milliseconds')

    # Measure the prediction time in milliseconds
    start_test_time = time.time()
    y_pred = model.predict(X_test)
    end_test_time = time.time()
    testing_time = (end_test_time - start_test_time) * 1000
    print(f'{name} Testing time: {testing_time:.8f} milliseconds')

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.8f}')

    # Predict probabilities for the test set
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f'{name} AUC: {roc_auc:.8f}')

    # Plot ROC curve
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.8f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()
