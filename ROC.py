import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb

# Load data from CSV files
train_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\training_set.csv')
test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\testing_set.csv')

# Separate the features (X) and the target variable (y) in the training data
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Separate the features (X) and the target variable (y) in the testing data
X_test = test_data.iloc[:, :-1]
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
plt.figure(figsize=(10, 8))

# Loop through models
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')

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
