import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

filename = "continuous_data"
# Load the training and unlabeled data
train_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\all_train_data.csv')
unlabeled_data = pd.read_csv(fr'C:\Users\bokch\PyCharm\Ergonomics\data\{filename}.csv')

# Separate the features (X) and the target variable (y) in the training data
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Separate the features (X) in the unlabeled data
X_test = unlabeled_data.iloc[:, :]

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the training set for validation
y_pred_train = model.predict(X_train)

# Calculate the accuracy of the model on training data
accuracy = accuracy_score(y_train, y_pred_train)

print(f'Training accuracy: {accuracy:.10f}')

# Make predictions on the unlabeled data
y_pred_unlabeled = model.predict(X_test)

# Assign the predicted labels to the unlabeled data in a new column "Posture"
unlabeled_data['Posture'] = y_pred_unlabeled

# Save the labeled data
unlabeled_data.to_csv(fr'C:\Users\bokch\PyCharm\Ergonomics\data\{filename}_labeled.csv', index=False)

print(f"\nLabeled data with 'Posture' column saved to '{filename}_labeled.csv'")
