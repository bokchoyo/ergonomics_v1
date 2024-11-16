import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\001_003_training_data.csv')
test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\0002_training_data.csv')

# Separate the features (X) and the target variable (y) in the training data
X_train = train_data.iloc[:, :-2]
y_train = train_data.iloc[:, -2]  # Second to last column

# Separate the features (X) and the target variable (y) in the testing data
X_test = test_data.iloc[:, :-2]
y_test = test_data.iloc[:, -2]  # Second to last column

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.10f}')

# Calculate the absolute errors
errors = np.abs(y_test - y_pred)

# Get the indices of the 50 largest errors
largest_error_indices = np.argsort(errors)[-100:]

# Get the indices of the 50 smallest errors
smallest_error_indices = np.argsort(errors)[:100]

# Print the predictions and actual values for the 50 largest errors
print("\nPredictions and actual values for the 100 largest errors:")
for index in largest_error_indices:
    print(f"Row: {index}, Actual: {y_test.iloc[index]}, Prediction: {y_pred[index]}, Error: {errors[index]}")

# Print the predictions and actual values for the 50 smallest errors
print("\nPredictions and actual values for the 100 smallest errors:")
for index in smallest_error_indices:
    print(f"Row: {index}, Actual: {y_test.iloc[index]}, Prediction: {y_pred[index]}, Error: {errors[index]}")
