import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_data = pd.read_csv(r'/data/001_003_training_data.csv')
test_data = pd.read_csv(r'/data/0002_training_data.csv')

# Optional: Plot a histogram of a selected feature colored by posture
# df = train_data
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='Left_Shoulder_To_Landmark_2_Distance', hue='Posture', palette='Set2')
# plt.title('Histogram of Data Column Colored by Group')
# plt.xlabel('Data Column')
# plt.ylabel('Frequency')
# plt.show()

id_col_sel = [22, 24]

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

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.10f}')

# Calculate the absolute errors
errors = np.abs(y_test - y_pred)

# Get the indices of the 20 largest errors
largest_error_indices = np.argsort(errors)[-50:]

# Get the indices of the 20 smallest errors
smallest_error_indices = np.argsort(errors)[:50]

# Print the predictions and actual values for the 20 largest errors
print("\nPredictions and actual values for the 20 largest errors:")
for index in largest_error_indices:
    print(f"Actual: {y_test.iloc[index]}, Prediction: {y_pred[index]}, Error: {errors[index]}")

# Print the predictions and actual values for the 20 smallest errors
print("\nPredictions and actual values for the 20 smallest errors:")
for index in smallest_error_indices:
    print(f"Actual: {y_test.iloc[index]}, Prediction: {y_pred[index]}, Error: {errors[index]}")
