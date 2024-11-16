import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # Changed import
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\training_data.csv')
test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\testing_data.csv')

# Separate the features (X) and the target variable (y) in the training data
X_train = train_data.iloc[:, :-2]
y_train = train_data.iloc[:, -2]  # Second to last column

# Separate the features (X) and the target variable (y) in the testing data
X_test = test_data.iloc[:, :-2]
y_test = test_data.iloc[:, -2]  # Second to last column
posture_test = test_data.iloc[:, -1]
# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Example with 100 trees

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)



# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.10f}')


# sns.scatterplot(x=y_test, y=y_pred, hue=posture_test, alpha=0.2)
#
# # Add labels and title
# plt.xlabel('Actual Angle')
# plt.ylabel('Predicted Angle')
# plt.title('Scatter Plot')
#
# plt.xlim(110, 170)
# plt.ylim(110, 170)
#
# plt.show()



# Calculate the absolute errors
errors = np.abs(y_test - y_pred)

# Get the indices of the 100 largest errors
largest_error_indices = np.argsort(errors)[-100:]

# Get the indices of the 100 smallest errors
smallest_error_indices = np.argsort(errors)[:100]

# Print the predictions and actual values for the 100 largest errors
print("\nPredictions and actual values for the 100 largest errors:")
for index in largest_error_indices:
    print(f"Row: {index}, Actual: {y_test.iloc[index]}, Prediction: {y_pred[index]}, Error: {errors[index]}")

# Print the predictions and actual values for the 100 smallest errors
print("\nPredictions and actual values for the 100 smallest errors:")
for index in smallest_error_indices:
    print(f"Row: {index}, Actual: {y_test.iloc[index]}, Prediction: {y_pred[index]}, Error: {errors[index]}")


plt.figure(figsize=(8, 6))
sns.histplot(data=test_data, x='Neck Angle', hue='Posture', palette='Set1', bins=20, alpha=0.5, multiple='layer')
plt.title('Histogram with Two Groups')
plt.xlabel('Data')
plt.ylabel('Frequency')
plt.legend(title='Group')
plt.grid(False)
plt.show()
