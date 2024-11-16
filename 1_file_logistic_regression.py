import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data from CSV file
data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\training_data.csv')

# Separate the features (X) and the target variable (y)
# X = data.iloc[:, :-1]  # all columns except the last one
X = data.iloc[:, 4:5]
y = data.iloc[:, -1]   # the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.columns)
print(X_test.shape)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# print("y_pred: \n", y_pred)
# print("y_test: \n", y_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.10f}')

# Optional: Print the coefficients of the model
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
