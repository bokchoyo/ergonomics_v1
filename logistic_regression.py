import pandas as pd
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV files
train_data = pd.read_csv(r'/data/001_003_training_data.csv')
test_data = pd.read_csv(r'/data/0002_training_data.csv')

# df = train_data
# # Plot a histogram using seaborn
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='Left_Shoulder_To_Landmark_2_Distance', hue='Posture', palette='Set2')
#
# # Add title and labels
# plt.title('Histogram of Data Column Colored by Group')
# plt.xlabel('Data Column')
# plt.ylabel('Frequency')
#
# # Show the plot
# plt.show()

id_col_sel = [22, 24]
# Separate the features (X) and the target variable (y) in the training data
X_train = train_data.iloc[:, id_col_sel]
y_train = train_data.iloc[:, -1]

# Separate the features (X) and the target variable (y) in the testing data
X_test = test_data.iloc[:, id_col_sel]
y_test = test_data.iloc[:, -1]

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

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.10f}')

# Optional: Print the coefficients of the model
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')


# iris = load_iris()
# disp = DecisionBoundaryDisplay.from_estimator(
#     model, X_test, response_method="predict",
#     xlabel=test_data.columns[id_col_sel[0]], ylabel=test_data.columns[id_col_sel[1]],
#     alpha=0.5,
# )
# disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
# plt.show()
