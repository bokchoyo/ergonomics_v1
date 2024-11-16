import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV files
train_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\training_data.csv')
test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\testing_data.csv')

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

# Initialize the random forest classifier model
model = RandomForestClassifier()

# Apply Recursive Feature Elimination (RFE)
rfe = RFE(estimator=model, n_features_to_select=10)  # Adjust the number of features to select as needed
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Train the model with the selected features
model.fit(X_train_rfe, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_rfe)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.10f}')

# Plot a histogram using seaborn (optional)
# df = train_data
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='Left_Shoulder_To_Landmark_2_Distance', hue='Posture', palette='Set2')
# plt.title('Histogram of Data Column Colored by Group')
# plt.xlabel('Data Column')
# plt.ylabel('Frequency')
# plt.show()

# Optional: Plot the decision boundary (not as straightforward with Random Forests)
# Uncomment and modify this section if needed
# iris = load_iris()
# disp = DecisionBoundaryDisplay.from_estimator(
#     model, X_test, response_method="predict",
#     xlabel=test_data.columns[id_col_sel[0]], ylabel=test_data.columns[id_col_sel[1]],
#     alpha=0.5,
# )
# disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
# plt.show()
