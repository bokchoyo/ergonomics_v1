import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV files
train_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\continuous_data_labeled.csv')
# test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\testing_data.csv')
#
#
# id_col_sel = [24, 25]
# # Separate the features (X) and the target variable (y) in the training data
# X_train = train_data.iloc[:, id_col_sel]
# y_train = train_data.iloc[:, -1]
#
# # Separate the features (X) and the target variable (y) in the testing data
# X_test = test_data.iloc[:, id_col_sel]
# y_test = test_data.iloc[:, -1]
#
# # Normalize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # Initialize the random forest classifier model
# model = RandomForestClassifier()
#
# # Train the model
# model.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = model.predict(X_test)
#
# # Calculate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.10f}')

df = train_data
# Plot a histogram using seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Neck Angle', hue='Posture', palette={0: 'red', 1: 'green'}, alpha=0.4)

# Add title and labels
plt.title('Continuous Video Neck Angles')
plt.xlabel('Neck Angle')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Optional: Plot the decision boundary (not as straightforward with Random Forests)
# Uncomment and modify this section if needed
# disp = DecisionBoundaryDisplay.from_estimator(
#     model, X_test, response_method="predict",
#     xlabel=test_data.columns[id_col_sel[0]], ylabel=test_data.columns[id_col_sel[1]],
#     alpha=0.5,
# )
# disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
# plt.show()

# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)

# Display the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()


# plt.figure(figsize=(12, 10))
# train_data_select = train_data.iloc[:, 1:10]
# correlation_matrix = train_data_select.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix Heatmap')
# plt.show()
#
# train_data_for_plot = pd.concat([pd.DataFrame(train_data_select, columns=train_data.columns[:-1]), y_train.reset_index(drop=True)], axis=1)
#
# sns.pairplot(train_data_for_plot, hue='Posture', diag_kind='hist')
# plt.show()
