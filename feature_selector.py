import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the training and testing data
train_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\training_data.csv')
test_data = pd.read_csv(r'C:\Users\bokch\PyCharm\Ergonomics\data\testing_data.csv')

# Get the number of columns, excluding the last one (target variable)
num_columns = train_data.shape[1] - 1

# Initialize a dictionary to store accuracies for each column
accuracies = {}

# Iterate through each column except the last one
for i in range(num_columns):
    # Separate the features (X) and the target variable (y) in the training data
    X_train = train_data.iloc[:, [i]]
    y_train = train_data.iloc[:, -1]

    # Separate the features (X) and the target variable (y) in the testing data
    X_test = test_data.iloc[:, [i]]
    y_test = test_data.iloc[:, -1]

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the random forest classifier model
    model = RandomForestClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Column {i} Accuracy: ", accuracy)

    # Store the accuracy in the dictionary
    accuracies[f'Column_{i}'] = accuracy

# Sort accuracies from highest to lowest
sorted_accuracies = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)

# Select top 10 most accurate columns
top_columns = [int(col.split('_')[1]) for col, acc in sorted_accuracies[:20]]

# Separate the features (X) and the target variable (y) in the training data using selected columns
X_train_selected = train_data.iloc[:, top_columns]
y_train = train_data.iloc[:, -1]

# Separate the features (X) and the target variable (y) in the testing data using selected columns
X_test_selected = test_data.iloc[:, top_columns]
y_test = test_data.iloc[:, -1]

# Normalize the features
scaler = StandardScaler()
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)

# Initialize the random forest classifier model
model = RandomForestClassifier()

# Train the model
model.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred_selected = model.predict(X_test_selected)

# Calculate the accuracy of the model
accuracy_selected = accuracy_score(y_test, y_pred_selected)

# Print accuracy
print(f'Accuracy using top 10 columns: {accuracy_selected:.10f}')
