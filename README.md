# PATIENT_DATASET_ZOBIA-AHMED-330-LAB-05
HOME TASK CODE:
#ZOBIA AHMED / 2022F-BSE-330 / LAB 05 / HOMETASK:
# Importing necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
# Load the dataset
df = pd.read_csv("patient_data_zobia.csv")
# Encoding categorical variables
df['Family History'] = df['Family History'].map({'No': 0, 'Yes': 1})
df['Diet Type'] = df['Diet Type'].map({'Balanced': 0, 'High Sugar': 1, 'Low Carb': 2})
df['Category'] = df['Category'].map({'Healthy': 0, 'Pre-Diabetic': 1, 'Diabetic': 2})
# Splitting the data into training and testing sets (first 30 for training, last 10 for testing)
train_data = df.iloc[:30]
test_data = df.iloc[30:]
# Separate features and target variable
X_train = train_data.drop(['Patient ID', 'Category'], axis=1)
y_train = train_data['Category']
X_test = test_data.drop(['Patient ID', 'Category'], axis=1)
y_test = test_data['Category']
# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
# Add the predicted values to the test data
test_data_with_predictions = test_data.copy()
test_data_with_predictions['Predicted Category'] = y_pred
print("ZOBIA AHMED / 2022F-BSE-330 / LAB 05 / HOMETASK:\n")
# Display the full test data with actual and predicted values
print("(1): Test Data Of Last 10 Rows With Predictions:")
print(test_data_with_predictions)
# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n(2): Accuracy of the model:", accuracy)
print("(3): Confusion Matrix:\n", conf_matrix)
