import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC

# Load data
# Use the script's directory to find the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data.csv')
data = pd.read_csv(csv_path)

# Reorder columns as requested
new_order = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
    'Credit_History', 'Property_Area', 'Loan_Status', 
    'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed'
]
data = data[new_order]

print(type(data))
print(data.head())
print(data.shape)
print(data.info())

# It is better to identify object columns first
obj_cols = data.select_dtypes(include=['object']).columns
for col in obj_cols:
    data[col] = data[col].astype('category')

print(data.info())

cat_cols = data.select_dtypes(include=['category']).columns
print("Categorical columns:", cat_cols)


# Fill numerical columns with mean
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
print("Numerical columns:", num_cols)
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

# Fill categorical columns with mode
for col in cat_cols:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].mode()[0])

print("Null values after filling:")
print(data.isnull().sum())


# Note: The column name in csv is 'LoanAmount', not 'Loan'
X = data[['ApplicantIncome', 'LoanAmount']]
y = data['Loan_Status']

# Encode target variable
print("Target value counts:")
print(y.value_counts())
y = y.map({'Y': 1, 'N': 0})

# Train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


cat_cols=data.select_dtypes(include=['category']).columns
print(cat_cols)


print(type(data['Gender'] ))
print(data.info())

model = SVC()
model.fit(X_train, y_train)
y_pred_SVM = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred_SVM)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")