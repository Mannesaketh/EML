import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('data.csv')
print(type(data))
print(data.head())
print(data.shape)
print(data.info())

# Change object data type to category
# It is better to identify object columns first
obj_cols = data.select_dtypes(include=['object']).columns
for col in obj_cols:
    data[col] = data[col].astype('category')

print(data.info())

cat_cols = data.select_dtypes(include=['category']).columns
print("Categorical columns:", cat_cols)

# Fill null values
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Predictions:", y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
