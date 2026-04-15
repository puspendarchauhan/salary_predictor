import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# STEP 1: Load dataset
data = pd.read_csv('data/job_salary_prediction_dataset.csv')

print("Dataset Preview:\n", data.head())

# STEP 2: Drop missing values
data = data.dropna()

# STEP 3: Separate target
y = data['salary']
X = data.drop('salary', axis=1)

# STEP 4: One-Hot Encoding (VERY IMPORTANT)
X = pd.get_dummies(X)

# STEP 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)
# STEP 7: Prediction
y_pred = model.predict(X_test)

# STEP 8: Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("R2 Score:", r2)


import os

# Create model folder if it doesn't exist
os.makedirs('model', exist_ok=True)

import pickle

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save columns (important for prediction)
with open('model/columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)

print("\nModel saved successfully!")

