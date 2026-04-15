import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv('data/job_salary_prediction_dataset.csv')

# Remove missing values
data = data.dropna()

# Target and features
y = data['salary']
X = data.drop('salary', axis=1)

# One-hot encoding
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

# Convert R2 into percentage for UI
accuracy_percent = round(r2 * 100, 2)

# Save everything
os.makedirs('model', exist_ok=True)

with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)

with open('model/metrics.pkl', 'wb') as f:
    pickle.dump({
        'mae': float(mae),
        'r2': float(r2),
        'accuracy_percent': accuracy_percent
    }, f)

print("Model, columns and metrics saved successfully!")