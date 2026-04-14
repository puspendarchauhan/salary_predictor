import pickle
import pandas as pd

# Load model
model = pickle.load(open('model/model.pkl', 'rb'))
columns = pickle.load(open('model/columns.pkl', 'rb'))

# Sample input (you can change values)
input_data = {
    'job_title': 'Data Scientist',
    'experience_years': 5,
    'education_level': 'Master',
    'skills_count': 10,
    'industry': 'IT',
    'company_size': 'Medium',
    'location': 'India',
    'remote_work': 'Yes',
    'certifications': 2
}

# Convert to DataFrame
df = pd.DataFrame([input_data])

# One-hot encoding
df = pd.get_dummies(df)

# Match training columns
df = df.reindex(columns=columns, fill_value=0)

# Predict
prediction = model.predict(df)

print("Predicted Salary:", prediction[0])