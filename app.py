
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates')
)

# Absolute paths for model files
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'model', 'columns.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'model', 'metrics.pkl')

# Load model files
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(COLUMNS_PATH, 'rb') as f:
    model_columns = pickle.load(f)

with open(METRICS_PATH, 'rb') as f:
    metrics = pickle.load(f)

MODEL_ACCURACY = metrics['accuracy_percent']
MODEL_R2 = round(metrics['r2'], 4)
MODEL_MAE = round(metrics['mae'], 2)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        education_map = {
            "High School": "High School",
            "Associate Degree": "Associate",
            "Bachelor's Degree": "Bachelor",
            "Master's Degree": "Master",
            "MBA": "MBA",
            "PhD / Doctorate": "PhD"
        }

        company_size_map = {
            "Startup (1–50)": "Small",
            "Small (51–200)": "Small",
            "Mid-size (201–1000)": "Medium",
            "Large (1001–5000)": "Large",
            "Enterprise (5000+)": "Enterprise"
        }

        industry_map = {
            "Technology": "Technology",
            "Finance & Banking": "Finance",
            "Healthcare": "Healthcare",
            "E-Commerce": "Retail",
            "Education": "Education",
            "Manufacturing": "Manufacturing",
            "Consulting": "Consulting",
            "Media & Entertainment": "Media",
            "Retail": "Retail",
            "Government": "Government"
        }

        skill_count = len([
            skill.strip()
            for skill in data['skills'].split(',')
            if skill.strip()
        ])

        input_data = {
            'job_title': [data['job_role']],
            'experience_years': [float(data['experience'])],
            'education_level': [education_map.get(data['education'], data['education'])],
            'skills_count': [skill_count],
            'industry': [industry_map.get(data['industry'], data['industry'])],
            'company_size': [company_size_map.get(data['company_size'], data['company_size'])],
            'location': [data['city']],
            'remote_work': ['No'],
            'certifications': [1 if skill_count >= 3 else 0]
        }

        df = pd.DataFrame(input_data)
        df = pd.get_dummies(df)

        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        df = df.reindex(columns=model_columns, fill_value=0)

        prediction = float(model.predict(df)[0])

        if prediction < 100000:
            prediction *= 10

        if prediction < 0:
            prediction = 0

        return jsonify({
            'predicted_salary': round(prediction, 2),
            'monthly_salary': round(prediction / 12, 2),
            'min_salary': round(prediction * 0.9, 2),
            'max_salary': round(prediction * 1.1, 2),
            'accuracy': MODEL_ACCURACY,
            'r2': MODEL_R2,
            'mae': MODEL_MAE
        })

    except Exception as e:
        return jsonify({
            'message': str(e)
        }), 500
if __name__ == '__main__':
    app.run(debug=True)