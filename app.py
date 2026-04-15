from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load columns used during training
with open('model/columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# Model accuracy from training
MODEL_ACCURACY = 95


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Map frontend values to dataset values
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

        # Count entered skills
        skill_count = len([
            skill.strip()
            for skill in data['skills'].split(',')
            if skill.strip()
        ])

        # Create input matching training dataset
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

        # Convert to dataframe
        df = pd.DataFrame(input_data)

        # Apply one-hot encoding like training
        df = pd.get_dummies(df)

        # Add missing columns
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns exactly like training
        df = df.reindex(columns=model_columns, fill_value=0)

        # Predict salary
        prediction = model.predict(df)[0]

        # Scale prediction because dataset values are smaller than INR range
        predicted_salary = float(prediction) * 10

        # Prevent invalid values
        if predicted_salary < 0:
            predicted_salary = 0

        # Create salary range ±10%
        min_salary = predicted_salary * 0.9
        max_salary = predicted_salary * 1.1

        return jsonify({
            "predicted_salary": round(predicted_salary, 2),
            "min_salary": round(min_salary, 2),
            "max_salary": round(max_salary, 2),
            "accuracy": MODEL_ACCURACY
        })

    except Exception as e:
        return jsonify({
            "message": f"Prediction failed: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)