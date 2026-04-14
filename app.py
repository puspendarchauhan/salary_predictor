from flask import Flask, render_template, request
import random

app = Flask(__name__)

# Dummy prediction + confidence
def predict_salary(experience):
    salary = 30000 + (experience * 5000)
    confidence = round(random.uniform(85, 98), 2)  # fake confidence %
    return round(salary, 2), confidence


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        job_role = request.form['job_role']
        gender = request.form['gender']
        experience = float(request.form['experience'])

        # Validation
        if experience < 0:
            return render_template('index.html',
                                   prediction_text="❌ Experience cannot be negative!")

        salary, confidence = predict_salary(experience)

        result = f"{name.upper()}, your predicted salary is ₹ {salary} \nModel Confidence: {confidence}%"

        return render_template('index.html', prediction_text=result)

    except:
        return render_template('index.html',
                               prediction_text="⚠️ Error occurred! Check inputs.")


if __name__ == "__main__":
    app.run(debug=True)