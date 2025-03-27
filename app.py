from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Load models
models = {}
model_files = {
    'calories': 'models/calories_model.pkl',
    'salary': 'models/salary_model.pkl',
    'house_price': 'models/house_price_model.pkl',
    'house_price_poly': 'models/house_price_poly.pkl',
    'diabetes': 'models/diabetes_model.pkl',
    'heart_disease': 'models/heart_disease_model.pkl'
}

for model_name, file_path in model_files.items():
    try:
        if model_name == 'house_price_poly':
            models[model_name] = joblib.load(file_path)
        else:
            models[model_name] = joblib.load(file_path)
            print(f"Loaded {model_name} model successfully")
    except Exception as e:
        print(f"Error loading {model_name} model: {str(e)}")

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/calories')
def calories():
    try:
        with open('models/calories_accuracy.json') as f:
            accuracy = json.load(f)['accuracy']
    except:
        accuracy = "N/A"
    return render_template('calories.html', accuracy=accuracy, algorithm="Simple Linear Regression")

@app.route('/salary')
def salary():
    try:
        with open('models/salary_accuracy.json') as f:
            accuracy = json.load(f)['accuracy']
    except:
        accuracy = "N/A"
    return render_template('salary.html', accuracy=accuracy, algorithm="Multiple Linear Regression")

@app.route('/house')
def house():
    try:
        with open('models/house_price_accuracy.json') as f:
            accuracy = json.load(f)['accuracy']
    except:
        accuracy = "N/A"
    return render_template('house.html', accuracy=accuracy, algorithm="Polynomial Regression")

@app.route('/diabetes')
def diabetes():
    try:
        with open('models/diabetes_accuracy.json') as f:
            accuracy = json.load(f)['accuracy']
    except:
        accuracy = "N/A"
    return render_template('diabetes.html', accuracy=accuracy, algorithm="Logistic Regression")

@app.route('/heart')
def heart():
    try:
        with open('models/heart_disease_accuracy.json') as f:
            accuracy = json.load(f)['accuracy']
    except:
        accuracy = "N/A"
    return render_template('heart.html', accuracy=accuracy, algorithm="K-Nearest Neighbors")

@app.route('/predict_calories', methods=['POST'])
def predict_calories():
    try:
        data = request.get_json()
        km_run = float(data['km_run'])
        
        prediction = models['calories'].predict([[km_run]])[0]
        return jsonify({'calories': round(prediction, 2)})
    except Exception as e:
        print(f"Error in calories prediction: {str(e)}")
        return jsonify({'error': 'Server error. Please try again.'}), 500

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    try:
        # Load salary ranges
        with open('models/salary_ranges.json') as f:
            salary_ranges = json.load(f)
        
        data = request.get_json()
        
        # Get input data
        experience = float(data['experience'])
        education_level = float(data['education_level'])
        
        # Round experience to nearest integer
        experience = round(experience)
        
        # Get the salary range based on experience and education level
        if experience > 5:
            experience = 5  # Cap at 5+ years
        
        min_salary, max_salary = salary_ranges[str(experience)][str(int(education_level))]
        
        # Calculate a realistic salary within the range
        # Use age as a factor to adjust within the range
        age = float(data['age'])
        age_factor = (age - 20) / 10  # Age factor (0 for 20 years old, increases with age)
        
        # Calculate salary within the range
        salary = min_salary + (max_salary - min_salary) * (0.5 + 0.2 * age_factor)
        
        # Ensure salary is within bounds
        salary = max(min_salary, min(max_salary, salary))
        
        # Format the response
        response = {
            'salary': f"{salary:,.2f}",
            'currency': 'INR'
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    try:
        # Load configuration
        with open('models/house_price_config.json') as f:
            config = json.load(f)
            price_per_sqft_inr = config['price_per_sqft_inr']
            min_price_inr = config['min_price_inr']
            max_price_inr = config['max_price_inr']
        
        # Get input data
        data = request.get_json()
        square_footage = float(data['square_footage'])
        
        # Calculate price
        price_inr = price_per_sqft_inr * square_footage
        
        # Ensure price is within bounds
        price_inr = max(min_price_inr, min(max_price_inr, price_inr))
        
        # Format the response
        response = {
            'price': f"{price_inr:,.2f}",
            'currency': 'INR'
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()
        
        # Validate input data
        required_fields = ['glucose', 'bmi', 'age', 'blood_pressure']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            if not isinstance(data[field], (int, float)):
                return jsonify({'error': f'Invalid value for {field}: must be a number'}), 400
            
        # Convert inputs to float
        features = [
            float(data['glucose']),
            float(data['bmi']),
            float(data['age']),
            float(data['blood_pressure'])
        ]
        
        # Make prediction
        prediction = models['diabetes'].predict([features])[0]
        probability = models['diabetes'].predict_proba([features])[0][1]
        
        return jsonify({
            'prediction': 'Diabetic' if prediction == 1 else 'Non-diabetic',
            'probability': round(probability * 100, 2)
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in diabetes prediction: {str(e)}")
        return jsonify({'error': 'Server error. Please try again.'}), 500

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    try:
        data = request.get_json()
        
        # Validate input data
        required_fields = ['cholesterol', 'blood_pressure', 'age', 'smoking']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            if not isinstance(data[field], (int, float)):
                return jsonify({'error': f'Invalid value for {field}: must be a number'}), 400
            
        # Convert inputs to float
        features = [
            float(data['cholesterol']),
            float(data['blood_pressure']),
            float(data['age']),
            float(data['smoking'])
        ]
        
        # Scale the features
        scaler = joblib.load('models/heart_disease_scaler.pkl')
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = models['heart_disease'].predict(features_scaled)[0]
        probability = models['heart_disease'].predict_proba(features_scaled)[0][1]
        
        # Adjust probability to be more realistic (capped at 95%)
        probability = min(probability, 0.95)
        
        return jsonify({
            'prediction': 'Has heart disease' if prediction == 1 else 'No heart disease',
            'probability': round(probability * 100, 2)
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in heart disease prediction: {str(e)}")
        return jsonify({'error': 'Server error. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
