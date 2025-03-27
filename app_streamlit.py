import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

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

# Load configuration
with open('models/house_price_config.json') as f:
    house_config = json.load(f)

with open('models/salary_ranges.json') as f:
    salary_ranges = json.load(f)

# Streamlit app
st.title("ML Predictions App")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select Prediction Type",
    ["Home", "Calories Prediction", "Salary Prediction", "House Price Prediction", "Diabetes Prediction", "Heart Disease Prediction"]
)

# Home page
def show_home():
    st.header("Welcome to ML Predictions")
    st.write("Choose a prediction type from the sidebar to get started.")
    st.write("Available predictions:")
    st.write("- Calories burned prediction")
    st.write("- Salary prediction (in INR)")
    st.write("- House price prediction (in INR)")
    st.write("- Diabetes prediction")
    st.write("- Heart disease prediction")

# Calories prediction
def predict_calories():
    st.header("Calories Burned Prediction")
    
    km_run = st.number_input("Distance Run (km)", min_value=0.0, value=5.0)
    
    if st.button("Predict"):        
        prediction = models['calories'].predict([[km_run]])[0]
        st.success(f"Estimated Calories Burned: {prediction:.2f}")

# Salary prediction
def predict_salary():
    st.header("Salary Prediction")
    
    experience = st.number_input("Years of Experience", min_value=0, max_value=10, value=2)
    education_level = st.selectbox("Education Level", ["Diploma", "Bachelor's", "Master's", "Doctorate"])
    age = st.number_input("Age", min_value=20, max_value=60, value=25)
    
    if st.button("Predict"):        
        # Get the salary range based on experience and education level
        if experience > 5:
            experience = 5  # Cap at 5+ years
        
        education_map = {
            "Diploma": "1",
            "Bachelor's": "2",
            "Master's": "3",
            "Doctorate": "4"
        }
        
        min_salary, max_salary = salary_ranges[str(experience)][education_map[education_level]]
        
        # Calculate a realistic salary within the range
        age_factor = (age - 20) / 10  # Age factor (0 for 20 years old, increases with age)
        
        # Calculate salary within the range
        salary = min_salary + (max_salary - min_salary) * (0.5 + 0.2 * age_factor)
        
        # Ensure salary is within bounds
        salary = max(min_salary, min(max_salary, salary))
        
        st.success(f"Estimated Salary: ₹{salary:,.2f}")

# House price prediction
def predict_house_price():
    st.header("House Price Prediction")
    
    square_footage = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
    
    if st.button("Predict"):        
        price_inr = house_config['price_per_sqft_inr'] * square_footage
        price_inr = max(house_config['min_price_inr'], min(house_config['max_price_inr'], price_inr))
        
        st.success(f"Estimated House Price: ₹{price_inr:,.2f}")

# Diabetes prediction
def predict_diabetes():
    st.header("Diabetes Prediction")
    
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
    bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
    age = st.number_input("Age", min_value=20, max_value=100, value=30)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
    
    if st.button("Predict"):        
        # Scale the features
        scaler = joblib.load('models/diabetes_scaler.pkl')
        features = [glucose, bmi, age, blood_pressure]
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = models['diabetes'].predict(features_scaled)[0]
        probability = models['diabetes'].predict_proba(features_scaled)[0][1]
        
        result = "Diabetic" if prediction == 1 else "Non-diabetic"
        st.success(f"Prediction: {result}")
        st.write(f"Probability: {probability:.2%}")

# Heart disease prediction
def predict_heart_disease():
    st.header("Heart Disease Prediction")
    
    cholesterol = st.number_input("Cholesterol Level", min_value=0, max_value=400, value=200)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=120)
    age = st.number_input("Age", min_value=20, max_value=100, value=30)
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    
    if st.button("Predict"):        
        # Convert smoking to binary
        smoking_val = 1 if smoking == "Yes" else 0
        
        # Scale the features
        scaler = joblib.load('models/heart_disease_scaler.pkl')
        features = [cholesterol, blood_pressure, age, smoking_val]
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = models['heart_disease'].predict(features_scaled)[0]
        probability = models['heart_disease'].predict_proba(features_scaled)[0][1]
        
        result = "Has heart disease" if prediction == 1 else "No heart disease"
        st.success(f"Prediction: {result}")
        st.write(f"Probability: {probability:.2%}")

# Main app
if page == "Home":
    show_home()
elif page == "Calories Prediction":
    predict_calories()
elif page == "Salary Prediction":
    predict_salary()
elif page == "House Price Prediction":
    predict_house_price()
elif page == "Diabetes Prediction":
    predict_diabetes()
elif page == "Heart Disease Prediction":
    predict_heart_disease()
