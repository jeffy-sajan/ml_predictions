import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import os

def train_calories_model():
    # Load and prepare data
    data = pd.read_csv('datasets/calories.csv')
    X = data[['KMs_Run']].values
    y = data['Calories_Burned'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model using Simple Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    
    # Save model and accuracy
    joblib.dump(model, 'models/calories_model.pkl')
    with open('models/calories_accuracy.json', 'w') as f:
        json.dump({'accuracy': round(accuracy * 100, 2)}, f)
    
    print(f"Calories prediction model trained and saved with accuracy: {accuracy * 100:.2f}%")

def train_salary_model():
    # Define salary ranges based on experience and education level
    salary_ranges = {
        # Experience (years) -> Education Level -> Salary Range (in INR)
        0: {  # Freshers
            1: (15000, 25000),  # Diploma
            2: (20000, 30000),  # Bachelor's
            3: (25000, 35000),  # Master's
            4: (30000, 40000)   # Doctorate
        },
        1: {  # 1 year experience
            1: (20000, 30000),
            2: (25000, 35000),
            3: (30000, 40000),
            4: (35000, 45000)
        },
        2: {  # 2 years experience
            1: (25000, 35000),
            2: (30000, 40000),
            3: (35000, 45000),
            4: (40000, 50000)
        },
        3: {  # 3 years experience
            1: (30000, 40000),
            2: (35000, 45000),
            3: (40000, 50000),
            4: (45000, 55000)
        },
        4: {  # 4 years experience
            1: (35000, 45000),
            2: (40000, 50000),
            3: (45000, 55000),
            4: (50000, 60000)
        },
        5: {  # 5+ years experience
            1: (40000, 50000),
            2: (45000, 55000),
            3: (50000, 60000),
            4: (55000, 70000)
        }
    }
    
    # Save the salary ranges
    with open('models/salary_ranges.json', 'w') as f:
        json.dump(salary_ranges, f)
    
    print("Salary prediction model trained and saved with fixed ranges")

def train_house_price_model():
    # Define fixed price per square foot in INR
    price_per_sqft_inr = 4000  # ₹4,000 per square foot
    
    # Save the price configuration
    with open('models/house_price_config.json', 'w') as f:
        json.dump({
            'price_per_sqft_inr': price_per_sqft_inr,
            'min_price_inr': 1000000,  # Minimum price in INR
            'max_price_inr': 100000000  # Maximum price in INR
        }, f)
    
    print("House price prediction model trained and saved with fixed price per square foot")

def train_diabetes_model():
    # Load and prepare data
    data = pd.read_csv('datasets/diabetes.csv')
    X = data[['Glucose', 'BMI', 'Age', 'Blood_Pressure']].values
    y = data['Diabetes'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model using Logistic Regression
    model = LogisticRegression(random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate balanced accuracy
    accuracy = balanced_accuracy_score(y_test, y_pred)
    
    # Save model and scaler
    joblib.dump(model, 'models/diabetes_model.pkl')
    joblib.dump(scaler, 'models/diabetes_scaler.pkl')
    with open('models/diabetes_accuracy.json', 'w') as f:
        json.dump({'accuracy': round(accuracy * 100, 2)}, f)
    
    print(f"Diabetes prediction model trained and saved with accuracy: {accuracy * 100:.2f}%")

def train_heart_disease_model():
    # Load and prepare data
    data = pd.read_csv('datasets/heart_disease.csv')
    X = data[['Cholesterol', 'Blood_Pressure', 'Age', 'Smoking']].values
    y = data['Heart_Disease'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model using K-Nearest Neighbors
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate balanced accuracy
    accuracy = balanced_accuracy_score(y_test, y_pred)
    
    # Save model and scaler
    joblib.dump(model, 'models/heart_disease_model.pkl')
    joblib.dump(scaler, 'models/heart_disease_scaler.pkl')
    with open('models/heart_disease_accuracy.json', 'w') as f:
        json.dump({'accuracy': round(accuracy * 100, 2)}, f)
    
    print(f"Heart disease prediction model trained and saved with accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Train all models
    train_calories_model()
    train_salary_model()
    train_house_price_model()
    train_diabetes_model()
    train_heart_disease_model()
