# Machine Learning Flask Application

This Flask application provides a web interface for various machine learning predictions including:
- Calories burned prediction
- Salary prediction
- House price prediction
- Diabetes prediction
- Heart disease prediction

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Move your datasets into the `datasets` folder:
- `calories.csv` - for calories prediction
- `salary_data.csv` - for salary prediction
- `house_prices.csv` - for house price prediction
- `diabetes.csv` - for diabetes prediction
- `heart_disease.csv` - for heart disease prediction

5. Train the models:
```bash
python train_models.py
```

6. Run the Flask application:
```bash
python app.py
```

## Deployment to Render

1. Create a new account on [Render](https://render.com)
2. Connect your GitHub account
3. Create a new Web Service:
   - Choose "Create New Web Service"
   - Select your GitHub repository
   - Choose the branch you want to deploy (usually main/master)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Environment Variables:
     - Add any required environment variables here
   - Click "Create Web Service"

4. Wait for the deployment to complete
5. Your application will be available at the URL provided by Render

## Using the Application

1. Open your web browser and navigate to `http://localhost:5000`
2. Use the forms to input data and get predictions for each model
3. Results will be displayed in real-time

## API Endpoints

The application provides the following API endpoints:
- `/predict_calories` - POST request with duration
- `/predict_salary` - POST request with years_experience
- `/predict_house_price` - POST request with area
- `/predict_diabetes` - POST request with diabetes features
- `/predict_heart_disease` - POST request with heart disease features

## Note

Make sure to have all required datasets in the `datasets` folder before training the models. The models will be saved in the `models` folder as pickle files.
