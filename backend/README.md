# Dengue Breeding Prediction Backend

This Flask backend serves the trained ML models for the Dengue Breeding Prediction System.

## Setup Instructions

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Enable Firebase Services

Before setting up credentials, ensure Firebase services are enabled:

1. **Enable Firestore API:**
   - Visit: https://console.developers.google.com/apis/api/firestore.googleapis.com/overview?project=dengue-prediction-9de60
   - Click "Enable" button
   - Wait a few minutes for activation

2. **Initialize Firestore Database:**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Select your project
   - Navigate to "Firestore Database"
   - Click "Create database"
   - Choose "Start in test mode" (for development)
   - Select a location

### 3. Setup Firebase Credentials

Copy your Firebase service account key:
```bash
cp firebase-credentials.example.json firebase-credentials.json
```

Then edit `firebase-credentials.json` with your actual Firebase credentials from the Firebase Console.

### 4. Place Your Trained Models

Create a `models` directory in the backend folder and place your trained models:

```
backend/
├── firebase-credentials.json              # Your Firebase credentials (gitignored)
├── models/
│   ├── random_forest_regression_model.pkl    # Your trained Random Forest model
│   └── timeseries_model.keras                # Your trained Time Series model
├── app.py
├── requirements.txt
└── README.md
```

### 5. Model File Requirements

#### Random Forest Model (`random_forest_regression_model.pkl`)
- Should be a scikit-learn model saved using pickle
- Must accept input features: [rainfall, temperature, water_content]
- Should output a single premise index value (0-100)

Example of how your model should be saved:
```python
import pickle
from sklearn.ensemble import RandomForestRegressor

# After training your model
with open('models/random_forest_regression_model.pkl', 'wb') as f:
    pickle.dump(your_trained_model, f)
```

#### Time Series Model (`timeseries_model.keras`)
- Should be a TensorFlow/Keras model saved in .keras format
- Used for generating future predictions/forecasts

Example of how your model should be saved:
```python
import tensorflow as tf

# After training your model
your_trained_model.save('models/timeseries_model.keras')
```

### 6. Start the Backend Server
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/api/health`
- Returns server status and model loading status

### Make Prediction
- **POST** `/api/predict`
- Body: `{"rainfall": float, "temperature": float, "waterContent": float}`
- Returns premise index prediction and risk level

### Get Forecast
- **GET** `/api/forecast?days=90`
- Returns time series forecast for specified number of days

## Model Integration Notes

1. **Input Preprocessing**: Modify the preprocessing in `app.py` to match your model's expected input format
2. **Output Processing**: Adjust the output processing based on your model's output format
3. **Feature Engineering**: Add any feature engineering steps your models require
4. **Error Handling**: The API includes comprehensive error handling for model failures

## Environment Variables

Create a `.env` file in the backend directory if needed:
```
FLASK_ENV=development
MODEL_DIR=./models
```
## Troubleshooting Firebase Issues

### Firestore API Not Enabled
- **Error**: `Cloud Firestore API has not been used in project`
- **Solution**: Enable Firestore API at the provided URL
- **Wait Time**: Allow 5-10 minutes after enabling

### Invalid JWT Signature
- **Error**: `Invalid JWT Signature`
- **Solution**: Generate new service account key from Firebase Console
- **Check**: Ensure project ID matches in all configuration files

### Permission Denied
- **Error**: `403 Permission denied`
- **Solution**: Verify service account has "Firebase Admin SDK" role
- **Check**: Ensure Firestore security rules allow admin access