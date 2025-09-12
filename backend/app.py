from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import logging
from pathlib import Path
import firebase_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- model and resource paths ---
RES_DIR = Path(__file__).parent / 'res'
RF_MODEL_PATH = RES_DIR / 'random_forest_regression_model.pkl'
TS_MODEL_PATH = RES_DIR / 'timeseries_model.h5'
TS_SCALER_PATH = RES_DIR / 'timeseries_scaler.pkl'
DATASET_PATH = RES_DIR / 'mosquito_dataset_2017_2024.csv'


# --- Global variables for models and scaler ---
rf_model = None
ts_model = None
ts_scaler = None
models_loaded = False
model_loading_error = None

def load_models():
    """Load the trained models and the scaler at startup."""
    global rf_model, ts_model, ts_scaler, models_loaded, model_loading_error
    
    try:
        if RF_MODEL_PATH.exists():
            with open(RF_MODEL_PATH, 'rb') as f:
                rf_model = pickle.load(f)
            logger.info("✅ Random Forest model loaded successfully")
        else:
            logger.warning(f"❌ Random Forest model not found at {RF_MODEL_PATH}")
        
        if TS_MODEL_PATH.exists():
            ts_model = tf.keras.models.load_model(str(TS_MODEL_PATH))
            logger.info("✅ Time Series model loaded successfully")
        else:
            logger.warning(f"❌ Time Series model not found at {TS_MODEL_PATH}")

        if TS_SCALER_PATH.exists():
            with open(TS_SCALER_PATH, 'rb') as f:
                ts_scaler = pickle.load(f)
            logger.info("✅ Time Series scaler loaded successfully")
        else:
            logger.warning(f"❌ Time Series scaler not found at {TS_SCALER_PATH}")
        
        models_loaded = all([rf_model, ts_model, ts_scaler])
        
        if models_loaded:
            logger.info("🚀 All models and scaler loaded successfully")
        else:
            logger.error("🔥 Failed to load one or more models or the scaler")
            
    except Exception as e:
        model_loading_error = str(e)
        logger.error(f"Error loading models: {model_loading_error}")
        models_loaded = False

def get_risk_level(premise_index):
    """Determine risk level based on premise index."""
    if premise_index < 30:
        return 'low'
    elif premise_index < 60:
        return 'medium'
    else:
        return 'high'

def get_season_wet_for_date(date):
    """
    Determines if a date is in the wet season (May-Sept).
    Returns 1 for Wet, 0 for Dry.
    """
    return 1 if 5 <= date.month <= 9 else 0

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    response = {
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'rf_model_loaded': rf_model is not None,
        'ts_model_loaded': ts_model is not None,
        'ts_scaler_loaded': ts_scaler is not None,
        'timestamp': datetime.now().isoformat()
    }
    if model_loading_error:
        response['error'] = model_loading_error
    return jsonify(response)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handles single-day predictions using the Random Forest model."""
    try:
        if not rf_model:
            return jsonify({'error': 'Random Forest model not loaded'}), 500
        
        data = request.get_json()
        required_fields = ['temperature', 'rainfall', 'water_content', 'rainfall_7d_avg', 'watercontent_7d_avg']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing one or more required fields'}), 400
        
        features = np.array([[
            float(data['temperature']),
            float(data['rainfall']),
            float(data['water_content']),
            float(data['rainfall_7d_avg']),
            float(data['watercontent_7d_avg'])
        ]])
        
        premise_index = rf_model.predict(features)[0]
        risk_level = get_risk_level(premise_index)
        
        response = {
            'premiseIndex': float(premise_index),
            'riskLevel': risk_level,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            prediction_data = {**data, **response, 'id': response['timestamp']}
            firebase_service.save_prediction(prediction_data)
        except Exception as fb_error:
            logger.error(f"Failed to save prediction to Firebase: {fb_error}")
        
        logger.info(f"Prediction made: {premise_index:.2f}% ({risk_level} risk)")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500

@app.route('/api/forecast', methods=['GET'])
def forecast():
    """Handles multi-day forecasting using the Time Series model autoregressively."""
    try:
        if not ts_model or not ts_scaler:
            return jsonify({'error': 'Time Series model or scaler not loaded'}), 500

        days_to_forecast = int(request.args.get('days', '42'))
        
        df = pd.read_csv(DATASET_PATH, parse_dates=['Date'], index_col='Date')
        df_encoded = pd.get_dummies(df, columns=['Season'], drop_first=True)
        features_df = df_encoded[['PremiseIndex', 'Season_Wet']]
        
        last_60_days = features_df.tail(60)
        # **FIX**: Start the forecast date from the current system time
        start_date = datetime.now()
        
        current_sequence_scaled = ts_scaler.transform(last_60_days)
        
        predictions = []

        for i in range(days_to_forecast):
            input_for_model = np.reshape(current_sequence_scaled, (1, 60, 2))
            
            scaled_prediction = ts_model.predict(input_for_model, verbose=0)[0][0]
            
            # **FIX**: Use the current date for the forecast, not the end of the dataset
            next_date = start_date + timedelta(days=i)
            next_season_wet = get_season_wet_for_date(next_date)

            prediction_reshaped = np.array([[scaled_prediction, 0]])
            final_prediction = ts_scaler.inverse_transform(prediction_reshaped)[0][0]

            predictions.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'premiseIndex': float(final_prediction)
            })
            
            new_scaled_point = np.array([[scaled_prediction, next_season_wet]])
            current_sequence_scaled = np.append(current_sequence_scaled[1:], new_scaled_point, axis=0)

        response = {
            'forecast': predictions,
            'confidence_intervals': {
                'lower': [max(0, p['premiseIndex'] - 10) for p in predictions],
                'upper': [min(100, p['premiseIndex'] + 10) for p in predictions]
            }
        }
        
        logger.info(f"Forecast generated for {days_to_forecast} days")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Forecast error: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during forecasting.'}), 500

@app.route('/api/predictions/clear', methods=['POST'])
def clear_predictions():
    """Clears all predictions from the database."""
    try:
        deleted_count = firebase_service.clear_all_predictions()
        return jsonify({'message': f'Successfully deleted {deleted_count} predictions.'}), 200
    except Exception as e:
        logger.error(f"Error clearing predictions: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during prediction clearing.'}), 500

if __name__ == '__main__':
    firebase_service.init_firebase()
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
