from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import firebase_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(__file__).parent / 'models'
RES_DIR = Path(__file__).parent / 'res'
RF_MODEL_PATH = RES_DIR / 'random_forest_regression_model.pkl'
TS_MODEL_PATH = RES_DIR / 'timeseries_model.keras'

rf_model = None
ts_model = None
models_loaded = False

def load_models():
    global rf_model, ts_model, models_loaded
    
    try:
        if RF_MODEL_PATH.exists():
            with open(RF_MODEL_PATH, 'rb') as f:
                rf_model = pickle.load(f)
            logger.info("Random Forest model loaded successfully")
        else:
            logger.warning(f"Random Forest model not found at {RF_MODEL_PATH}")
        
        if TS_MODEL_PATH.exists():
            ts_model = tf.keras.models.load_model(str(TS_MODEL_PATH))
            logger.info("Time Series model loaded successfully")
        else:
            logger.warning(f"Time Series model not found at {TS_MODEL_PATH}")
        
        models_loaded = rf_model is not None and ts_model is not None
        
        if models_loaded:
            logger.info("All models loaded successfully")
        else:
            logger.error("Failed to load one or more models")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        models_loaded = False

def get_risk_level(premise_index):
    if premise_index < 30:
        return 'low'
    elif premise_index < 60:
        return 'medium'
    else:
        return 'high'

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not rf_model:
            return jsonify({'error': 'Random Forest model not loaded'}), 500
        
        data = request.get_json()
        
        required_fields = ['temperature', 'rainfall', 'water_content', 'rainfall_7d_avg', 'watercontent_7d_avg']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        features = np.array([[
            float(data['temperature']),
            float(data['rainfall']),
            float(data['water_content']),
            float(data['rainfall_7d_avg']),
            float(data['watercontent_7d_avg'])
        ]])
        
        premise_index = rf_model.predict(features)[0]
        
        confidence = 0.85
        if hasattr(rf_model, 'predict_proba'):
            probabilities = rf_model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
        
        risk_level = get_risk_level(premise_index)
        
        response = {
            'premiseIndex': float(premise_index),
            'riskLevel': risk_level,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            prediction_data = {
                'id': response['timestamp'],
                'timestamp': response['timestamp'],
                'premiseIndex': response['premiseIndex'],
                'rainfall': float(data['rainfall']),
                'temperature': float(data['temperature']),
                'water_content': float(data['water_content']),
                'rainfall_7d_avg': float(data['rainfall_7d_avg']),
                'watercontent_7d_avg': float(data['watercontent_7d_avg']),
                'riskLevel': response['riskLevel'],
                'confidence': response['confidence']
            }
            firebase_service.save_prediction(prediction_data)
        except Exception as firebase_error:
            logger.error(f"Failed to save to Firebase: {firebase_error}")
        
        logger.info(f"Prediction made: {premise_index:.2f}% ({risk_level} risk)")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['GET'])
def forecast():
    try:
        if not ts_model or not rf_model:
            logger.error("Forecast error: One or more models are not loaded.")
            return jsonify({'error': 'One or more models are not loaded. Check server logs.'}), 500

        days_str = request.args.get('days', '90')
        days = int(days_str)
        
        df = pd.read_csv(RES_DIR / 'mosquito_dataset_2017_2024.csv', parse_dates=['Date'])
        
        features_ts = ['Temperature', 'Rainfall']
        historical_data = df[features_ts].values
        
        last_60_days = historical_data[-60:]
        input_sequence = np.array([last_60_days])

        predicted_weather = ts_model.predict(input_sequence)
        predicted_weather = predicted_weather.squeeze()

        historical_full_features = df[['Temperature', 'Rainfall', 'WaterContent']].tail(7)
        final_premise_index_forecast = []

        for i in range(days):
            pred_temp = predicted_weather[i, 0]
            pred_rain = predicted_weather[i, 1]
            pred_wc = pred_rain * 3.5

            today_features = pd.DataFrame({
                'Temperature': [pred_temp],
                'Rainfall': [pred_rain],
                'WaterContent': [pred_wc]
            })

            updated_history = pd.concat([historical_full_features, today_features], ignore_index=True)
            
            rainfall_7d_avg = updated_history['Rainfall'].rolling(window=7).mean().iloc[-1]
            watercontent_7d_avg = updated_history['WaterContent'].rolling(window=7).mean().iloc[-1]

            rf_input = np.array([[
                pred_temp,
                pred_rain,
                pred_wc,
                rainfall_7d_avg,
                watercontent_7d_avg
            ]])

            premise_index_pred = rf_model.predict(rf_input)
            final_premise_index_forecast.append(premise_index_pred[0])

            historical_full_features = updated_history.iloc[1:]

        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        
        response = {
            'dates': dates,
            'predictions': [float(p) for p in final_premise_index_forecast],
            'confidence_intervals': {
                'lower': [max(0, float(p) - 10) for p in final_premise_index_forecast],
                'upper': [min(100, float(p) + 10) for p in final_premise_index_forecast]
            }
        }
        
        logger.info(f"Forecast generated for {days} days")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Forecast error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    firebase_service.init_firebase()
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
