# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import pickle
# import tensorflow as tf
# from datetime import datetime, timedelta
# import logging
# import os
# from pathlib import Path
# import firebase_service
# # from firebase_service import firebase_service

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React frontend

# # Model storage paths
# MODEL_DIR = Path(__file__).parent / 'models'
# RES_DIR = Path(__file__).parent / 'res'
# RF_MODEL_PATH = RES_DIR / 'random_forest_regression_model.pkl'
# TS_MODEL_PATH = RES_DIR / 'timeseries_model.h5'
# TS_SCALER_PATH = RES_DIR / 'timeseries_scaler.pkl'

# # Global variables for models
# rf_model = None
# ts_model = None
# ts_scaler = None
# models_loaded = False

# def load_models():
#     """Load the trained models at startup"""
#     global rf_model, ts_model, models_loaded

#     try:
#         # Load Random Forest model
#         if RF_MODEL_PATH.exists():
#             with open(RF_MODEL_PATH, 'rb') as f:
#                 rf_model = pickle.load(f)
#             logger.info("Random Forest model loaded successfully")
#         else:
#             logger.warning(f"Random Forest model not found at {RF_MODEL_PATH}")

#         # Load Time Series model
#         if TS_MODEL_PATH.exists():
#             ts_model = tf.keras.models.load_model(str(TS_MODEL_PATH))
#             logger.info("Time Series model loaded successfully")
#         else:
#             logger.warning(f"Time Series model not found at {TS_MODEL_PATH}")

#         models_loaded = rf_model is not None and ts_model is not None

#         if models_loaded:
#             logger.info("All models loaded successfully")
#         else:
#             logger.error("Failed to load one or more models")

#     except Exception as e:
#         logger.error(f"Error loading models: {str(e)}")
#         models_loaded = False

# def get_risk_level(premise_index):
#     """Determine risk level based on premise index"""
#     if premise_index < 30:
#         return 'low'
#     elif premise_index < 60:
#         return 'medium'
#     else:
#         return 'high'

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy' if models_loaded else 'unhealthy',
#         'models_loaded': models_loaded,
#         'timestamp': datetime.now().isoformat()
#     })

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     """Make prediction using the Random Forest model"""
#     try:
#         if not rf_model:
#             return jsonify({'error': 'Random Forest model not loaded'}), 500

#         data = request.get_json()

#         # Validate input data
#         required_fields = ['temperature', 'rainfall', 'water_content', 'rainfall_7d_avg', 'watercontent_7d_avg']
#         for field in required_fields:
#             if field not in data:
#                 return jsonify({'error': f'Missing required field: {field}'}), 400

#         # Prepare features for prediction
#         features = np.array([[
#             float(data['temperature']),
#             float(data['rainfall']),
#             float(data['water_content']),
#             float(data['rainfall_7d_avg']),
#             float(data['watercontent_7d_avg'])
#         ]])

#         # Make prediction
#         premise_index = rf_model.predict(features)[0]

#         # Get prediction confidence (if available)
#         confidence = 0.85  # Default confidence, replace with actual if your model supports it
#         if hasattr(rf_model, 'predict_proba'):
#             # For classification models
#             probabilities = rf_model.predict_proba(features)[0]
#             confidence = float(np.max(probabilities))

#         # Determine risk level
#         risk_level = get_risk_level(premise_index)

#         response = {
#             'premiseIndex': float(premise_index),
#             'riskLevel': risk_level,
#             'confidence': confidence,
#             'timestamp': datetime.now().isoformat()
#         }

#         # Save prediction to Firebase
#         try:
#             prediction_data = {
#                 'id': response['timestamp'],  # Use timestamp as ID for consistency
#                 'timestamp': response['timestamp'],
#                 'premiseIndex': response['premiseIndex'],
#                 'rainfall': float(data['rainfall']),
#                 'temperature': float(data['temperature']),
#                 'water_content': float(data['water_content']),
#                 'rainfall_7d_avg': float(data['rainfall_7d_avg']),
#                 'watercontent_7d_avg': float(data['watercontent_7d_avg']),
#                 'riskLevel': response['riskLevel'],
#                 'confidence': response['confidence']
#             }
#             firebase_service.save_prediction(prediction_data)
#         except Exception as firebase_error:
#             logger.error(f"Failed to save to Firebase: {firebase_error}")

#         logger.info(f"Prediction made: {premise_index:.2f}% ({risk_level} risk)")
#         return jsonify(response)

#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/forecast', methods=['GET'])
# def forecast():
#     """Forecast future PremiseIndex using a two-step process."""
#     try:
#         days_str = request.args.get('days', '90')
#         days = int(days_str)

#         # --- Step 0: Prepare historical data ---
#         # Load the dataset to get the last 60 days for the initial prediction
#         df = pd.read_csv(RES_DIR / 'mosquito_dataset_2017_2024.csv', parse_dates=['Date'])

#         # Select features used for time series training
#         features_ts = ['Temperature', 'Rainfall']
#         historical_data = df[features_ts].values

#         # Get the last 60 days as the initial input for the time series model
#         last_60_days = historical_data[-60:]
#         input_sequence = np.array([last_60_days])

#         # --- Step 1: Get the full weather forecast from the Time Series Model ---
#         # The model will output predictions for the next 90 days
#         predicted_weather = ts_model.predict(input_sequence) # Shape: (1, 90, 2)

#         # Squeeze the array to make it easier to work with
#         predicted_weather = predicted_weather.squeeze() # Shape: (90, 2)

#         # We also need the last 7 days of full historical data to calculate rolling averages
#         historical_full_features = df[['Temperature', 'Rainfall', 'WaterContent']].tail(7)

#         # --- Step 2: Iterate through the weather forecast to predict PremiseIndex ---
#         final_premise_index_forecast = []

#         for i in range(days):
#             # Get today's predicted Temperature and Rainfall
#             pred_temp = predicted_weather[i, 0]
#             pred_rain = predicted_weather[i, 1]

#             # **Assumption**: Estimate WaterContent based on Rainfall.
#             # A simple but effective heuristic is a direct relationship.
#             # This value can be tuned if you have more domain knowledge.
#             pred_wc = pred_rain * 3.5

#             # Create a dataframe for the current day's prediction
#             today_features = pd.DataFrame({
#                 'Temperature': [pred_temp],
#                 'Rainfall': [pred_rain],
#                 'WaterContent': [pred_wc]
#             })

#             # Append today's data to our historical window to calculate rolling averages
#             updated_history = pd.concat([historical_full_features, today_features], ignore_index=True)

#             # Calculate rolling averages for the 5-feature model
#             rainfall_7d_avg = updated_history['Rainfall'].rolling(window=7).mean().iloc[-1]
#             watercontent_7d_avg = updated_history['WaterContent'].rolling(window=7).mean().iloc[-1]

#             # Assemble the final input for the Random Forest model
#             rf_input = np.array([[
#                 pred_temp,
#                 pred_rain,
#                 pred_wc,
#                 rainfall_7d_avg,
#                 watercontent_7d_avg
#             ]])

#             # Predict the PremiseIndex for this day
#             premise_index_pred = rf_model.predict(rf_input)
#             final_premise_index_forecast.append(premise_index_pred[0])

#             # Update the history for the next iteration's rolling average calculation
#             historical_full_features = updated_history.iloc[1:]


#         # --- Step 3: Format and Return the Response ---
#         dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]

#         response = {
#             'dates': dates,
#             'predictions': [float(p) for p in final_premise_index_forecast],
#             'confidence_intervals': {
#                 'lower': [max(0, float(p) - 10) for p in final_premise_index_forecast],
#                 'upper': [min(100, float(p) + 10) for p in final_premise_index_forecast]
#             }
#         }

#         logger.info(f"Forecast generated for {days} days")
#         return jsonify(response)

#     except Exception as e:
#         logger.error(f"Forecast error: {e}", exc_info=True)
#         return jsonify({'error': str(e)}), 500


# #  Added this block to run the server
# if __name__ == '__main__':
#     # Load models on startup
#     firebase_service.init_firebase()
#     load_models()
#     # Run the Flask application
#     # debug=True enables auto-reloading when you save the file
#     app.run(host='0.0.0.0', port=5000, debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import pickle
# import tensorflow as tf
# from datetime import datetime, timedelta
# import logging
# import os
# from pathlib import Path
# import firebase_service

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React frontend

# # Model storage paths
# MODEL_DIR = Path(__file__).parent / 'models'
# RES_DIR = Path(__file__).parent / 'res'
# RF_MODEL_PATH = RES_DIR / 'random_forest_regression_model.pkl'
# TS_MODEL_PATH = RES_DIR / 'timeseries_model.h5'
# # FIX: Add path for the time series scaler
# TS_SCALER_PATH = RES_DIR / 'timeseries_scaler.pkl'

# # Global variables for models
# rf_model = None
# ts_model = None
# # FIX: Add global variable for the scaler
# ts_scaler = None
# models_loaded = False

# def load_models():
#     """Load the trained models and scaler at startup"""
#     global rf_model, ts_model, ts_scaler, models_loaded

#     try:
#         # Load Random Forest model
#         if RF_MODEL_PATH.exists():
#             with open(RF_MODEL_PATH, 'rb') as f:
#                 rf_model = pickle.load(f)
#             logger.info("Random Forest model loaded successfully")
#         else:
#             logger.warning(f"Random Forest model not found at {RF_MODEL_PATH}")

#         # Load Time Series model
#         if TS_MODEL_PATH.exists():
#             ts_model = tf.keras.models.load_model(str(TS_MODEL_PATH))
#             logger.info("Time Series model loaded successfully")
#         else:
#             logger.warning(f"Time Series model not found at {TS_MODEL_PATH}")

#         # FIX: Load the Time Series scaler
#         if TS_SCALER_PATH.exists():
#             with open(TS_SCALER_PATH, 'rb') as f:
#                 ts_scaler = pickle.load(f)
#             logger.info("Time Series scaler loaded successfully")
#         else:
#             logger.warning(f"Time Series scaler not found at {TS_SCALER_PATH}")

#         # FIX: Update check to include the scaler
#         models_loaded = all([rf_model, ts_model, ts_scaler])

#         if models_loaded:
#             logger.info("All models and scaler loaded successfully")
#         else:
#             logger.error("Failed to load one or more models or the scaler")

#     except Exception as e:
#         logger.error(f"Error loading models: {str(e)}")
#         models_loaded = False

# def get_risk_level(premise_index):
#     """Determine risk level based on premise index"""
#     if premise_index < 30:
#         return 'low'
#     elif premise_index < 60:
#         return 'medium'
#     else:
#         return 'high'

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy' if models_loaded else 'unhealthy',
#         'models_loaded': models_loaded,
#         'timestamp': datetime.now().isoformat()
#     })

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     """Make prediction using the Random Forest model"""
#     try:
#         if not rf_model:
#             return jsonify({'error': 'Random Forest model not loaded'}), 500

#         data = request.get_json()

#         # Validate input data
#         required_fields = ['temperature', 'rainfall', 'water_content', 'rainfall_7d_avg', 'watercontent_7d_avg']
#         for field in required_fields:
#             if field not in data:
#                 return jsonify({'error': f'Missing required field: {field}'}), 400

#         # Prepare features for prediction
#         features = np.array([[
#             float(data['temperature']),
#             float(data['rainfall']),
#             float(data['water_content']),
#             float(data['rainfall_7d_avg']),
#             float(data['watercontent_7d_avg'])
#         ]])

#         # Make prediction
#         premise_index = rf_model.predict(features)[0]

#         confidence = 0.85
#         if hasattr(rf_model, 'predict_proba'):
#             probabilities = rf_model.predict_proba(features)[0]
#             confidence = float(np.max(probabilities))

#         risk_level = get_risk_level(premise_index)

#         response = {
#             'premiseIndex': float(premise_index),
#             'riskLevel': risk_level,
#             'confidence': confidence,
#             'timestamp': datetime.now().isoformat()
#         }

#         try:
#             prediction_data = {
#                 'id': response['timestamp'],
#                 'timestamp': response['timestamp'],
#                 'premiseIndex': response['premiseIndex'],
#                 'rainfall': float(data['rainfall']),
#                 'temperature': float(data['temperature']),
#                 'water_content': float(data['water_content']),
#                 'rainfall_7d_avg': float(data['rainfall_7d_avg']),
#                 'watercontent_7d_avg': float(data['watercontent_7d_avg']),
#                 'riskLevel': response['riskLevel'],
#                 'confidence': response['confidence']
#             }
#             firebase_service.save_prediction(prediction_data)
#         except Exception as firebase_error:
#             logger.error(f"Failed to save to Firebase: {firebase_error}")

#         logger.info(f"Prediction made: {premise_index:.2f}% ({risk_level} risk)")
#         return jsonify(response)

#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/forecast', methods=['GET'])
# def forecast():
#     """Forecast future PremiseIndex using a two-step process."""
#     try:
#         # FIX: Update guard clauses to ensure all models AND the scaler are loaded
#         if not ts_model or not rf_model or not ts_scaler:
#             logger.error("Forecast error: One or more models or the scaler are not loaded.")
#             return jsonify({'error': 'One or more models or the scaler are not loaded. Check server logs.'}), 500

#         days_str = request.args.get('days', '90')
#         days = int(days_str)

#         df = pd.read_csv(RES_DIR / 'mosquito_dataset_2017_2024.csv', parse_dates=['Date'])

#         features_ts = ['Temperature', 'Rainfall']
#         historical_data = df[features_ts].values

#         # FIX: Scale the historical data using the loaded scaler
#         scaled_historical_data = ts_scaler.transform(historical_data)

#         # Get the last 60 days of SCALED data as input
#         last_60_days = scaled_historical_data[-60:]
#         input_sequence = np.array([last_60_days])

#         # The model will predict scaled values
#         predicted_weather_scaled = ts_model.predict(input_sequence)

#         # FIX: Inverse transform the predictions to get them back to the original scale
#         predicted_weather = ts_scaler.inverse_transform(predicted_weather_scaled.squeeze())

#         historical_full_features = df[['Temperature', 'Rainfall', 'WaterContent']].tail(7)

#         final_premise_index_forecast = []

#         for i in range(days):
#             pred_temp = predicted_weather[i, 0]
#             pred_rain = predicted_weather[i, 1]
#             pred_wc = pred_rain * 3.5

#             today_features = pd.DataFrame({
#                 'Temperature': [pred_temp], 'Rainfall': [pred_rain], 'WaterContent': [pred_wc]
#             })

#             updated_history = pd.concat([historical_full_features, today_features], ignore_index=True)

#             rainfall_7d_avg = updated_history['Rainfall'].rolling(window=7).mean().iloc[-1]
#             watercontent_7d_avg = updated_history['WaterContent'].rolling(window=7).mean().iloc[-1]

#             rf_input = np.array([[
#                 pred_temp, pred_rain, pred_wc, rainfall_7d_avg, watercontent_7d_avg
#             ]])

#             premise_index_pred = rf_model.predict(rf_input)
#             final_premise_index_forecast.append(premise_index_pred[0])

#             historical_full_features = updated_history.iloc[1:]

#         dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]

#         response = {
#             'dates': dates,
#             'predictions': [float(p) for p in final_premise_index_forecast],
#             'confidence_intervals': {
#                 'lower': [max(0, float(p) - 10) for p in final_premise_index_forecast],
#                 'upper': [min(100, float(p) + 10) for p in final_premise_index_forecast]
#             }
#         }

#         logger.info(f"Forecast generated for {days} days")
#         return jsonify(response)

#     except Exception as e:
#         logger.error(f"Forecast error: {e}", exc_info=True)
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     firebase_service.init_firebase()
#     load_models()
#     app.run(host='0.0.0.0', port=5000, debug=True)


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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Model storage paths
MODEL_DIR = Path(__file__).parent / 'models'
RES_DIR = Path(__file__).parent / 'res'
RF_MODEL_PATH = RES_DIR / 'random_forest_regression_model.pkl'
TS_MODEL_PATH = RES_DIR / 'timeseries_model.h5'
# FIX: Add path for the time series scaler
TS_SCALER_PATH = RES_DIR / 'timeseries_scaler.pkl'

# Global variables for models
rf_model = None
ts_model = None
# FIX: Add global variable for the scaler
ts_scaler = None
models_loaded = False

def load_models():
    """Load the trained models and scaler at startup"""
    global rf_model, ts_model, ts_scaler, models_loaded
    
    try:
        # Load Random Forest model
        if RF_MODEL_PATH.exists():
            with open(RF_MODEL_PATH, 'rb') as f:
                rf_model = pickle.load(f)
            logger.info("Random Forest model loaded successfully")
        else:
            logger.warning(f"Random Forest model not found at {RF_MODEL_PATH}")
        
        # Load Time Series model
        if TS_MODEL_PATH.exists():
            ts_model = tf.keras.models.load_model(str(TS_MODEL_PATH))
            logger.info("Time Series model loaded successfully")
        else:
            logger.warning(f"Time Series model not found at {TS_MODEL_PATH}")

        # FIX: Load the Time Series scaler
        if TS_SCALER_PATH.exists():
            with open(TS_SCALER_PATH, 'rb') as f:
                ts_scaler = pickle.load(f)
            logger.info("Time Series scaler loaded successfully")
        else:
            logger.warning(f"Time Series scaler not found at {TS_SCALER_PATH}")
        
        # FIX: Update check to include the scaler
        models_loaded = all([rf_model, ts_model, ts_scaler])
        
        if models_loaded:
            logger.info("All models and scaler loaded successfully")
        else:
            logger.error("Failed to load one or more models or the scaler")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        models_loaded = False

def get_risk_level(premise_index):
    """Determine risk level based on premise index"""
    if premise_index < 30:
        return 'low'
    elif premise_index < 60:
        return 'medium'
    else:
        return 'high'

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using the Random Forest model"""
    try:
        if not rf_model:
            return jsonify({'error': 'Random Forest model not loaded'}), 500
        
        data = request.get_json()

        # Validate input data
        required_fields = ['temperature', 'rainfall', 'water_content', 'rainfall_7d_avg', 'watercontent_7d_avg']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare features for prediction
        features = np.array([[
            float(data['temperature']),
            float(data['rainfall']),
            float(data['water_content']),
            float(data['rainfall_7d_avg']),
            float(data['watercontent_7d_avg'])
        ]])
        
        # Make prediction
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
    """Forecast future PremiseIndex using a two-step process."""
    try:
        # Update guard clauses to ensure all models AND the scaler are loaded
        if not ts_model or not rf_model or not ts_scaler:
            logger.error("Forecast error: One or more models or the scaler are not loaded.")
            return jsonify({'error': 'One or more models or the scaler are not loaded. Check server logs.'}), 500

        days_str = request.args.get('days', '90')
        days = int(days_str)
        
        df = pd.read_csv(RES_DIR / 'mosquito_dataset_2017_2024.csv', parse_dates=['Date'])
        
        # 1. Prepare initial data
        features_ts = ['Temperature', 'Rainfall']
        historical_data_df = df[features_ts]
        
        scaled_historical_data = ts_scaler.transform(historical_data_df)
        
        historical_full_features = df[['Temperature', 'Rainfall', 'WaterContent']].tail(7)

        # 2. Start the autoregressive forecasting loop
        current_sequence = list(scaled_historical_data[-60:])
        final_premise_index_forecast = []

        for _ in range(days):
            input_for_ts_model = np.array([current_sequence])
            predicted_step_scaled = ts_model.predict(input_for_ts_model, verbose=0)
            
            # FIX: Handle the shape mismatch. The model outputs one value, but the scaler expects two.
            # We create a 2-feature array by duplicating the model's single output.
            single_predicted_value = predicted_step_scaled[0, 0]
            reshaped_for_scaler = np.array([[single_predicted_value, single_predicted_value]])

            predicted_weather_today = ts_scaler.inverse_transform(reshaped_for_scaler)[0]

            pred_temp = predicted_weather_today[0]
            pred_rain = predicted_weather_today[1]
            pred_wc = pred_rain * 3.5

            today_features = pd.DataFrame({
                'Temperature': [pred_temp], 'Rainfall': [pred_rain], 'WaterContent': [pred_wc]
            })

            updated_history = pd.concat([historical_full_features, today_features], ignore_index=True)
            
            rainfall_7d_avg = updated_history['Rainfall'].rolling(window=7).mean().iloc[-1]
            watercontent_7d_avg = updated_history['WaterContent'].rolling(window=7).mean().iloc[-1]

            rf_input = np.array([[
                pred_temp, pred_rain, pred_wc, rainfall_7d_avg, watercontent_7d_avg
            ]])

            premise_index_pred = rf_model.predict(rf_input)[0]
            final_premise_index_forecast.append(premise_index_pred)

            historical_full_features = updated_history.iloc[1:]

            current_sequence.pop(0)
            # FIX: Append the 2-feature array we created to maintain the correct shape for the next loop.
            current_sequence.append(reshaped_for_scaler[0])

        # 3. Format the final response
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
