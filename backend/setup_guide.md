# Backend Setup Guide

## Quick Setup (Windows)

### Option 1: Automatic Installation
1. Double-click `install.bat` to install all dependencies
2. Place your model files in the `models` folder
3. Double-click `start.bat` to start the server

### Option 2: Manual Installation
1. Open Command Prompt in the backend folder
2. Run: `pip install -r requirements.txt`
3. Place your model files in the `models` folder
4. Run: `python app.py`

## Model File Placement

Create a `models` folder and place these files:

```
backend/
├── models/
│   ├── random_forest_regression_model.pkl    ← Your Random Forest model
│   └── timeseries_model.keras                ← Your Time Series model
├── app.py
└── requirements.txt
```

## Model Requirements

### Random Forest Model
- File: `random_forest_regression_model.pkl`
- Format: Scikit-learn model saved with pickle
- Input: [rainfall, temperature, water_content]
- Output: Single premise index (0-100)

### Time Series Model  
- File: `timeseries_model.keras`
- Format: TensorFlow/Keras model (.keras format)
- Purpose: Generate 90-day forecasts

## Troubleshooting

### "Module not found" errors
- Run `install.bat` or `pip install -r requirements.txt`

### "Models not loaded" warning
- Ensure model files are in the `models` folder
- Check file names match exactly
- Verify models were saved in correct format

### Port already in use
- Change port in `app.py`: `app.run(port=5001)`
- Update frontend `.env`: `REACT_APP_API_URL=http://localhost:5001`

## Testing the Backend

1. Start the server: `python app.py`
2. Test health endpoint: http://localhost:5000/api/health
3. Should return: `{"status": "healthy", "models_loaded": true}`