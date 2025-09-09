@echo off
echo Starting Dengue Prediction Backend Server...
echo.

REM Check if models directory exists
if not exist "models" (
    echo Creating models directory...
    mkdir models
    echo.
    echo WARNING: Models directory was empty!
    echo Please place your trained models in the 'models' folder:
    echo - random_forest_regression_model.pkl
    echo - timeseries_model.keras
    echo.
)

REM Check if model files exist
if not exist "models\random_forest_regression_model.pkl" (
    echo WARNING: random_forest_regression_model.pkl not found in models folder
)

if not exist "models\timeseries_model.keras" (
    echo WARNING: timeseries_model.keras not found in models folder
)

echo Starting Flask server...
python app.py
pause