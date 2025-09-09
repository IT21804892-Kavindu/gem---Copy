@echo off
echo Installing Python dependencies for Dengue Prediction Backend...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo Installing required packages...
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install tensorflow==2.13.0
pip install python-dotenv==1.0.0

echo.
echo Installation complete!
echo.
echo Next steps:
echo 1. Place your trained models in the 'models' folder:
echo    - random_forest_regression_model.pkl
echo    - timeseries_model.keras
echo 2. Run: python app.py
echo.
pause