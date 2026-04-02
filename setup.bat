@echo off
echo ============================================
echo   Sign Language Detector - Windows Setup
echo ============================================
echo.

:: Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Install dependencies
echo.
echo [2/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

:: Create necessary folders
echo.
echo [3/4] Creating directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir models 2>nul
mkdir screenshots 2>nul

echo.
echo [4/4] Setup complete!
echo.
echo ============================================
echo   NEXT STEPS:
echo   1. python src/collect_data.py
echo   2. python src/preprocess.py
echo   3. python src/train.py
echo   4. streamlit run app/streamlit_app.py
echo ============================================
echo.
pause
