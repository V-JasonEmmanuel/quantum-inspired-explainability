@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo  QIEMF Live Demo - Interactive Web Frontend
echo  Quantum-Inspired Explainability Metrics for Regulated-Sector
echo  Deployment - Clinical Breast Cancer Diagnosis Validation
echo ================================================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Please install Python 3.11 or later from https://www.python.org/downloads/
    echo and ensure it is added to PATH, then re-run this script.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/4] Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment .venv already exists, reusing it.
)

echo [2/4] Activating virtual environment ...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate the virtual environment.
    pause
    exit /b 1
)

echo [3/4] Installing dependencies from requirements.txt ...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo [4/4] Launching the QIEMF live demo in your browser ...
echo.
echo The app will open at http://localhost:8501
echo Press CTRL+C in this window to stop the demo.
echo.
streamlit run app.py

pause
