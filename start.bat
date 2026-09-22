@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo  QIEMF Proof-of-Concept - Automated Runner
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

echo [4/4] Running the QIEMF pipeline (qiemf.py) ...
echo.
python qiemf.py
if errorlevel 1 (
    echo.
    echo [ERROR] The QIEMF pipeline exited with an error. See logs\qiemf.log for details.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  QIEMF pipeline completed successfully.
echo  Figures : figures\
echo  Tables  : tables\
echo  Results : results\  (see results\academic_writeup.md)
echo  Logs    : logs\qiemf.log
echo ================================================================
pause
