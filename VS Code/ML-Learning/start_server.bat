@echo off
title Quantum IoT Anomaly Detection - Launch Server
cls

echo =======================================================
echo   QUANTUM-ENABLED IoT ANOMALY DETECTION DASHBOARD
echo =======================================================
echo.
echo [1/3] Initializing environment...

:: Set Python Path
set PYTHON_EXE=C:\Saatvik\Python\python.exe
if not exist "%PYTHON_EXE%" (
    echo Specific Python path not found, falling back to system PATH...
    set PYTHON_EXE=python
)

:: Set Project Directory
cd /d "%~dp0"

echo [2/3] Project directory: %CD%
echo [3/3] Launching Streamlit Research Server on port 8501...
echo.
echo Dashboard URL: http://localhost:8501
echo Press Ctrl+C in this window to stop the server anytime.
echo.

:: Automatically open default browser
start "" http://localhost:8501

:: Start Streamlit Server
"%PYTHON_EXE%" -m streamlit run app.py --server.port 8501

pause
