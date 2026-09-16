@echo off
title Quantum IoT Anomaly Detection - Test Suite
cls

echo =======================================================
echo   RUNNING FULL AUTOMATED UNIT & INTEGRATION TESTS
echo =======================================================
echo.

set PYTHON_EXE=C:\Saatvik\Python\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

cd /d "%~dp0VS Code\ML-Learning"

"%PYTHON_EXE%" -m pytest tests/ -v

echo.
echo =======================================================
echo Tests completed.
pause
