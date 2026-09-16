@echo off
title Push Quantum Project to GitHub
cls

echo =======================================================
echo   PUSH QUANTUM IoT PROJECT TO GITHUB
echo =======================================================
echo.
echo All 43 code files, tests, and documentation are already committed locally!
echo.
echo Step 1: Go to https://github.com/new and create a new repository.
echo         (Leave "Add a README file" UNCHECKED).
echo.
echo Step 2: Copy the HTTPS repository URL.
echo         Example: https://github.com/YourUsername/Quantum-IoT-Anomaly-Detection.git
echo.
set /p REPO_URL="Enter your GitHub Repository URL: "

if "%REPO_URL%"=="" (
    echo Error: No URL entered. Aborting.
    pause
    exit /b
)

echo.
echo [1/2] Setting remote origin to %REPO_URL%...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo [2/2] Pushing main branch to GitHub...
git push -u origin main

echo.
echo =======================================================
if %ERRORLEVEL% equ 0 (
    echo SUCCESS! Your project has been pushed to GitHub.
) else (
    echo Push failed. Please verify your URL and GitHub credentials.
)
echo =======================================================
pause
