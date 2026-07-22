@echo off
REM ==============================================
REM Smart Image Analyzer - One-click Launcher (Windows)
REM Maintainer: Member B
REM Usage: Double-click run.bat
REM ==============================================

title Smart Image Analyzer

echo ========================================
echo   Smart Image Analyzer - Starting...
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo [OK] Python version:
python --version
echo.

REM Check dependencies
echo [INFO] Checking dependencies...
python -c "import torch; import ultralytics; import gradio; import cv2; import numpy; import PIL; import facenet_pytorch" 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Dependencies missing, installing...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    if %errorlevel% neq 0 (
        echo [ERROR] Install failed. Try: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies ready
)

REM Create directories
if not exist images\sample mkdir images\sample

echo.
echo [INFO] Starting application...
echo   Open http://localhost:7860 in your browser
echo   Press Ctrl+C to stop
echo.

REM Open browser after a short delay
start "" http://localhost:7860

python app.py
pause
