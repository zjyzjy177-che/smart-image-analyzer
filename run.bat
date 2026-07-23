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

REM Prefer Windows Python Launcher. It works even when python.exe is not on PATH.
set "PYTHON_CMD=py -3"
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    set "PYTHON_CMD=python"
    %PYTHON_CMD% --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python not found. Please install Python 3.9+
        pause
        exit /b 1
    )
)

echo [OK] Python version:
%PYTHON_CMD% --version
echo.

REM Check dependencies
echo [INFO] Checking dependencies...
%PYTHON_CMD% -c "import torch, ultralytics, gradio, cv2, numpy, PIL, facenet_pytorch, transformers, easyocr; assert int(gradio.__version__.split('.')[0]) >= 6; assert int(transformers.__version__.split('.')[0]) >= 5" 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Dependencies missing or outdated, installing...
    %PYTHON_CMD% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    if %errorlevel% neq 0 (
        echo [ERROR] Install failed. Try: py -3 -m pip install -r requirements.txt
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

%PYTHON_CMD% app.py
pause
