@echo off
REM ==============================================
REM 智能图片分析网站 — 一键启动脚本 (Windows)
REM 负责人：组员 B
REM 用法：双击运行 run.bat
REM ==============================================

echo ========================================
echo   🖼️  智能图片分析网站 - 启动中...
echo ========================================

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)

echo ✅ Python 版本:
python --version

REM 检查依赖
echo 📦 检查依赖...
python -c "import torch; import ultralytics; import gradio; import cv2" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  依赖未完全安装，正在安装...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ 依赖安装失败，请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo ✅ 依赖安装完成
) else (
    echo ✅ 依赖已安装
)

REM 创建目录
if not exist images\sample mkdir images\sample

echo.
echo 🚀 正在启动应用...
echo    本地访问: http://localhost:7860
echo    按 Ctrl+C 停止服务
echo.

python app.py
pause
