#!/bin/bash
# ==============================================
# Smart Image Analyzer - One-click Launcher (macOS / Linux)
# Maintainer: Member B
# Usage: bash run.sh
# ==============================================

echo "========================================"
echo "  Smart Image Analyzer - Starting..."
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.9+"
    exit 1
fi

echo "[OK] Python version: $(python3 --version)"
echo ""

# Check dependencies
echo "[INFO] Checking dependencies..."
python3 -c "import torch; import ultralytics; import gradio; import cv2; import numpy; import PIL; import facenet_pytorch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARN] Dependencies missing, installing..."
    pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    if [ $? -ne 0 ]; then
        echo "[ERROR] Install failed. Try: pip3 install -r requirements.txt"
        exit 1
    fi
    echo "[OK] Dependencies installed"
else
    echo "[OK] Dependencies ready"
fi

# Create directories
mkdir -p images/sample

echo ""
echo "[INFO] Starting application..."
echo "  Open http://localhost:7860 in your browser"
echo "  Press Ctrl+C to stop"
echo ""

# Open browser (platform-specific)
if command -v open &> /dev/null; then
    open http://localhost:7860
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:7860
fi

python3 app.py
