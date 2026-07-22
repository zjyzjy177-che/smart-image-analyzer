#!/bin/bash
# ==============================================
# 智能图片分析网站 — 一键启动脚本 (macOS / Linux)
# 负责人：组员 B
# 用法：在终端运行: bash run.sh
# ==============================================

echo "========================================"
echo "  🖼️  智能图片分析网站 - 启动中..."
echo "========================================"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python，请先安装 Python 3.9+"
    exit 1
fi

echo "✅ Python 版本: $(python3 --version)"

# 检查依赖是否安装
echo "📦 检查依赖..."
python3 -c "import torch; import ultralytics; import gradio; import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  依赖未完全安装，正在安装..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败，请手动运行: pip install -r requirements.txt"
        exit 1
    fi
    echo "✅ 依赖安装完成"
else
    echo "✅ 依赖已安装"
fi

# 创建必要目录
mkdir -p images/sample

# 启动应用
echo ""
echo "🚀 正在启动应用..."
echo "   本地访问: http://localhost:7860"
echo "   按 Ctrl+C 停止服务"
echo ""

python3 app.py
