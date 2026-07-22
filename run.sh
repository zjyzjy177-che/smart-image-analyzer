#!/bin/bash
# ==============================================
# 智能图片分析系统 — 一键启动 (macOS / Linux)
# ==============================================

echo "========================================"
echo "  智能图片分析系统 - 启动中..."
echo "========================================"
echo ""

# 进入脚本所在目录
cd "$(dirname "$0")" || exit 1

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3，请安装 Python 3.9+"
    exit 1
fi

echo "[✓] Python: $(python3 --version)"

# 检查核心依赖
python3 -c "import gradio; import ultralytics; import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[提示] 安装依赖中..."
    pip3 install gradio ultralytics opencv-python numpy pillow torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    if [ $? -ne 0 ]; then
        echo "[错误] 安装失败，请手动运行: pip3 install -r requirements.txt"
        exit 1
    fi
    echo "[✓] 依赖安装完成"
else
    echo "[✓] 依赖已就绪"
fi

mkdir -p images/sample

echo ""
echo "  正在启动浏览器..."
echo "  如未自动打开，请访问 http://localhost:7860"
echo "  按 Ctrl+C 停止服务"
echo ""

# 启动应用
python3 app.py &

# 等服务器就绪后打开浏览器
sleep 3
if command -v open &>/dev/null; then
    open http://localhost:7860
elif command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:7860
fi

wait
