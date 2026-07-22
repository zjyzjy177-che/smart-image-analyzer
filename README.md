# 🖼️ 智能图片分析网站

> 创新应用综合实训 · 小组项目

上传一张图片，即可体验 **物体检测 · 图像分类 · 文字识别 · 人脸检测 · 风格迁移** 五大 AI 功能。

---

## 🚀 快速开始

### 1️⃣ 安装依赖（每人都在自己电脑上装）

```bash
pip install torch torchvision ultralytics gradio numpy opencv-python pillow paddlepaddle paddleocr
```

> ⚠️ 如果 `paddleocr` 装不上，可换成 `easyocr`：`pip install easyocr`

### 2️⃣ 运行

```bash
python app.py
```

浏览器会自动打开 `http://localhost:7860`

> 也可以双击 `run.bat`（Windows）或 `bash run.sh`（Mac/Linux）

---

## 📁 项目结构

```
smart-image-analyzer/
│
├── app.py              # 🫵 主程序（组长负责） - 界面 + 总控
├── detector.py         # 🫵 YOLO 物体检测（组长负责）
│
├── classifier.py       # 🧑 A 负责 - 图像分类
├── ocr.py              # 🧑 A 负责 - 文字识别 (OCR)
│
├── face_detect.py      # 🧑 B 负责 - 人脸检测
├── style_transfer.py   # 🧑 B 负责 - 风格迁移（加分项）
│
├── run.sh              # 🧑 B 负责 - Mac/Linux 启动脚本
├── run.bat             # 🧑 B 负责 - Windows 启动脚本
│
├── requirements.txt    # 依赖列表
├── .gitignore
└── README.md
```

---

## 👥 分工

| 成员 | 负责模块 | 文件 | 难度 |
|:----:|----------|------|:----:|
| **🫵 组长** | 主程序 + YOLO 物体检测 | `app.py` + `detector.py` | ★★☆ |
| **🧑 A** | 图像分类 + 文字识别(OCR) | `classifier.py` + `ocr.py` | ★★☆ |
| **🧑 B** | 人脸检测 + 启动脚本 + 风格迁移 | `face_detect.py` + 脚本 | ★☆☆(主要) / ★★☆(加分) |

---

## 📋 开发步骤

### 第一步：每个人装环境（今天）
在你的电脑上运行：
```bash
pip install torch torchvision ultralytics gradio numpy opencv-python pillow
```
然后运行 `python app.py`，能看到网页就算成功（虽然按钮点了会报"开发中"）。

### 第二步：各写各的模块（7/25前）
> 三人同时进行，互不依赖

**🫵 组长** → 完善 `detector.py` + `app.py`
- 确保 YOLO 检测能跑通
- 等A和B写完，把他们的模块集成到 `app.py` 里

**🧑 A** → 完善 `classifier.py` + `ocr.py`
- 在各自文件里用 `if __name__ == "__main__"` 自测
- 跑通了告诉我

**🧑 B** → 完善 `face_detect.py` + `run.sh`/`run.bat`
- 人脸检测写完后，有空再搞风格迁移
- 启动脚本保证双击就能跑

### 第三步：集成（7/27前）
组长把三个人的代码合到 `app.py`，所有功能一起跑通。

### 第四步：美化 + 收尾（8月）
写报告、做PPT、录演示视频。

---

## 🎯 各模块技术要点

### detector.py（组长）
- 使用 `ultralytics` 库加载 `yolov8n.pt`（自动下载）
- 核心函数 `detect_objects(image)` → 返回标注图 + 检测结果列表
- 无需训练，直接用预训练模型

### classifier.py（A）
- 使用 `torchvision.models.resnet50` 预训练模型
- 首次运行自动下载权重（~100MB）
- 识别 1000 种常见物体类别

### ocr.py（A）
- 使用 `paddleocr`，支持中英文
- 识别图片中的文字并标注位置
- 备选方案：`easyocr`

### face_detect.py（B）
- 使用 OpenCV 内置 Haar Cascade（无需下载）
- 翻墙后可升级为 DNN 模型，精度更高

### style_transfer.py（B - 加分项）
- 简化版颜色映射（先保证能跑）
- 可选升级：预训练风格迁移模型

---

## 🧪 自测方式

每个模块都可以独立运行自测：
```bash
python detector.py      # 测试物体检测
python classifier.py    # 测试图像分类
python ocr.py           # 测试文字识别
python face_detect.py   # 测试人脸检测
python style_transfer.py # 测试风格迁移
```

---

## 📝 Git 协作

```bash
# 克隆
git clone https://github.com/zjyzjy177-che/smart-image-analyzer.git

# 每次开始工作前先拉取最新
git pull

# 改完后提交
git add .
git commit -m "A: 完成了图像分类模块"
git push
```

> ⚠️ 改别人的文件前先 git pull，push 前也先 git pull 避免冲突
