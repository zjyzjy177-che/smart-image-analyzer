# 🖼️ 智能图片分析网站

> **创新应用综合实训 · 小组项目**
>
> 上传一张图片，即可体验 **物体检测 · 图像分类 · 文字识别 · 人脸检测 · 风格迁移** 五大 AI 功能。
>
> 项目链接：https://github.com/zjyzjy177-che/smart-image-analyzer

---

## 📦 环境安装

在本地终端运行：

```bash
# 克隆项目
git clone https://github.com/zjyzjy177-che/smart-image-analyzer.git
cd smart-image-analyzer

# 安装依赖
pip3 install torch torchvision ultralytics gradio numpy opencv-python pillow
```

> ⚠️ 如果 `pip3` 不可用，尝试 `pip` 或 `python3 -m pip install ...`

装完后运行以下命令确认界面可正常启动：
```bash
python3 app.py
```
浏览器打开 `http://localhost:7860`，看到网页即表示运行成功（目前仅物体检测功能可用，其余功能标注为"开发中"）。

---

## 👥 人员分工

| 成员 | 负责模块 | 文件 | 难度 |
|:----:|----------|------|:----:|
| **组长** | 主程序 + YOLO 物体检测 + 集成 | `app.py` + `detector.py` | ★★☆ |
| **组员 A** | 图像分类 + 文字识别(OCR) | `classifier.py` + `ocr.py` | ★★☆ |
| **组员 B** | 人脸检测 + 启动脚本 + 风格迁移(可选) | `face_detect.py` + `run.sh`/`run.bat` | ★☆☆~★★☆ |

---

## 🎯 组员 A 任务看板

**负责文件：** `classifier.py` + `ocr.py`

---

### 任务 1：图像分类（`classifier.py`）

**目标：** 上传图片 → 输出图片内容类别（猫、狗、汽车、飞机等）

**难度：** ★☆☆

**操作步骤：**

1. 安装依赖
   ```bash
   pip3 install torch torchvision pillow numpy
   ```

2. 运行自测
   ```bash
   cd smart-image-analyzer
   python3 classifier.py
   ```
   首次运行自动下载 ResNet50 模型（~100MB），输出示例：
   ```
   🏷️ 分类结果: 平顶山
   📊 置信度: 5.43%
   ```
   > 测试图是纯色块，置信度低是正常现象。传入真实照片后置信度会显著提升。

3. **需要完成的工作：**
   - 确保 `classifier.py` 能正常运行并输出分类结果
   - 代码中的 `classify_image()` 函数已提供完整实现，无需额外编写

4. **常见问题：**
   - ❌ 模型下载慢 → 等待约 2-3 分钟
   - ❌ 网络连接失败 → 更换网络环境
   - ❌ torchvision 版本问题 → 执行 `pip3 install --upgrade torchvision`

---

### 任务 2：OCR 文字识别（`ocr.py`）

**目标：** 上传带文字的图片 → 提取并显示图片中的文字

**难度：** ★★☆

**操作步骤：**

**方案 A（优先）：PaddleOCR**
```bash
pip3 install paddlepaddle paddleocr
python3 ocr.py
```

**方案 B（备选）：若 PaddleOCR 安装失败，换用 easyocr**
```bash
pip3 install easyocr
```

然后在 `ocr.py` 的 `get_ocr()` 函数中替换为：
```python
import easyocr
ocr = easyocr.Reader(['ch_sim', 'en'])
return ocr
```

**预期输出：**
```
📝 识别文字: Hello World 你好世界
```

**常见问题：**
- ❌ PaddleOCR 安装失败 → 改用 easyocr
- ❌ easyocr 仍失败 → 告知组长，换用 pytesseract

---

## 🎯 组员 B 任务看板

**负责文件：** `face_detect.py` + `run.sh` / `run.bat` (+ `style_transfer.py` 可选)

---

### 任务 1：人脸检测（`face_detect.py`）

**目标：** 上传含人脸的图片 → 框出人脸位置

**难度：** ★☆☆

**操作步骤：**

1. 安装依赖
   ```bash
   pip3 install opencv-python numpy pillow
   ```

2. 准备测试图片
   - 下载一张包含人脸的真人照片
   - 修改 `face_detect.py` 底部测试代码中的图片路径

3. 运行自测
   ```bash
   cd smart-image-analyzer
   python3 face_detect.py
   ```

4. 测试代码修改示例：
   ```python
   # 将原有测试图创建代码替换为：
   test_img = cv2.imread("照片路径.jpg")
   test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
   ```

5. 完成标准：能正确框出照片中的人脸位置

**常见问题：**
- ❌ cv2 报错 → 检查图片路径是否正确
- 检测不到人脸 → 换用正面、光线充足的照片

---

### 任务 2：启动脚本（`run.sh` + `run.bat`）

**目标：** 双击即可启动项目，无需输入命令

**难度：** ★☆☆

**两个脚本已编写完成，需要验证：**

- **macOS / Linux**：终端执行 `bash run.sh` 或 `chmod +x run.sh && ./run.sh`
- **Windows**：双击 `run.bat`

**常见问题：**
- 提示"Python 未找到" → 检查 Python 是否加入系统 PATH

---

### 任务 3：风格迁移（可选加分项，`style_transfer.py`）

**目标：** 将照片转换为梵高/莫奈等艺术风格

**难度：** ★★☆

**前置条件：** 任务 1 和任务 2 完成后有余力再执行

**当前状态：** 代码中提供的是简化的颜色映射方案（仅调整色调）。如需真实风格迁移效果，可集成预训练模型。

**可选升级方案：**
```bash
pip3 install torch torchvision pillow
```
替换 `style_transfer.py` 中 `transfer_style()` 函数的实现。

**注意：** 即使不做此功能，前 4 个核心功能已满足项目要求。

---

## 🔗 集成步骤（组长执行）

待组员 A 和 B 确认各自模块运行正常后，组长在 `app.py` 中执行以下操作：

1. 找到文件开头的 `import` 区域，取消以下行的注释：
   ```python
   from classifier import classify_image
   from ocr import extract_text
   from face_detect import detect_faces
   ```

2. 找到 `process_image()` 函数中对应各功能的 `# TODO` 注释行，取消注释并删除下方的占位返回语句

具体修改位置届时由组长确认后执行，可咨询指导。

---

## 📝 Git 协作规范

```bash
# 首次克隆
git clone https://github.com/zjyzjy177-che/smart-image-analyzer.git

# 每次开始前拉取最新代码
git pull

# 修改后提交
git add 个人负责的文件.py      # 只提交自己负责的文件
git commit -m "A: 完成了图像分类模块"
git push
```

**⚠️ 规则：**
- 每人只修改自己负责的文件，不要改动他人文件
- 不要使用 `git add .`，应指定具体文件
- `git push` 前先 `git pull`
- 遇到 merge conflict 时在群里沟通

---

## 📅 时间线

| 时间 | 目标 |
|:----:|------|
| **7/22-7/24** | 环境搭建 + 各自模块跑通 |
| **7/25-7/27** | 组长集成，全部功能联调完成 |
| **7/28-8/10** | 界面美化 + 风格迁移（可选） |
| **8/10-8/20** | 报告 + PPT + 演示视频 + 提交 |

---

## 🧪 自测命令

```bash
python3 detector.py       # 测试物体检测（组长）
python3 classifier.py     # 测试图像分类（A）
python3 ocr.py            # 测试文字识别（A）
python3 face_detect.py    # 测试人脸检测（B）
python3 style_transfer.py # 测试风格迁移（B，可选）
python3 app.py            # 启动完整应用
```

---

## ❓ 常见问题

**Q: pip3 安装报错？**
A: 可能是网络问题（换国内源）、版本问题（升级 pip）、或缺少依赖。在群里沟通。

**Q: 模型下载太慢？**
A: 首次下载需等待：YOLO（6MB）、ResNet（100MB）、PaddleOCR（~50MB）。建议使用稳定网络。

**Q: git push 报错？**
A: 先 `git pull` 再重试 `git push`。

**Q: 代码看不太懂？**
A: 代码已有完整框架，只需运行自测确认功能正常即可，不需要完全理解每行代码的含义。
