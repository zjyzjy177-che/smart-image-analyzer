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
pip3 install torch torchvision ultralytics gradio numpy opencv-python pillow facenet-pytorch
```

> ⚠️ 如果 `pip3` 不可用，尝试 `pip` 或 `python3 -m pip install ...`
>
> ⚠️ 新版 `huggingface-hub` 可能与 `gradio 4.x` 不兼容。如启动报 `HfFolder` 错误，执行：
> ```bash
> pip3 install "huggingface-hub>=0.19.0,<1.0.0"
> ```

装完后运行以下命令确认界面可正常启动：
```bash
python3 app.py
```
浏览器打开 `http://localhost:7860`，看到网页即表示运行成功（物体检测、人脸检测、风格迁移已可用，其余功能标注为"开发中"）。

---

## 👥 人员分工

| 成员 | 负责模块 | 文件 | 难度 |
|:----:|----------|------|:----:|
| **组长** | 主程序 + YOLO 物体检测 + 集成 | `app.py` + `detector.py` | ★★☆ |
| **组员 A** | 图像分类 + 文字识别(OCR) | `classifier.py` + `ocr.py` | ★★☆ |
| **组员 B** | 人脸检测 + 启动脚本 + 风格迁移 | `face_detect.py` + `run.sh`/`run.bat` + `style_transfer.py` | ★☆☆~★★☆ |

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

**负责文件：** `face_detect.py` + `run.sh` + `run.bat` + `style_transfer.py`

---

### 任务 1：人脸检测（`face_detect.py`）✅ 已完成

**目标：** 上传含人脸的图片 → 框出人脸位置

**难度：** ★☆☆

**实现方案：三级降级策略，自动选择可用方案**

| 优先级 | 方案 | 模型 | 精度 | 说明 |
|:---:|------|------|:---:|------|
| 1 | MTCNN | facenet-pytorch | ★★★ | 首选，多尺度级联，侧脸/小脸检出率高 |
| 2 | DNN (SSD) | OpenCV Caffe | ★★☆ | 需下载 `models/` 下的模型文件 |
| 3 | Haar Cascade | OpenCV 内置 | ★☆☆ | 无需下载，最终兜底方案 |

**MTCNN 原理（首选方案）：**

MTCNN（Multi-task Cascaded Convolutional Networks）采用三阶段级联架构：

1. **P-Net（Proposal Network）：** 以不同尺度（图像金字塔）扫描图片，快速生成候选的人脸区域边界框
2. **R-Net（Refine Network）：** 对 P-Net 产出的候选框进行精炼，过滤大部分非人脸区域
3. **O-Net（Output Network）：** 最终精细化定位，输出人脸边界框和 5 个关键点（双眼、鼻尖、嘴角）

每一阶段都是独立的 CNN，输入分辨率逐级提高，显式处理多尺度问题，因此对小脸、侧脸、密集型多人场景的检出率显著优于单阶段 SSD。

**DNN 多尺度检测原理（备选方案）：**

SSD 单次检测将图片缩放至 300×300，小脸可能被压缩到仅几个像素。本模块通过多尺度推理解决：
- 分别以 1.0x、1.5x、2.0x 放大图片后送入网络
- 检测结果缩放回原图坐标
- 使用 NMS（非极大值抑制，IoU < 0.4）合并重叠框

**操作步骤：**

1. 安装依赖
   ```bash
   pip3 install opencv-python numpy pillow facenet-pytorch
   ```

2. 准备测试图片放到 `images/sample/face.jpg`

3. 运行自测
   ```bash
   cd smart-image-analyzer
   python3 face_detect.py
   ```

4. 完成标准：能正确框出照片中的人脸位置

**常见问题：**
- ❌ cv2 报错 → 检查图片路径是否正确
- 检测不到人脸 → 换用正面、光线充足的照片
- MTCNN 加载失败 → 确认 `facenet-pytorch` 已安装

---

### 任务 2：启动脚本（`run.sh` + `run.bat`）✅ 已验证

**目标：** 一键启动项目，无需手动输入命令

**难度：** ★☆☆

**使用方式：**

- **macOS / Linux**：终端执行 `bash run.sh`
- **Windows**：双击 `run.bat`

脚本会自动检查依赖是否安装，通过浏览器打开 `http://localhost:7860`。

**常见问题：**
- 提示"Python 未找到" → 检查 Python 是否加入系统 PATH

---

### 任务 3：风格迁移（`style_transfer.py`）✅ 已完成

**目标：** 将照片转换为两种艺术风格

**难度：** ★★☆

**已实现风格：**

| 风格 | 核心算法 | 速度 |
|------|------|:---:|
| 素描·黑白线条画 | 直方图均衡化 + 三层颜色减淡 + 纸面颗粒纹理 + 锐化 | <1s |
| 二次元·动漫风格 | 双边滤波 + K-means 颜色量化(k=16) + 自适应阈值轮廓 | ~2s |

两种方案均为纯 OpenCV 计算机视觉算法实现，无需 GPU 或预训练深度学习模型，秒级出结果。

**素描风格原理：**

素描效果模拟的是铅笔素描的核心技法——通过线条密度和阴影深浅来表现被画物体的形态。

1. **直方图均衡化：** 对灰度图进行对比度增强，将像素值分布均匀化。这一步让暗部更暗、亮部更亮，后续线条和阴影层次更丰富。

2. **反相（Invert）：** 灰度图取反（`255 - gray`），原本亮的区域变暗，暗的区域变亮。这是颜色减淡的前置条件。

3. **三层颜色减淡（Color Dodge Blend）：**
   - 颜色减淡的数学公式：`result = gray / (255 - blurred) × 256`
   - **细线层（5×5 高斯核）：** 小模糊核仅扩散邻近像素，留下的反相边缘锐利 → 铅笔轮廓线
   - **结构层（13×13 高斯核）：** 中等模糊产生的反相过渡 → 面部结构和纹理褶皱
   - **阴影层（31×31 高斯核）：** 大模糊核产生的柔和渐变 → 大面积光影和深度感
   - 三层按 `0.5 : 0.3 : 0.15` 的权重叠加：细线为主保持锐度，中景加结构信息，粗层只提供柔和阴影

4. **纸面纹理：** 叠加标准差为 8 的正态分布随机噪声（再经 3×3 高斯平滑），模拟素描纸的颗粒质感。

5. **锐化滤波：** 3×3 锐化卷积核（中心权重 7，周围 -0.5 ~ -1）增强边缘对比，让铅笔线条更加清晰利落。

**二次元动漫风格原理：**

动漫风格的核心特征是色块分明（"平涂"）且边界有清晰墨线。

1. **双边滤波（Bilateral Filter，3 次迭代）：** 同时考虑空间距离和颜色差异的平滑滤波——同一物体内部颜色被抹平，但物体边缘（颜色跳变处）被保留。`d=9, sigmaColor=75, sigmaSpace=75` 的参数组合在平滑皮肤和背景的同时保护了眉毛、眼睛等关键边缘。

2. **K-means 颜色量化（k=16）：** 将图像所有像素聚类到 16 个颜色中心，每个像素替换为所属簇的中心颜色。这一步直接产生动漫的"平涂色块"效果——皮肤、头发、衣服各归一到少数几个纯色。

3. **自适应阈值轮廓提取：**
   - 先对灰度图做 7×7 中值模糊去除噪点和细碎纹理
   - 自适应阈值（`blockSize=9, C=2`）：以每个像素的局部邻域均值为基准做二值化，不同区域阈值不同 → 无论亮部暗部都能提取出完整的边缘线
   - 2×2 椭圆核膨胀边缘 1 次 → 墨线加粗，产生漫画勾线效果

4. **融合：** 将提取的黑色轮廓线叠加在量化后的平涂色块上——非边缘区域保留动漫色块，边缘区域呈现黑色墨线，完成整幅二次元画面。

**操作步骤：**

1. 确认依赖已安装（opencv-python, numpy 已在环境安装中覆盖）

2. 运行自测
   ```bash
   cd smart-image-analyzer
   python3 style_transfer.py
   ```

3. 自测会自动对 `images/sample/face.jpg` 应用两种风格并输出结果尺寸

**注意：** 风格迁移使用复杂的 OpenCV 计算机视觉算法，效果比原版的简单颜色调色显著提升。

---

## 🔗 集成步骤（组长执行）

待组员 A 和 B 确认各自模块运行正常后，组长在 `app.py` 中执行以下操作：

> **当前状态：** 人脸检测和风格迁移已由组员B集成完毕，物体检测由组长实现。仅图像分类和OCR待组员A完成。

1. 找到文件开头的 `import` 区域，取消以下行的注释：
   ```python
   from classifier import classify_image
   from ocr import extract_text
   # from face_detect import detect_faces      -- 已集成
   # from style_transfer import transfer_style -- 已集成
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
python3 style_transfer.py # 测试风格迁移（B）
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
