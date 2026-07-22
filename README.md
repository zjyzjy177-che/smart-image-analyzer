# 🖼️ 智能图片分析网站

> **创新应用综合实训 · 小组项目**
>
> 上传一张图片，即可体验 **物体检测 · 图像分类 · 文字识别 · 人脸检测 · 风格迁移** 五大 AI 功能。
>
> 项目链接：https://github.com/zjyzjy177-che/smart-image-analyzer

---

## 📦 第一步：每个人装环境

在你自己电脑上打开终端，运行：

```bash
# 克隆项目
git clone https://github.com/zjyzjy177-che/smart-image-analyzer.git
cd smart-image-analyzer

# 安装依赖
pip3 install torch torchvision ultralytics gradio numpy opencv-python pillow
```

> ⚠️ 如果 `pip3` 找不到，试试 `pip` 或 `python3 -m pip install ...`
>
> ⏱️ 首次安装大概 5-10 分钟，Torch 和 Ultralytics 比较大

装完后测试一下界面：
```bash
python3 app.py
```
浏览器打开 `http://localhost:7860`，能看到网页就算成功（目前只有物体检测能用，其他按钮会显示"开发中"）。

---

## 👥 第二步：各自开发（完整任务看板）

### 🫵 组长任务看板

| 文件 | 状态 | 说明 |
|:----:|:----:|------|
| `detector.py` | ✅ **已完成** | YOLOv8 物体检测，直接就能用 |
| `app.py` | ✅ **已完成** | Gradio 主界面已写好 |
| **集成 A+B 的代码** | ⏳ 7/25后 | A和B跑通后，取消几行注释即可 |

**你不需要写新代码**，你的任务是：
1. 确认 YOLO 检测跑通（已跑通 ✅）
2. 等 A 和 B 跑通后，在 `app.py` 里取消注释（到时候我告诉你是哪几行）
3. 管好进度，催 A 和 B

---

### 🧑 A — 完整任务看板

**你要改的文件：** `classifier.py` + `ocr.py`

#### 🎯 任务 1：图像分类（`classifier.py`）

**目标：** 上传一张图片 → 告诉你是猫、狗、汽车、飞机...

**难度：** ★☆☆（调包就行）

**具体步骤：**

1. 装依赖
   ```bash
   pip3 install torch torchvision pillow numpy
   ```

2. 运行测试
   ```bash
   cd smart-image-analyzer
   python3 classifier.py
   ```
   它会装 ResNet50 模型（~100MB），然后输出类似：
   ```
   🏷️ 分类结果: 平顶山
   📊 置信度: 5.43%
   ```
   > 因为测试图是纯色块，置信度低是正常的。上传真实照片置信度会到 80%+

3. **你的代码要做的 3 件事（代码里都有框架了，你只需要完善）：**

   **① `classifier.py` 里的 `classify_image()` 函数**
   - 代码已经写好了，你只需要确保能跑通
   - 核心逻辑：加载 ResNet → 预处理图片 → 推理 → 返回 (标签, 置信度)

   **② 加一点"显示 Top-3"功能（加分）**
   - 现在只返回 Top-1，你可以改成返回 Top-3
   - 参考代码里的 `torch.topk(probabilities, 3)` 已经有了

   **③ 跑通了告诉组长："我分类跑通了"**

4. **常见问题：**
   - ❌ 网络下载模型慢 → 等一会儿（100MB 大概 2-3 分钟）
   - ❌ `cannot connect` → 换个网络环境 / 用手机热点
   - ❌ 报错说 `models.resnet50` 找不到 → 更新 torchvision：`pip3 install --upgrade torchvision`

---

#### 🎯 任务 2：OCR 文字识别（`ocr.py`）

**目标：** 上传一张带文字的图片 → 提取出里面的文字

**难度：** ★★☆（PaddleOCR 容易装不上，有备选方案）

**具体步骤：**

**方案 A（优先）：用 PaddleOCR**
```bash
pip3 install paddlepaddle paddleocr
python3 ocr.py
```

**方案 B（备选）：如果 PaddleOCR 装不上，用 easyocr**
```bash
pip3 install easyocr
```

然后把 `ocr.py` 里这段代码换掉：
```python
# 找到 get_ocr() 函数，把里面的内容替换为：
import easyocr
ocr = easyocr.Reader(['ch_sim', 'en'])
return ocr
```

**效果预期：**
```bash
$ python3 ocr.py
📝 识别文字: Hello World 你好世界
```

**常见问题：**
- ❌ PaddleOCR 装不上 → 直接用方案 B（easyocr），效果差不多
- ❌ easyocr 也装不上 → 告诉组长，还有方案 C（用 pytesseract）

---

### 🧑 B — 完整任务看板

**你要改的文件：** `face_detect.py` + 启动脚本

#### 🎯 任务 1：人脸检测（`face_detect.py`）

**目标：** 上传一张有人脸的照片 → 框出人脸位置

**难度：** ★☆☆（OpenCV 内置功能）

**具体步骤：**

1. 装依赖
   ```bash
   pip3 install opencv-python numpy pillow
   ```

2. 运行测试
   ```bash
   cd smart-image-analyzer
   python3 face_detect.py
   ```
   输出类似：
   ```
   检测到 0 张人脸
   ⚠️ 测试图可能太简单，换一张真实照片试试
   ```
   > 测试图是画出来的椭圆，不是真人脸，所以检测不到是正常的。
   > **你需要下载一张真人照片来测试。**

3. **你的代码要做的 2 件事：**

   **① 找一张真人照片测试**
   - 随便从网上下载一张有人脸的照片
   - 在 `face_detect.py` 文件底部找到测试代码，把文件路径换成你的照片
   ```python
   # 大概在第 100 行附近，找到这行：
   test_img = np.zeros((400, 400, 3), dtype=np.uint8)
   # 替换为：
   test_img = cv2.imread("你的照片路径.jpg")
   test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
   ```

   **② 跑通后告诉组长 + 组长让你干啥就干啥**

4. **常见问题：**
   - ❌ `cv2.error: ...` → 检查照片路径对不对
   - 检测不到人脸 → 换正面大头照试试

---

#### 🎯 任务 2：启动脚本（`run.sh` + `run.bat`）

**目标：** 双击就能启动项目，不用打命令

**难度：** ★☆☆

**具体步骤：**

两个脚本都已经写好了，你只需要 **确保它们在你电脑上能双击运行**：

- **Mac 用户**：双击 `run.sh` → 如果提示"没有权限"，先运行 `chmod +x run.sh`
- **Windows 用户**：双击 `run.bat`

**如果双击报错：** 检查 Python 有没有加到环境变量。在终端运行 `python3` / `python` 看能不能找到。

---

#### 🎯 任务 3：风格迁移 — 加分项（`style_transfer.py`）

**目标：** 把照片变成梵高/莫奈风格

**难度：** ★★☆

**如果前面的任务都搞定了，还有时间再做这个。**

现在代码里是"伪风格迁移"（只是改颜色），真正的风格迁移需要额外装模型。如果你想做：

```bash
pip3 install torch torchvision pillow
```

然后用预训练的风格迁移模型替换 `style_transfer.py` 里的 `transfer_style()` 函数。

**注意：这个功能不一定要做，评分时前面 4 个功能已经够了。**

---

## 🔗 第三步：集成（7/27前，组长做）

等 A 和 B 都跑通后，组长在 `app.py` 里把 `# TODO: A 实现后取消注释` 之类的行取消注释，所有功能就一起上线了。

具体取消哪些行，到时候 A 和 B 说"跑通了"，你问我，我告诉你怎么改。

---

## 📝 Git 协作规范

```bash
# 第一次克隆
git clone https://github.com/zjyzjy177-che/smart-image-analyzer.git

# 每次开始写代码前 —— 拉取最新代码
git pull

# 写完代码后 —— 提交并推送
git add 你的文件.py      # 只加你自己的文件，不要 git add .
git commit -m "A: 完成了图像分类模块"
git push
```

**⚠️ 关键规则：**
- **只改你自己的文件！** 组长负责 `app.py` 和 `detector.py`，A 负责 `classifier.py` 和 `ocr.py`，B 负责 `face_detect.py` 和脚本
- 不要 `git add .` 提交别人的文件
- `git push` 之前先 `git pull`，避免冲突
- 如果提示 "merge conflict"，不要慌，在群里喊我

---

## 📅 时间线

| 时间 | 目标 |
|:----:|------|
| **7/22-7/24** | 各自装环境 + 跑通自己的模块 |
| **7/25-7/27** | 组长集成，所有功能一起跑通 |
| **7/28-8/10** | 美化界面 + B 做风格迁移（可选） |
| **8/10-8/20** | 写报告 + 做PPT + 录演示视频 + 提交 |

---

## 🧪 自测命令汇总

每个人都可以独立测试自己的模块：

```bash
python3 detector.py       # 🫵 组长 - 测试物体检测
python3 classifier.py     # 🧑 A - 测试图像分类
python3 ocr.py            # 🧑 A - 测试文字识别
python3 face_detect.py    # 🧑 B - 测试人脸检测
python3 style_transfer.py # 🧑 B - 测试风格迁移
python3 app.py            # 全员 - 启动完整应用
```

---

## ❓ FAQ

**Q: pip3 装东西报错怎么办？**
A: 先在群里问，大概率是：网络问题（换个源）、版本问题（升级 pip）、或者少装了啥。

**Q: 模型下载很慢怎么办？**
A: 第一次下载需要等。YOLO（6MB）、ResNet（100MB）、PaddleOCR（~50MB）。换个快的 WiFi 或者用手机热点试试。

**Q: 我改完代码 push 报错怎么办？**
A: 先 `git pull` 再 `git push`。如果还有问题，在群里喊。

**Q: 完全不会 Python / 看不懂代码怎么办？**
A: 不需要全看懂。你只需要：找到你的文件 → 运行它 → 确保不报错 → 告诉组长。代码已经有框架了。
