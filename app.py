"""
app.py — 智能图片分析网站 主程序
负责人：组长 张建宇
功能：Gradio Web 界面，集成所有模块
运行：python app.py
"""

import gradio as gr
import numpy as np
from PIL import Image

# 导入各模块
from detector import detect_objects
# from classifier import classify_image     # 组员A 岳思铭 负责
# from ocr import extract_text              # 组员A 岳思铭 负责
from face_detect import detect_faces        # 组员B 李汇洋 负责
from style_transfer import transfer_style   # 组员B 李汇洋 负责


def process_image(image, mode, conf_threshold, style_name):
    """
    核心处理函数：根据选择的功能模式调用不同的模块
    """
    if image is None:
        return None, "请先上传图片"

    result_image = image.copy()
    description = ""

    if mode == "物体检测":
        annotated, detections = detect_objects(image, conf_threshold)
        result_image = annotated
        if detections:
            description = f"检测到 {len(detections)} 个物体：\n"
            for d in detections:
                description += f"  · {d['label']} ({d['confidence']:.1%})\n"
        else:
            description = "未检测到物体，请尝试降低置信度阈值"

    elif mode == "图像分类":
        # TODO: 组员A 完成后取消下一行注释
        # label, confidence = classify_image(image)
        # description = f"识别结果：{label}（置信度：{confidence:.1%}）"
        description = "图像分类模块开发中（组员A 岳思铭）"

    elif mode == "文字识别 (OCR)":
        # TODO: 组员A 完成后取消下一行注释
        # text, annotated_img = extract_text(image)
        # result_image = annotated_img
        # description = f"识别文字：\n{text}"
        description = "OCR 模块开发中（组员A 岳思铭）"

    elif mode == "人脸检测":
        result_image, faces = detect_faces(image)
        if faces:
            description = f"检测到 {len(faces)} 张人脸"
        else:
            description = "未检测到人脸"

    elif mode == "风格迁移":
        result_image = transfer_style(image, style_name)
        description = f"风格：{style_name}"

    return result_image, description


def on_mode_change(mode):
    """切换功能模式时，控制风格选择下拉框的显隐"""
    show = mode == "风格迁移"
    return gr.update(visible=show)


# ========== 构建 Gradio 界面 ==========

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=Noto+Serif+SC:wght@400;600&display=swap');

* {
    font-family: '微软雅黑', 'Microsoft YaHei', 'Noto Sans SC', Arial, sans-serif;
}

.gradio-container {
    max-width: 1280px !important;
    margin: auto;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 16px;
    padding: 20px !important;
}

header {
    text-align: center;
    padding: 30px 20px 20px 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

header h1 {
    font-family: '微软雅黑', 'Microsoft YaHei', 'Noto Sans SC', sans-serif;
    font-size: 2em;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: 2px;
    color: #ffffff;
}

header p {
    font-family: '宋体', 'Noto Serif SC', 'Times New Roman', Times, serif;
    font-size: 1.05em;
    margin: 0;
    opacity: 0.9;
    color: #f0e6ff;
    letter-spacing: 1px;
}

/* 卡片样式 */
.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}

/* 按钮 */
button#submit-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-family: '微软雅黑', 'Microsoft YaHei', sans-serif;
    font-weight: 500;
    font-size: 1.1em;
    padding: 12px 32px;
    border-radius: 8px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

button#submit-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* 页脚 */
footer {
    text-align: center;
    padding: 24px 20px 12px 20px;
    color: #666;
    font-size: 0.85em;
    border-top: 1px solid #e8e8e8;
    margin-top: 20px;
}

footer .copyright {
    font-family: '楷体', 'KaiTi', 'Noto Serif SC', serif;
    font-size: 0.95em;
    color: #555;
    letter-spacing: 1px;
}

footer .sub {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 0.8em;
    color: #999;
    margin-top: 4px;
}

/* 标签美化 */
label {
    font-family: '微软雅黑', 'Microsoft YaHei', sans-serif !important;
    font-weight: 500 !important;
}

/* 结果文本框 */
textarea {
    font-family: '楷体', 'KaiTi', 'Noto Serif SC', serif !important;
    font-size: 0.95em !important;
}
"""

with gr.Blocks(
    title="智能图片分析网站",
) as app:

    # ---------- 页面标题 ----------
    gr.HTML("""
    <header>
        <h1>智能图片分析网站</h1>
        <p>上传图片 · 体验 AI 视觉分析 · 物体检测 · 图像分类 · 文字识别 · 人脸检测 · 风格迁移</p>
    </header>
    """)

    # ---------- 主内容区 ----------
    with gr.Row(equal_height=False):

        # ===== 左侧：输入区 =====
        with gr.Column(scale=1, min_width=400):
            with gr.Group():
                image_input = gr.Image(
                    label="上传图片",
                    type="numpy",
                    height=400,
                    container=True,
                )

            # 示例图片
            gr.Examples(
                examples=[["images/sample/face.jpg", "人脸检测"]],
                inputs=[image_input],
                label="示例图片",
            )

        # ===== 右侧：设置 + 结果 =====
        with gr.Column(scale=1, min_width=400):
            with gr.Group():
                mode = gr.Radio(
                    choices=[
                        "物体检测",
                        "图像分类",
                        "文字识别 (OCR)",
                        "人脸检测",
                        "风格迁移",
                    ],
                    label="功能选择",
                    value="物体检测",
                    container=True,
                )

                conf_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.4,
                    step=0.05,
                    label="置信度阈值",
                    info="仅物体检测有效，值越低检测越多但误检也越多",
                )

                style_choice = gr.Dropdown(
                    choices=["素描·黑白线条画", "二次元·动漫风格"],
                    label="风格选择",
                    value="素描·黑白线条画",
                    visible=False,
                )

                submit_btn = gr.Button(
                    "开始分析",
                    variant="primary",
                    size="lg",
                    elem_id="submit-btn",
                )

    # ---------- 结果显示 ----------
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                image_output = gr.Image(
                    label="分析结果",
                    height=480,
                )

        with gr.Column(scale=1):
            with gr.Group():
                result_text = gr.Textbox(
                    label="结果说明",
                    lines=12,
                    placeholder="分析结果将显示在这里...",
                    container=True,
                )

    # ---------- 页脚 ----------
    gr.HTML("""
    <footer>
        <div class="copyright">北京交通大学 2024级计算机科学与技术 创新应用综合实训-z24281213组 版权所有</div>
        <div class="sub">Group Members: 张建宇 (Leader) · 岳思铭 · 李汇洋</div>
        <div class="sub" style="margin-top:6px;">&copy; 2026 Beijing Jiaotong University. All Rights Reserved.</div>
    </footer>
    """)

    # ---------- 事件绑定 ----------
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, mode, conf_threshold, style_choice],
        outputs=[image_output, result_text],
    )

    # 切换功能时控制风格选择框显隐
    mode.change(
        fn=on_mode_change,
        inputs=mode,
        outputs=style_choice,
    )


# ========== 启动 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("  智能图片分析网站启动中...")
    print("  本地访问: http://localhost:7860")
    print("  按 Ctrl+C 停止服务")
    print("=" * 50)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme="soft",
        css=CSS,
    )
