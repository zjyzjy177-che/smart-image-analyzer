"""
app.py — 智能图片分析网站 主程序
负责人：组长（你）
功能：Gradio Web 界面，集成所有模块
运行：python app.py
"""

import gradio as gr
import numpy as np
from PIL import Image

# 导入各模块（TODO: A和B完成他们的模块后取消注释）
from detector import detect_objects
# from classifier import classify_image     # A 负责
# from ocr import extract_text              # A 负责
# from face_detect import detect_faces      # B 负责
# from style_transfer import transfer_style # B 负责（加分项）


def process_image(image, mode, conf_threshold):
    """
    核心处理函数：根据选择的功能模式调用不同的模块

    参数:
        image: 输入图片 (numpy array)
        mode: 功能模式
        conf_threshold: 置信度阈值

    返回:
        处理后的图片 + 结果描述文字
    """
    if image is None:
        return None, "⚠️ 请先上传图片"

    result_image = image.copy()
    description = ""

    if mode == "🎯 物体检测":
        # === 你（组长）做的功能 ===
        annotated, detections = detect_objects(image, conf_threshold)
        result_image = annotated

        if detections:
            description = f"✅ 检测到 {len(detections)} 个物体：\n"
            for d in detections:
                description += f"  · {d['label']} ({d['confidence']:.1%})\n"
        else:
            description = "⚠️ 未检测到物体，请尝试降低置信度阈值"

    elif mode == "🏷️ 图像分类":
        # === A 做的功能 ===
        # TODO: A 实现后取消注释
        # label, confidence = classify_image(image)
        # description = f"🏷️ 识别结果：{label}（置信度：{confidence:.1%}）"
        description = "⏳ 图像分类模块开发中（组员A）"

    elif mode == "📝 文字识别 (OCR)":
        # === A 做的功能 ===
        # TODO: A 实现后取消注释
        # text = extract_text(image)
        # description = f"📝 识别到的文字：\n{text}"
        description = "⏳ OCR 模块开发中（组员A）"

    elif mode == "😃 人脸检测":
        # === B 做的功能 ===
        # TODO: B 实现后取消注释
        # result_image, faces = detect_faces(image)
        # if faces:
        #     description = f"✅ 检测到 {len(faces)} 张人脸"
        # else:
        #     description = "未检测到人脸"
        description = "⏳ 人脸检测模块开发中（组员B）"

    elif mode == "🎨 风格迁移":
        # === B 做的功能（加分项）===
        # TODO: B 实现后取消注释
        # result_image = transfer_style(image)
        # description = "✅ 风格迁移完成！"
        description = "⏳ 风格迁移模块开发中（组员B）"

    return result_image, description


# ========== 构建 Gradio 界面 ==========

# 定义界面主题色
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
)

with gr.Blocks(
    theme=theme,
    title="智能图片分析网站",
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto;
    }
    header {
        text-align: center;
        padding: 20px 0;
    }
    header h1 {
        font-size: 2em;
        margin-bottom: 5px;
    }
    header p {
        color: #666;
        font-size: 1.1em;
    }
    """
) as app:

    # ---------- 页面标题 ----------
    gr.HTML("""
    <header>
        <h1>🖼️ 智能图片分析网站</h1>
        <p>上传一张图片，体验 AI 物体检测 · 图像分类 · 文字识别 · 人脸检测 · 风格迁移</p>
    </header>
    """)

    with gr.Row():
        # ---------- 左侧：输入 ----------
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📁 上传图片",
                type="numpy",
                height=400,
            )

            # 示例图片
            gr.Examples(
                examples=[["images/sample/cat.jpg", "🎯 物体检测"],
                          ["images/sample/car.jpg", "🎯 物体检测"],
                          ["images/sample/text.jpg", "📝 文字识别 (OCR)"]],
                inputs=[image_input],
                label="📸 试试示例图片（TODO: 放几张示例图到 images/sample/ 目录）"
            )

        # ---------- 右侧：设置 + 结果 ----------
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=[
                    "🎯 物体检测",
                    "🏷️ 图像分类",
                    "📝 文字识别 (OCR)",
                    "😃 人脸检测",
                    "🎨 风格迁移",
                ],
                label="🔧 选择功能",
                value="🎯 物体检测",
            )

            conf_threshold = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.4,
                step=0.05,
                label="🎚️ 置信度阈值（仅物体检测有效）",
            )

            submit_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

    # ---------- 结果显示 ----------
    with gr.Row():
        with gr.Column():
            image_output = gr.Image(
                label="📊 分析结果",
                height=450,
            )

        with gr.Column():
            result_text = gr.Textbox(
                label="📋 结果说明",
                lines=10,
                placeholder="分析结果将显示在这里...",
            )

    # ---------- 页脚 ----------
    gr.HTML("""
    <footer style="text-align:center;padding:20px;color:#999;font-size:0.9em;">
        <p>📚 创新应用综合实训 · 小组项目 · 基于 YOLOv8 + PyTorch + Gradio</p>
        <p>组长：xxx · 组员A：xxx · 组员B：xxx</p>
    </footer>
    """)

    # ---------- 绑定事件 ----------
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, mode, conf_threshold],
        outputs=[image_output, result_text],
    )


# ========== 启动 ==========
if __name__ == "__main__":
    print("🚀 智能图片分析网站启动中...")
    print("  本地访问: http://localhost:7860")
    print("  按 Ctrl+C 停止服务")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
