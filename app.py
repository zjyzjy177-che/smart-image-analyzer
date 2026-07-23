"""
app.py — 智能图片分析系统
组别：z24281213 | 组长：张建宇 | 组员：岳思铭、李汇洋
运行：python app.py
"""

import base64
import gradio as gr
import numpy as np

from detector import detect_objects
# from classifier import classify_image
# from ocr import extract_text
from face_detect import detect_faces
from style_transfer import transfer_style


def load_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

BADGE = load_b64("images/badge.png")
BG = load_b64("images/bg.jpg")
BADGE_IMG = f'<img src="data:image/png;base64,{BADGE}" class="badge"/>' if BADGE else ""
BG_URL = f"data:image/jpeg;base64,{BG}" if BG else ""


def process(image, mode, thresh, style):
    if image is None:
        return None, "请先上传一张图片 / Please upload an image"
    r, d = image.copy(), ""
    if mode == "物体检测":
        a, dets = detect_objects(image, thresh)
        r = a
        if dets:
            d = f"检测到 {len(dets)} 个物体 / {len(dets)} objects:\n"
            for o in dets:
                d += f"  · {o['label']}  ({o['confidence']:.1%})\n"
        else:
            d = "未检测到物体 / No objects detected"
    elif mode == "图像分类":
        d = "模块开发中（组员A 岳思铭）"
    elif mode == "文字识别 (OCR)":
        d = "模块开发中（组员A 岳思铭）"
    elif mode == "人脸检测":
        r, faces = detect_faces(image, thresh)
        if faces:
            d = f"检测到 {len(faces)} 张人脸 / {len(faces)} face(s)\n"
            for i, (x, y, w, h) in enumerate(faces, 1):
                d += f"  · Face {i}: ({x},{y}) {w}×{h}\n"
        else:
            d = "未检测到人脸 / No face detected"
    elif mode == "风格迁移":
        r = transfer_style(image, style)
        d = f"风格 / Style: {style}"
    return r, d

def on_mode_change(mode):
    return gr.update(visible=(mode == "风格迁移"))


CSS = f"""
:root {{
    --blue:#0055B3; --blue-dark:#003d8a; --blue-light:#e3edf9;
    --blue-bg:#f2f6fc; --text:#222; --text2:#555; --text3:#999;
    --card:#fff; --side:#fff; --bd:#d0dae8;
    --sd:rgba(0,85,179,0.07); --bgh:#e8f0fa;
}}
* {{ font-family:'微软雅黑','Microsoft YaHei','PingFang SC','Noto Sans SC',Arial,sans-serif; box-sizing:border-box; }}
body {{ background:var(--blue-bg)!important; }}
.gradio-container {{ max-width:1440px!important; margin:0 auto; background:transparent!important; padding:0!important; }}

/* Top bar */
.topbar {{
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 32px; background:var(--card);
    border-bottom:3px solid var(--blue); box-shadow:0 2px 14px var(--sd);
}}
.topbar-l {{ display:flex; align-items:center; gap:18px; flex:1; }}
.badge {{ width:46px; height:46px; flex-shrink:0; }}
.tleft {{ font-size:1.35em; font-weight:700; color:var(--blue); letter-spacing:3px; }}
.tright {{ font-size:1.35em; font-weight:700; color:var(--blue); letter-spacing:2px; }}
.tdiv {{ color:var(--text3); font-weight:100; font-size:1.3em; }}
.tsub {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:4px; }}
.tsub .en {{ font-family:'Times New Roman',Times,serif; font-size:0.9em; color:var(--text2); letter-spacing:1px; font-style:italic; }}
.tsub .tag {{ font-family:'宋体','SimSun','Noto Serif SC',serif; font-size:0.85em; color:var(--text2); padding-left:14px; border-left:1px solid var(--bd); }}

/* Content */
.body-wrap {{ display:flex!important; flex-direction:row!important; padding:20px 28px!important; position:relative; min-height:600px; }}
.body-wrap::before {{
    content:''; position:fixed; top:0;left:0;right:0;bottom:0;
    background-image:url('{BG_URL}'); background-size:cover; background-position:center;
    opacity:0.20; pointer-events:none; z-index:0;
}}

/* Sidebar */
.sidebar {{ width:260px!important; min-width:260px!important; background:var(--side); border-radius:10px; border:1px solid var(--bd); box-shadow:0 2px 12px var(--sd); padding:14px; margin-right:18px; height:fit-content; position:relative; z-index:1; }}
.sidebar-sec {{ margin-bottom:14px; }}
.sidebar-hd {{ font-size:0.88em; font-weight:700; color:var(--blue); letter-spacing:2px; padding-bottom:6px; border-bottom:2px solid var(--blue-light); margin-bottom:8px; display:flex; align-items:center; gap:5px; }}
.sidebar-hd::before {{ content:''; width:4px; height:13px; background:var(--blue); border-radius:2px; display:inline-block; }}
.feat-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:5px; }}
.feat-item {{
    display:flex; align-items:center; justify-content:center; padding:10px 6px; border-radius:8px;
    border:1.5px solid var(--bd); background:var(--card); cursor:pointer;
    font-size:0.88em; font-weight:500; color:var(--text2); text-align:center; line-height:1.35;
    transition:all .2s;
}}
.feat-item:hover {{ border-color:var(--blue); color:var(--blue); background:var(--blue-light); }}
.feat-item.on {{ border-color:var(--blue); background:var(--blue); color:#fff; }}

/* Main area */
.main-area {{ flex:1!important; min-width:0!important; display:flex!important; flex-direction:column!important; gap:14px!important; position:relative; z-index:1; }}
.ct-card {{ background:var(--card); border-radius:10px; border:1px solid var(--bd); box-shadow:0 2px 12px var(--sd); overflow:hidden; }}
.ct-hd {{
    background:linear-gradient(135deg,var(--blue-light) 0%,transparent 100%);
    padding:9px 16px; border-bottom:1px solid var(--bd);
    font-size:0.9em; font-weight:600; color:var(--blue);
    display:flex; align-items:center; gap:7px;
}}
.ct-hd .dot {{ width:7px; height:7px; border-radius:50%; background:var(--blue); display:inline-block; }}
.ct-bd {{ padding:12px; }}

/* Result grid — fullscreen safe */
.r-grid {{ display:flex!important; flex-direction:row!important; gap:14px!important; width:100%!important; }}
.r-col-img {{ flex:3!important; min-width:280px!important; width:100%!important; min-height:300px!important; }}
.r-col-img img {{ max-height:520px!important; min-height:300px!important; width:100%!important; object-fit:contain!important; background:var(--blue-bg); border-radius:8px; }}
.r-col-txt {{ flex:2!important; min-width:260px!important; width:100%!important; }}

/* Button */
button#sb-btn {{
    background:var(--blue)!important; border:none!important; color:#fff!important;
    font-weight:600!important; font-size:0.95em!important; padding:10px 22px!important;
    border-radius:6px!important; transition:all .25s!important;
    box-shadow:0 4px 12px rgba(0,85,179,0.3)!important; letter-spacing:1px!important; width:100%!important;
}}
button#sb-btn:hover {{ background:var(--blue-dark)!important; transform:translateY(-1px)!important; }}

/* Footer */
.footer {{ text-align:center; padding:16px 28px 12px; border-top:1px solid var(--bd); background:var(--card); }}
.f1 {{ font-family:'楷体','KaiTi','STKaiti','Noto Serif SC',serif; font-size:0.82em; color:var(--text2); letter-spacing:1px; }}
.f2 {{ font-family:Arial,sans-serif; font-size:0.68em; color:var(--text3); margin-top:2px; }}

/* Gradio overrides */
.gr-form,.gr-box,.gr-group {{ border:none!important; box-shadow:none!important; background:transparent!important; }}
label {{ font-weight:500!important; color:var(--text)!important; font-size:0.95em!important; }}
textarea {{ font-family:'楷体','KaiTi','STKaiti','Noto Serif SC',serif!important; font-size:1.2em!important; line-height:1.8!important; background:var(--card)!important; color:#CC3300!important; text-align:center!important; font-weight:600!important; }}
.gr-box textarea, .gr-form textarea, .svelte-1l0r20v textarea {{ color:#CC3300!important; }}
textarea::placeholder {{ text-align:center!important; color:var(--text3)!important; font-weight:400!important; }}
input[type=range] {{ accent-color:var(--blue)!important; }}
footer,.gradio-footer,.built-with,.footer-nav {{ display:none!important; }}
/* 隐藏 Radio — 移到屏幕外不占空间，但保持可点击 */
.gr-box:has(.gr-radio), .gr-form:has(.gr-radio) {{ position:absolute!important; left:-9999px!important; top:0; width:0; height:0; overflow:hidden; opacity:0; pointer-events:none; }}

@media (max-width:900px) {{
    .body-wrap {{ flex-direction:column!important; padding:10px!important; }}
    .sidebar {{ width:100%!important; min-width:100%!important; margin-right:0!important; margin-bottom:12px!important; }}
    .r-grid {{ flex-direction:column!important; }}
}}
"""


# ========== 侧栏点击事件（只有这个需要一点点 JS） ==========
HEAD = """
<script>
document.addEventListener('click', function(e){
  var item = e.target.closest('.feat-item');
  if(!item) return;
  document.querySelectorAll('.feat-item').forEach(function(el){ el.classList.remove('on'); });
  item.classList.add('on');
  var mode = item.dataset.mode;
  document.querySelectorAll('input[type="radio"]').forEach(function(r){
    if(r.value===mode) r.click();
  });
});
</script>
"""


# ========== 构建界面 ==========

with gr.Blocks(title="智能图片分析系统 — 北京交通大学", css=CSS, head=HEAD) as app:

    # ===== 顶栏 =====
    gr.HTML(f"""
    <div class="topbar">
        <div class="topbar-l">
            {BADGE_IMG}
            <div>
                <div style="display:flex;align-items:baseline;gap:20px;flex-wrap:wrap;">
                    <span class="tleft">北京交通大学</span>
                    <span class="tdiv">|</span>
                    <span class="tright">智能图片分析系统</span>
                </div>
                <div class="tsub">
                    <span class="en">Beijing Jiaotong Univ. · Smart Image Analysis</span>
                    <span class="tag">🏫 计算机科学与技术学院 · School of CS &amp; Technology</span>
                </div>
            </div>
        </div>
    </div>
    """)

    # ===== 主体 =====
    with gr.Column(elem_classes="body-wrap"):

        # -- 侧栏 --
        with gr.Column(elem_classes="sidebar"):

            gr.HTML('<div class="sidebar-hd">🎯 功能 / Functions</div>')
            gr.HTML("""
            <div class="feat-grid">
                <div class="feat-item on" data-mode="物体检测">🎯物体检测<br><span style="font-size:0.75em;opacity:0.7">Detection</span></div>
                <div class="feat-item" data-mode="图像分类">🏷️图像分类<br><span style="font-size:0.75em;opacity:0.7">Classify</span></div>
                <div class="feat-item" data-mode="文字识别 (OCR)">📝文字识别<br><span style="font-size:0.75em;opacity:0.7">OCR</span></div>
                <div class="feat-item" data-mode="人脸检测">😃人脸检测<br><span style="font-size:0.75em;opacity:0.7">Face</span></div>
                <div class="feat-item" data-mode="风格迁移">🎨风格迁移<br><span style="font-size:0.75em;opacity:0.7">Style</span></div>
            </div>
            """)

            mode = gr.Radio(
                choices=["物体检测","图像分类","文字识别 (OCR)","人脸检测","风格迁移"],
                value="物体检测", container=False,
            )

            gr.HTML('<div class="sidebar-hd">⚙️ 参数 / Settings</div>')

            conf_threshold = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                label="置信度阈值 / Confidence Threshold",
                info="物体检测/人脸检测通用 / Object & Face Detection",
            )

            style_choice = gr.Dropdown(
                choices=["素描·黑白线条画","二次元·动漫风格"],
                label="风格 / Style", value="素描·黑白线条画", visible=False,
            )
            sb = gr.Button("🚀 开始分析 / Analyze", variant="primary", elem_id="sb-btn")

        # -- 主区 --
        with gr.Column(elem_classes="main-area"):

            # 上传
            with gr.Group(elem_classes="ct-card"):
                gr.HTML('<div class="ct-hd"><span class="dot"></span> 📁 上传图片 / Upload Image</div>')
                with gr.Column(elem_classes="ct-bd"):
                    image_input = gr.Image(label="", type="numpy", height=300, show_label=False, container=False)
                    gr.Examples(
                        examples=[["images/sample/face.jpg","人脸检测"]],
                        inputs=[image_input], label="示例 / Examples", run_on_click=False,
                    )

            # 结果
            with gr.Group(elem_classes="ct-card"):
                gr.HTML('<div class="ct-hd"><span class="dot"></span> 📊 分析结果 / Result</div>')
                with gr.Column(elem_classes="ct-bd"):
                    with gr.Row(elem_classes="r-grid"):
                        with gr.Column(elem_classes="r-col-img"):
                            image_output = gr.Image(
                                label="", show_label=False, height=420, container=False,
                            )
                        with gr.Column(elem_classes="r-col-txt"):
                            result_text = gr.Textbox(
                                label="", lines=16, container=False, show_label=False,
                                placeholder="结果说明 / Description...",
                            )

    # ===== 页脚 =====
    gr.HTML("""
    <div class="footer">
        <div class="f1">北京交通大学 2024级计算机科学与技术 · 创新应用综合实训 · z24281213组</div>
        <div class="f2">张建宇（组长）· 岳思铭 · 李汇洋 | &copy; 2026 BJTU · CS &amp; Tech</div>
    </div>
    """)

    # ===== 事件 =====
    sb.click(fn=process, inputs=[image_input, mode, conf_threshold, style_choice], outputs=[image_output, result_text])
    mode.change(fn=on_mode_change, inputs=mode, outputs=style_choice)


if __name__ == "__main__":
    print("="*50)
    print("  智能图片分析系统 — 北京交通大学")
    print("  本地: http://localhost:7860")
    print("="*50)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, quiet=True)
