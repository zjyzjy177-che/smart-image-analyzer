"""
app.py — 智能图片分析网站 主程序
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


def _load_badge():
    try:
        with open("images/badge.png", "rb") as f:
            return f'<img src="data:image/png;base64,{base64.b64encode(f.read()).decode()}" class="badge"/>'
    except:
        return ""

def _load_bg():
    try:
        with open("images/bg-college.jpg", "rb") as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    except:
        return ""

BADGE_HTML = _load_badge()
BG_DATA = _load_bg()


def process_image(image, mode, conf_threshold, style_name):
    if image is None:
        return None, "请先上传一张图片"
    r = image.copy()
    d = ""
    if mode == "物体检测":
        a, dets = detect_objects(image, conf_threshold)
        r = a
        if dets:
            d = f"检测到 {len(dets)} 个物体：\n"
            for o in dets:
                d += f"  · {o['label']}  ({o['confidence']:.1%})\n"
        else:
            d = "未检测到物体，可降低置信度阈值"
    elif mode == "图像分类":
        d = "模块开发中（组员A 岳思铭）"
    elif mode == "文字识别 (OCR)":
        d = "模块开发中（组员A 岳思铭）"
    elif mode == "人脸检测":
        r, faces = detect_faces(image)
        if faces:
            d = f"检测到 {len(faces)} 张人脸"
            for i, (x, y, w, h) in enumerate(faces, 1):
                d += f"\n  · 人脸{i}: ({x},{y}) {w}×{h}"
        else:
            d = "未检测到人脸"
    elif mode == "风格迁移":
        r = transfer_style(image, style_name)
        d = f"风格：{style_name}"
    return r, d

def on_mode_change(mode):
    return gr.update(visible=(mode == "风格迁移"))


# ========== CSS ==========

CSS = f"""
/* ========== 变量 ========== */
:root {{
    --blue: #003690;
    --blue-dark: #002a6a;
    --blue-light: #e5edf8;
    --blue-bg: #f4f7fc;
    --text: #222;
    --text2: #555;
    --text3: #999;
    --card: #ffffff;
    --side: #ffffff;
    --bd: #d5dee8;
    --sd: rgba(0,54,144,0.07);
    --bgh: #eef4fa;
}}
.dark-theme {{
    --blue: #4a8ad4;
    --blue-dark: #5a9ae4;
    --blue-light: #1a2645;
    --blue-bg: #16162a;
    --text: #ddd;
    --text2: #aaa;
    --text3: #777;
    --card: #1e1e3a;
    --side: #1e1e3a;
    --bd: #333368;
    --sd: rgba(0,0,0,0.35);
    --bgh: #252548;
}}

/* ========== 全局 ========== */
* {{ font-family:'微软雅黑','Microsoft YaHei',Arial,sans-serif; box-sizing:border-box; }}
body {{ background:var(--blue-bg) !important; }}
.gradio-container {{ max-width:1440px!important; margin:0 auto; background:transparent!important; padding:0!important; }}

/* ========== 顶栏 ========== */
.topbar {{
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 32px; background:var(--card);
    border-bottom:3px solid var(--blue);
    box-shadow:0 2px 14px var(--sd);
}}
.topbar-l {{ display:flex; align-items:center; gap:18px; flex:1; }}
.badge {{ width:46px; height:46px; flex-shrink:0; }}

/* 左右双标题同级 */
.title-pair {{
    display:flex; align-items:baseline; gap:20px; flex-wrap:wrap;
}}
.title-pair .left {{ font-size:1.35em; font-weight:700; color:var(--blue); letter-spacing:3px; }}
.title-pair .right {{ font-size:1.35em; font-weight:700; color:var(--blue); letter-spacing:2px; }}
.title-pair .divider {{ color:var(--bd); font-weight:100; font-size:1.3em; }}

.sub-title-row {{
    display:flex; gap:16px; flex-wrap:wrap; margin-top:2px;
}}
.sub-title-row .en {{
    font-family:'Times New Roman',Times,serif; font-size:0.7em; color:var(--text3); letter-spacing:1px; font-style:italic;
}}
.sub-title-row .tag {{
    font-family:'宋体','Noto Serif SC',serif; font-size:0.63em; color:var(--text3);
    padding-left:14px; border-left:1px solid var(--bd);
}}
.topbar-r {{ display:flex; align-items:center; gap:8px; }}
.tg {{ display:flex; background:var(--blue-bg); border-radius:16px; padding:2px; border:1px solid var(--bd); }}
.tg button {{
    padding:2px 11px; border:none; background:transparent; border-radius:12px;
    font-size:0.72em; cursor:pointer; color:var(--text2);
    transition:all .2s;
}}
.tg button.on {{ background:var(--blue); color:#fff; }}

/* ========== 主体 ========== */
.body-wrap {{
    display:flex!important; flex-direction:row!important; gap:0!important;
    padding:20px 28px!important; position:relative; min-height:600px;
}}
.body-wrap::before {{
    content:''; position:fixed; top:0; left:0; right:0; bottom:0;
    background-image:url('{BG_DATA}');
    background-size:cover; background-position:center;
    opacity:0.08; pointer-events:none; z-index:0;
}}

/* 侧栏 */
.sidebar {{ width:250px!important; min-width:250px!important; background:var(--side); border-radius:10px; border:1px solid var(--bd); box-shadow:0 2px 12px var(--sd); padding:14px; margin-right:18px; height:fit-content; position:relative; z-index:1; }}
.sidebar-sec {{ margin-bottom:14px; }}
.sidebar-hd {{ font-size:0.76em; font-weight:700; color:var(--blue); letter-spacing:2px; padding-bottom:6px; border-bottom:2px solid var(--blue-light); margin-bottom:8px; display:flex; align-items:center; gap:5px; }}
.sidebar-hd::before {{ content:''; width:4px; height:13px; background:var(--blue); border-radius:2px; display:inline-block; }}
.feat-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:5px; }}
.feat-item {{
    display:flex; align-items:center; justify-content:center; padding:9px 5px; border-radius:6px;
    border:1.5px solid var(--bd); background:var(--card); cursor:pointer;
    font-size:0.8em; font-weight:500; color:var(--text2); text-align:center; line-height:1.2;
    transition:all .2s;
}}
.feat-item:hover {{ border-color:var(--blue); color:var(--blue); background:var(--blue-light); }}
.feat-item.on {{ border-color:var(--blue); background:var(--blue); color:#fff; }}
.param-box {{ background:var(--blue-bg); border-radius:6px; padding:10px 12px; border:1px solid var(--bd); }}

/* 主区 */
.main-area {{ flex:1!important; min-width:0!important; display:flex!important; flex-direction:column!important; gap:14px!important; position:relative; z-index:1; }}
.ct-card {{ background:var(--card); border-radius:10px; border:1px solid var(--bd); box-shadow:0 2px 12px var(--sd); overflow:hidden; }}
.ct-hd {{
    background:linear-gradient(135deg,var(--blue-light) 0%,transparent 100%);
    padding:9px 16px; border-bottom:1px solid var(--bd);
    font-size:0.82em; font-weight:600; color:var(--blue);
    display:flex; align-items:center; gap:7px;
}}
.ct-hd .dot {{ width:7px; height:7px; border-radius:50%; background:var(--blue); display:inline-block; }}
.ct-bd {{ padding:12px; }}

/* 结果区 */
.r-grid {{ display:flex!important; flex-direction:row!important; gap:14px!important; }}
.r-grid .c-img {{ flex:2!important; min-width:0!important; }}
.r-grid .c-txt {{ flex:3!important; min-width:280px!important; }}

/* 按钮 */
button#sb-btn {{
    background:var(--blue)!important; border:none!important; color:#fff!important;
    font-weight:600!important; font-size:0.93em!important; padding:10px 22px!important;
    border-radius:6px!important; transition:all .25s!important;
    box-shadow:0 4px 12px rgba(0,54,144,0.3)!important; letter-spacing:1px!important; width:100%!important;
}}
button#sb-btn:hover {{ background:var(--blue-dark)!important; transform:translateY(-1px)!important; }}

/* 页脚 */
.footer {{ text-align:center; padding:16px 28px 12px; border-top:1px solid var(--bd); background:var(--card); }}
.footer .f1 {{ font-family:'楷体','KaiTi','Noto Serif SC',serif; font-size:0.82em; color:var(--text2); letter-spacing:1px; }}
.footer .f2 {{ font-family:Arial,sans-serif; font-size:0.68em; color:var(--text3); margin-top:2px; }}

/* Gradio 覆盖 */
.gr-form,.gr-box,.gr-group {{ border:none!important; box-shadow:none!important; background:transparent!important; }}
label {{ font-weight:500!important; color:var(--text)!important; font-size:0.82em!important; }}
textarea {{ font-family:'楷体','KaiTi','Noto Serif SC',serif!important; font-size:0.88em!important; background:var(--card)!important; color:var(--text)!important; }}
input[type=range] {{ accent-color:var(--blue)!important; }}

/* 语言切换 — 用类名控制显示 */
.zh {{ display:inline; }}
.en {{ display:none; }}
body.show-en .zh {{ display:none; }}
body.show-en .en {{ display:inline; }}

@media (max-width:900px) {{
    .body-wrap {{ flex-direction:column!important; padding:10px!important; }}
    .sidebar {{ width:100%!important; min-width:100%!important; margin-right:0!important; margin-bottom:12px!important; }}
    .r-grid {{ flex-direction:column!important; }}
}}
"""

# ========== 头部脚本 ==========

HEAD = """
<script>
window.onload = function(){
  // 主题恢复
  var t = localStorage.getItem('t') || 'light';
  if(t==='dark') document.body.classList.add('dark-theme');
  document.querySelectorAll('#tg-t button').forEach(function(b){
    if(b.dataset.t === t) b.classList.add('on');
  });

  // 语言恢复
  var l = localStorage.getItem('l') || 'zh';
  if(l==='en') document.body.classList.add('show-en');
  document.querySelectorAll('#tg-l button').forEach(function(b){
    if(b.dataset.l === l) b.classList.add('on');
  });
};

function setTheme(m){
  document.body.classList.toggle('dark-theme', m==='dark');
  localStorage.setItem('t', m);
  document.querySelectorAll('#tg-t button').forEach(function(b){
    b.classList.toggle('on', b.dataset.t===m);
  });
}

function setLang(l){
  document.body.classList.toggle('show-en', l==='en');
  localStorage.setItem('l', l);
  document.querySelectorAll('#tg-l button').forEach(function(b){
    b.classList.toggle('on', b.dataset.l===l);
  });
}

// 跟随系统
(function(){
  var t = localStorage.getItem('t');
  if(!t || t==='system'){
    if(window.matchMedia('(prefers-color-scheme:dark)').matches){
      document.body.classList.add('dark-theme');
    }
  }
})();

// 侧栏功能点击
document.addEventListener('click', function(e){
  var item = e.target.closest('.feat-item');
  if(!item) return;
  document.querySelectorAll('.feat-item').forEach(function(el){ el.classList.remove('on'); });
  item.classList.add('on');
  var mode = item.dataset.mode;
  var radios = document.querySelectorAll('input[type="radio"]');
  for(var i=0; i<radios.length; i++){
    if(radios[i].value === mode){ radios[i].click(); break; }
  }
});
</script>
"""


# ========== 构建 ==========

with gr.Blocks(title="智能图片分析系统 — 北京交通大学") as app:

    # ===== 顶栏 =====
    gr.HTML(f"""
    <div class="topbar">
        <div class="topbar-l">
            {BADGE_HTML}
            <div>
                <div class="title-pair">
                    <span class="left"><span class="zh">北京交通大学</span><span class="en">Beijing Jiaotong Univ.</span></span>
                    <span class="divider">|</span>
                    <span class="right"><span class="zh">智能图片分析系统</span><span class="en">Smart Image Analysis</span></span>
                </div>
                <div class="sub-title-row">
                    <span class="en"><span class="zh">Beijing Jiaotong University · Smart Image Analysis Platform</span><span class="en">北京交通大学 · 智能图片分析系统</span></span>
                    <span class="tag"><span class="zh">计算机科学与技术学院</span><span class="en">School of CS &amp; Technology</span></span>
                </div>
            </div>
        </div>
        <div class="topbar-r">
            <div class="tg" id="tg-l">
                <button data-l="zh" onclick="setLang('zh')">简</button>
                <button data-l="en" onclick="setLang('en')">EN</button>
            </div>
            <div class="tg" id="tg-t">
                <button data-t="light" onclick="setTheme('light')">☀</button>
                <button data-t="system" onclick="localStorage.setItem('t','system');setTheme(window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light')">🖥</button>
                <button data-t="dark" onclick="setTheme('dark')">🌙</button>
            </div>
        </div>
    </div>
    """)

    # ===== 主体 =====
    with gr.Column(elem_classes="body-wrap"):

        # -- 侧栏 --
        with gr.Column(elem_classes="sidebar"):
            gr.HTML("""
            <div class="sidebar-sec">
                <div class="sidebar-hd"><span class="zh">分析功能</span><span class="en">Functions</span></div>
                <div class="feat-grid">
                    <div class="feat-item on" data-mode="物体检测"><span class="zh">物体检测</span><span class="en">Detection</span></div>
                    <div class="feat-item" data-mode="图像分类"><span class="zh">图像分类</span><span class="en">Classify</span></div>
                    <div class="feat-item" data-mode="文字识别 (OCR)"><span class="zh">文字识别</span><span class="en">OCR</span></div>
                    <div class="feat-item" data-mode="人脸检测"><span class="zh">人脸检测</span><span class="en">Face</span></div>
                    <div class="feat-item" data-mode="风格迁移"><span class="zh">风格迁移</span><span class="en">Style</span></div>
                </div>
            </div>
            """)

            mode = gr.Radio(
                choices=["物体检测","图像分类","文字识别 (OCR)","人脸检测","风格迁移"],
                value="物体检测", visible=False, container=False,
            )

            gr.HTML("""
            <div class="sidebar-sec">
                <div class="sidebar-hd"><span class="zh">参数设置</span><span class="en">Settings</span></div>
                <div class="param-box">
            """)

            conf_threshold = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                label="置信度阈值",
                info="低值检测更多，高值更精准（仅物体检测）",
            )

            style_choice = gr.Dropdown(
                choices=["素描·黑白线条画","二次元·动漫风格"],
                label="风格选择", value="素描·黑白线条画", visible=False,
            )

            gr.HTML("""
                </div>
            </div>
            """)

            sb = gr.Button("开始分析", variant="primary", elem_id="sb-btn")

        # -- 主区 --
        with gr.Column(elem_classes="main-area"):
            # 上传区
            with gr.Group(elem_classes="ct-card"):
                gr.HTML('<div class="ct-hd"><span class="dot"></span> <span class="zh">上传图片</span><span class="en">Upload Image</span></div>')
                with gr.Column(elem_classes="ct-bd"):
                    image_input = gr.Image(label="", type="numpy", height=300, show_label=False, container=False)
                    gr.Examples(
                        examples=[["images/sample/face.jpg","人脸检测"]],
                        inputs=[image_input], label="", run_on_click=False,
                    )

            # 结果区
            with gr.Group(elem_classes="ct-card"):
                gr.HTML('<div class="ct-hd"><span class="dot"></span> <span class="zh">分析结果</span><span class="en">Result</span></div>')
                with gr.Column(elem_classes="ct-bd"):
                    with gr.Row(elem_classes="r-grid"):
                        with gr.Column(elem_classes="c-img"):
                            image_output = gr.Image(label="", show_label=False, height=350, container=False)
                        with gr.Column(elem_classes="c-txt"):
                            result_text = gr.Textbox(
                                label="", lines=14, container=False, show_label=False,
                                placeholder="点击「开始分析」查看结果...",
                            )

    # ===== 页脚 =====
    gr.HTML("""
    <div class="footer">
        <div class="f1"><span class="zh">北京交通大学 2024级计算机科学与技术 · 创新应用综合实训 · z24281213组 版权所有</span><span class="en">BJTU · CS&amp;Tech · Innovation Training · Group z24281213</span></div>
        <div class="f2">张建宇（组长）· 岳思铭 · 李汇洋</div>
        <div class="f2">&copy; 2026 Beijing Jiaotong University. All Rights Reserved.</div>
    </div>
    """)

    # ===== 事件 =====
    sb.click(fn=process_image, inputs=[image_input, mode, conf_threshold, style_choice], outputs=[image_output, result_text])
    mode.change(fn=on_mode_change, inputs=mode, outputs=style_choice)


if __name__ == "__main__":
    print("="*50)
    print("  智能图片分析系统 — 北京交通大学")
    print("  本地: http://localhost:7860")
    print("="*50)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS, head=HEAD, inbrowser=True)
