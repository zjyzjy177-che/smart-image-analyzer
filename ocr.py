"""
ocr.py — 文字识别 (OCR) 模块
负责人：组员 A
功能：使用 PaddleOCR 从图片中提取文字
运行：python ocr.py  # 自测
"""

import numpy as np
import cv2


def get_ocr():
    """
    加载 PaddleOCR 模型
    TODO: 首次加载需要下载模型权重（~50MB），请确保网络畅通
    如果安装 paddleocr 报错，可以：
        pip install paddlepaddle paddleocr
    或者使用备选的 easyocr： pip install easyocr （然后把下面代码改成 easyocr）
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError(
            "请先安装 PaddleOCR: pip install paddlepaddle paddleocr\n"
            "或者使用备选方案: pip install easyocr"
        )

    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    return ocr


# 全局缓存
_ocr = None


def extract_text(image: np.ndarray):
    """
    从图片中提取文字

    参数:
        image: numpy 数组 (H, W, 3)

    返回:
        text: str 提取到的文字内容
        result_img: 在原图上标注文字区域的图片
    """
    global _ocr
    if _ocr is None:
        print("🔄 正在加载 OCR 模型（首次加载需要下载~50MB权重）...")
        _ocr = get_ocr()
        print("✅ OCR 模型加载完成！")

    # PaddleOCR 输入要求 BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 如果输入是 RGB（Gradio 默认），转为 BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image

    # 文字识别
    result = _ocr.ocr(img_bgr, cls=True)

    # 解析结果
    texts = []
    result_img = image.copy()

    if result and result[0]:
        for line in result[0]:
            bbox = line[0]  # 四个角坐标
            text = line[1][0]  # 识别的文字
            confidence = line[1][1]  # 置信度

            texts.append(f"{text}")

            # 绘制文字区域框（把四个角坐标转为矩形）
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(result_img, [pts], isClosed=True,
                          color=(0, 255, 0), thickness=2)

            # 在框上方写上识别的文字
            x = int(min(bbox[0][0], bbox[3][0]))
            y = int(min(bbox[0][1], bbox[1][1])) - 10
            cv2.putText(result_img, text, (x, max(y, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    full_text = "\n".join(texts) if texts else "未识别到文字"
    return full_text, result_img


if __name__ == "__main__":
    """自测：python ocr.py"""
    print("=" * 50)
    print("OCR 文字识别模块 — 自测模式")
    print("=" * 50)

    # 创建一个含文字的测试图
    test_img = np.zeros((200, 600, 3), dtype=np.uint8)
    test_img[:] = (255, 255, 255)
    cv2.putText(test_img, "Hello World 你好世界", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    text, annotated = extract_text(test_img)
    print(f"📝 识别文字: {text}")
    print("✅ OCR 模块测试完成！")
