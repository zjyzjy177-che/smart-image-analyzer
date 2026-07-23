"""
ocr.py — 文字识别 (OCR) 模块
负责人：组员 A
功能：优先使用 PaddleOCR，并兼容 EasyOCR 作为备选
运行：python ocr.py
"""

import os
import sys
from typing import List, Sequence, Tuple

import cv2
import numpy as np


_ocr = None
_backend = None


def get_ocr():
    """加载并缓存可用的 OCR 引擎。"""
    global _ocr, _backend
    if _ocr is not None:
        return _ocr

    # PaddleOCR 3.x 在 Windows + Python 3.13 下存在 oneDNN 推理兼容问题，
    # 该环境直接使用已验证可用的 EasyOCR，避免每次启动都初始化失败。
    prefer_easyocr = os.name == "nt" and sys.version_info >= (3, 13)

    if not prefer_easyocr:
      try:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x 与 2.x 的初始化参数不同。
        try:
            _ocr = PaddleOCR(
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
        except (TypeError, ValueError):
            _ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

        _backend = "paddle"
        return _ocr
      except Exception as exc:
        print(f"[WARN] PaddleOCR 不可用，将切换到 EasyOCR：{exc}")

    try:
        import easyocr

        _ocr = easyocr.Reader(["ch_sim", "en"])
        _backend = "easyocr"
        return _ocr
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "未找到可用的 OCR 引擎，请安装 paddlepaddle/paddleocr 或 easyocr。"
        ) from exc


def _normalise_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("输入图片为空或格式不正确")
    if image.dtype != np.uint8:
        image = np.nan_to_num(image)
        if image.max(initial=0) <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        rgb = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = image.copy()
    else:
        raise ValueError(f"不支持的图片形状：{image.shape}")
    return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _parse_paddle_result(raw_result) -> List[Tuple[Sequence, str, float]]:
    """统一 PaddleOCR 2.x/3.x 输出为 (多边形, 文字, 置信度)。"""
    items = []

    # 2.x: [[box, (text, score)], ...]，外层通常还有一层图片列表。
    legacy = raw_result
    if legacy and isinstance(legacy, list) and len(legacy) == 1:
        legacy = legacy[0]
    if isinstance(legacy, list):
        for line in legacy:
            if (
                isinstance(line, (list, tuple))
                and len(line) >= 2
                and isinstance(line[1], (list, tuple))
                and len(line[1]) >= 2
            ):
                items.append((line[0], str(line[1][0]), float(line[1][1])))
        if items:
            return items

    # 3.x: OCRResult.json -> {"res": {"rec_texts", "rec_scores", "rec_polys"}}
    results = raw_result if isinstance(raw_result, list) else [raw_result]
    for result in results:
        data = getattr(result, "json", result)
        if callable(data):
            data = data()
        if not isinstance(data, dict):
            continue
        data = data.get("res", data)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        polys = data.get("rec_polys", data.get("dt_polys", []))
        for poly, text, score in zip(polys, texts, scores):
            items.append((poly, str(text), float(score)))
    return items


def _recognise(image_bgr: np.ndarray):
    engine = get_ocr()
    if _backend == "easyocr":
        return [(box, text, score) for box, text, score in engine.readtext(image_bgr)]

    if hasattr(engine, "predict"):
        try:
            return _parse_paddle_result(engine.predict(image_bgr))
        except (TypeError, AttributeError):
            pass
    try:
        return _parse_paddle_result(engine.ocr(image_bgr, cls=True))
    except TypeError:
        return _parse_paddle_result(engine.ocr(image_bgr))


def extract_text(
    image: np.ndarray,
    confidence_threshold: float = 0.4,
    return_details: bool = False,
):
    """
    提取文字并在结果图中框出文字区域。

    返回:
        默认返回 (文字, RGB 标注图)。
        return_details=True 时额外返回每段文字、置信度及过滤数量。
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold 必须在 0 到 1 之间")

    rgb, bgr = _normalise_image(image)
    print("[INFO] 正在进行文字识别...")
    results = _recognise(bgr)
    annotated = rgb.copy()
    texts = []
    details = []
    filtered_count = 0

    for box, text, confidence in results:
        confidence = float(confidence)
        if not text:
            continue
        if confidence < confidence_threshold:
            filtered_count += 1
            continue
        points = np.asarray(box, dtype=np.int32).reshape(-1, 2)
        if len(points) >= 3:
            cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
        texts.append(text)
        details.append({"text": text, "confidence": confidence})

    full_text = "\n".join(texts) if texts else "未识别到达到置信度阈值的文字"
    if return_details:
        return full_text, annotated, {
            "items": details,
            "filtered_count": filtered_count,
            "confidence_threshold": confidence_threshold,
        }
    return full_text, annotated


if __name__ == "__main__":
    print("=" * 50)
    print("OCR 文字识别模块 — 自测模式")
    print("=" * 50)
    test_img = np.full((200, 600, 3), 255, dtype=np.uint8)
    cv2.putText(
        test_img, "Hello World", (50, 115),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
    )
    text, _ = extract_text(test_img)
    print(f"识别文字: {text}")
    print("[OK] OCR 模块测试完成！")
