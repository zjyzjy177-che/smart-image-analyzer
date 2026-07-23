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
# EasyOCR 的不同文字体系不能随意混装在同一个 Reader 中。
# 使用明确的语言预设，按需加载并缓存，避免启动时下载所有模型。
LANGUAGE_PRESETS = {
    "简体中文 + English": ["ch_sim", "en"],
    "繁體中文 + English": ["ch_tra", "en"],
    "日本語 + English": ["ja", "en"],
    "한국어 + English": ["ko", "en"],
    "Latin 欧洲语言": ["en", "fr", "de", "es", "pt", "it", "nl", "pl"],
    "Русский + English": ["ru", "en"],
    "العربية + English": ["ar", "en"],
    "हिन्दी + English": ["hi", "en"],
    "ไทย + English": ["th", "en"],
    "Tiếng Việt + English": ["vi", "en"],
}
DEFAULT_LANGUAGE = "简体中文 + English"
_easy_readers: dict = {}


def _get_easyocr(language_preset: str):
    if language_preset not in LANGUAGE_PRESETS:
        raise ValueError(f"不支持的 OCR 语言预设：{language_preset}")
    key = tuple(LANGUAGE_PRESETS[language_preset])
    if key not in _easy_readers:
        import easyocr

        print(f"[INFO] 正在加载 OCR 语言模型：{language_preset}")
        _easy_readers[key] = easyocr.Reader(list(key), verbose=False)
    return _easy_readers[key]


def get_ocr(language_preset: str = DEFAULT_LANGUAGE):
    """加载并缓存可用的 OCR 引擎。"""
    global _ocr, _backend
    if language_preset != DEFAULT_LANGUAGE:
        _backend = "easyocr"
        return _get_easyocr(language_preset)
    if _ocr is not None and _backend == "paddle":
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
        _ocr = _get_easyocr(language_preset)
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


def _recognise(image_bgr: np.ndarray, language_preset: str):
    engine = get_ocr(language_preset)
    if _backend == "easyocr":
        # 降低文字检测阶段门槛并放大图片，改善小字、浅色字和低对比文字漏检。
        return [
            (box, text, score)
            for box, text, score in engine.readtext(
                image_bgr,
                detail=1,
                paragraph=False,
                decoder="greedy",
                text_threshold=0.45,
                low_text=0.20,
                link_threshold=0.30,
                canvas_size=2560,
                mag_ratio=1.5,
                contrast_ths=0.05,
                adjust_contrast=0.7,
            )
        ]

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
    language_preset: str = DEFAULT_LANGUAGE,
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
    results = _recognise(bgr, language_preset)
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
            "language_preset": language_preset,
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
