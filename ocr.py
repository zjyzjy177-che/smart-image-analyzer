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
AUTO_LANGUAGE = "自动检测（已安装语种）"
_easy_readers: dict = {}
_unavailable_local_presets: set = set()
_last_auto_preset = None


def _get_easyocr(language_preset: str, download_enabled: bool = True):
    if language_preset not in LANGUAGE_PRESETS:
        raise ValueError(f"不支持的 OCR 语言预设：{language_preset}")
    key = tuple(LANGUAGE_PRESETS[language_preset])
    if key not in _easy_readers:
        import easyocr

        action = "加载" if download_enabled else "检查"
        print(f"[INFO] 正在{action} OCR 语言模型：{language_preset}")
        _easy_readers[key] = easyocr.Reader(
            list(key),
            verbose=False,
            download_enabled=download_enabled,
        )
    if download_enabled:
        _unavailable_local_presets.discard(language_preset)
    return _easy_readers[key]


def get_ocr(language_preset: str = DEFAULT_LANGUAGE):
    """加载并缓存可用的 OCR 引擎。"""
    global _ocr, _backend
    if language_preset == AUTO_LANGUAGE:
        raise ValueError("自动模式应通过 _auto_recognise() 调用")
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


def _read_easyocr(engine, image_bgr: np.ndarray):
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


def _script_coverage(text: str, preset: str) -> float:
    """统计识别文本与目标文字体系的匹配程度。"""
    chars = [char for char in text if char.isalpha()]
    if not chars:
        return 0.0

    def in_ranges(char, ranges):
        code = ord(char)
        return any(start <= code <= end for start, end in ranges)

    ranges_by_preset = {
        "简体中文 + English": [(0x3400, 0x9FFF)],
        "繁體中文 + English": [(0x3400, 0x9FFF)],
        "日本語 + English": [(0x3040, 0x30FF), (0x3400, 0x9FFF)],
        "한국어 + English": [(0xAC00, 0xD7AF)],
        "Русский + English": [(0x0400, 0x052F)],
        "العربية + English": [(0x0600, 0x06FF)],
        "हिन्दी + English": [(0x0900, 0x097F)],
        "ไทย + English": [(0x0E00, 0x0E7F)],
        "Tiếng Việt + English": [(0x0041, 0x024F), (0x1E00, 0x1EFF)],
        "Latin 欧洲语言": [(0x0041, 0x024F), (0x1E00, 0x1EFF)],
    }
    ranges = ranges_by_preset.get(preset, [])
    matched = sum(in_ranges(char, ranges) for char in chars)
    # English is included in every Reader, so ASCII letters remain weak evidence.
    ascii_letters = sum(char.isascii() and char.isalpha() for char in chars)
    return min(1.0, (matched + ascii_letters * 0.25) / len(chars))


def _is_ascii_text(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and all(char.isascii() for char in letters)


def _auto_recognise(image_bgr: np.ndarray):
    """按可信度逐个尝试本地模型，得到可靠结果后立即停止。"""
    global _last_auto_preset
    candidates = []
    preset_order = list(LANGUAGE_PRESETS)
    if _last_auto_preset in preset_order:
        preset_order.remove(_last_auto_preset)
        preset_order.insert(0, _last_auto_preset)

    for preset in preset_order:
        if preset in _unavailable_local_presets:
            continue
        try:
            reader = _get_easyocr(preset, download_enabled=False)
            results = _read_easyocr(reader, image_bgr)
        except Exception:
            _unavailable_local_presets.add(preset)
            continue
        if not results:
            continue

        text = " ".join(item[1] for item in results if item[1])
        confidences = [float(item[2]) for item in results if item[1]]
        if not confidences:
            continue
        mean_confidence = sum(confidences) / len(confidences)
        coverage = _script_coverage(text, preset)
        # 置信度为主，文字体系匹配度用于避免英文模型把外文误认成乱码。
        score = mean_confidence * (0.55 + 0.45 * coverage)
        candidates.append((score, preset, results))

        # 所有 Reader 都包含英语，高可信纯 ASCII 无需再跑其他语言模型。
        reliable_english = _is_ascii_text(text) and mean_confidence >= 0.82
        # 对应文字体系高度匹配且置信度较高时也可直接采纳。
        reliable_script = coverage >= 0.55 and mean_confidence >= 0.78
        if reliable_english or reliable_script:
            _last_auto_preset = preset
            return results, preset

    if not candidates:
        return [], None
    _, preset, results = max(candidates, key=lambda item: item[0])
    _last_auto_preset = preset
    return results, preset


def _recognise(image_bgr: np.ndarray, language_preset: str):
    if language_preset == AUTO_LANGUAGE:
        results, detected_preset = _auto_recognise(image_bgr)
        return results, detected_preset

    engine = get_ocr(language_preset)
    if _backend == "easyocr":
        return _read_easyocr(engine, image_bgr), language_preset

    if hasattr(engine, "predict"):
        try:
            return _parse_paddle_result(engine.predict(image_bgr)), language_preset
        except (TypeError, AttributeError):
            pass
    try:
        return _parse_paddle_result(engine.ocr(image_bgr, cls=True)), language_preset
    except TypeError:
        return _parse_paddle_result(engine.ocr(image_bgr)), language_preset


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
    results, detected_preset = _recognise(bgr, language_preset)
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

    if language_preset == AUTO_LANGUAGE and detected_preset is None:
        full_text = (
            "自动检测未找到可用的本地语言模型；"
            "请从 OCR 语言下拉框选择语种，程序会自动下载安装。"
        )
    else:
        full_text = "\n".join(texts) if texts else "未识别到达到置信度阈值的文字"
    if return_details:
        return full_text, annotated, {
            "items": details,
            "filtered_count": filtered_count,
            "confidence_threshold": confidence_threshold,
            "language_preset": detected_preset or language_preset,
            "auto_detected": language_preset == AUTO_LANGUAGE,
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
