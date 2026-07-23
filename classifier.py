"""
图像分类模块

使用 torchvision 官方 ImageNet ResNet50 模型，提供：
1. Top-K 分类结果，而非只返回一个武断答案
2. 中文大类与英文细分类
3. 可信度等级及低置信度提示
4. RGB、RGBA、灰度、浮点图片输入兼容
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


_model = None
_preprocess = None
_labels = None
_caption_model = None
_caption_processor = None
_caption_error = None
_translator_model = None
_translator_tokenizer = None
_translator_error = None
_PROJECT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _PROJECT_DIR / "models"
_RESNET_PATH = _MODEL_DIR / "resnet50" / "resnet50-11ad3fa6.pth"
_BLIP_PATH = _MODEL_DIR / "blip-image-captioning-base"
_TRANSLATOR_PATH = _MODEL_DIR / "opus-mt-en-zh"


# ImageNet 中常见类别的中文细分类。未列出的类别仍会显示官方英文名称，
# 同时由 _coarse_category() 给出中文大类，避免错误或生硬的机器翻译。
_ZH_LABELS = {
    "airliner": "客机",
    "ambulance": "救护车",
    "backpack": "双肩包",
    "banana": "香蕉",
    "bell cote": "钟楼",
    "bicycle-built-for-two": "双人自行车",
    "book jacket": "书籍封面",
    "bottlecap": "瓶盖",
    "cab": "出租车",
    "camera": "相机",
    "cellular telephone": "手机",
    "coffee mug": "咖啡杯",
    "computer keyboard": "电脑键盘",
    "convertible": "敞篷车",
    "desktop computer": "台式电脑",
    "dining table": "餐桌",
    "electric fan": "电风扇",
    "espresso": "浓缩咖啡",
    "guitar": "吉他",
    "hand-held computer": "手持电脑",
    "head cabbage": "卷心菜",
    "jean": "牛仔裤",
    "jersey": "运动衫",
    "laptop": "笔记本电脑",
    "lemon": "柠檬",
    "library": "图书馆",
    "minibus": "小型公交车",
    "minivan": "小型客货车",
    "monitor": "显示器",
    "mouse": "鼠标",
    "orange": "橙子",
    "parking meter": "停车计时器",
    "pickup": "皮卡车",
    "pineapple": "菠萝",
    "plate": "盘子",
    "printer": "打印机",
    "refrigerator": "冰箱",
    "remote control": "遥控器",
    "school bus": "校车",
    "screen": "屏幕",
    "sports car": "跑车",
    "strawberry": "草莓",
    "sunglass": "太阳镜",
    "television": "电视",
    "toilet seat": "马桶座圈",
    "traffic light": "交通信号灯",
    "triumphal arch": "凯旋门",
    "vault": "拱顶建筑",
    "water bottle": "水瓶",
    "warplane": "军用飞机",
}


def get_model():
    """延迟加载并缓存模型、官方预处理和标签。"""
    global _model, _preprocess, _labels
    if _model is None:
        print("[INFO] 正在加载 ResNet50 图像分类模型...")
        weights = ResNet50_Weights.DEFAULT
        if _RESNET_PATH.is_file():
            _model = resnet50(weights=None)
            _model.load_state_dict(
                torch.load(_RESNET_PATH, map_location="cpu", weights_only=True)
            )
        else:
            _model = resnet50(weights=weights)
        _model.eval()
        _preprocess = weights.transforms()
        _labels = weights.meta["categories"]
        print("[OK] 图像分类模型加载完成")
    return _model, _preprocess, _labels


def get_caption_model():
    """延迟加载 BLIP 整图描述模型；加载失败时允许 ResNet 继续工作。"""
    global _caption_model, _caption_processor, _caption_error
    if _caption_model is not None:
        return _caption_model, _caption_processor
    if _caption_error is not None:
        return None, None

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        from huggingface_hub import snapshot_download

        model_name = "Salesforce/blip-image-captioning-base"
        print("[INFO] 正在加载 BLIP 整图描述模型...")
        if _BLIP_PATH.is_dir():
            model_source = str(_BLIP_PATH)
            local_only = True
        else:
          try:
            model_source = snapshot_download(
                model_name,
                local_files_only=True,
            )
            local_only = True
          except Exception:
            model_source = model_name
            local_only = False
        _caption_processor = BlipProcessor.from_pretrained(
            model_source,
            use_fast=False,
            local_files_only=local_only,
        )
        # 当前缓存为 pytorch_model.bin。明确关闭 safetensors 探测，
        # 避免网络较慢时反复查询不存在的另一种权重格式。
        _caption_model = BlipForConditionalGeneration.from_pretrained(
            model_source,
            use_safetensors=False,
            local_files_only=local_only,
        )
        _caption_model.eval()
        print("[OK] BLIP 整图描述模型加载完成")
        return _caption_model, _caption_processor
    except Exception as exc:
        _caption_error = str(exc)
        print(f"[WARN] 整图描述模型不可用，将保留基础分类：{exc}")
        return None, None


def _to_rgb_pil(image: np.ndarray) -> Image.Image:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("输入图片为空或格式不正确")

    if image.dtype != np.uint8:
        image = np.nan_to_num(image)
        if float(np.max(image)) <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.ndim != 3:
        raise ValueError(f"不支持的图片形状：{image.shape}")
    if image.shape[2] == 1:
        return Image.fromarray(image[:, :, 0]).convert("RGB")
    if image.shape[2] == 4:
        return Image.fromarray(image, mode="RGBA").convert("RGB")
    if image.shape[2] == 3:
        return Image.fromarray(image, mode="RGB")
    raise ValueError(f"不支持的图片通道数：{image.shape[2]}")


def _coarse_category(index: int, label: str) -> str:
    """依据 ImageNet 类别索引和名称返回稳定、易懂的中文大类。"""
    if 0 <= index <= 398:
        if 151 <= index <= 268:
            return "狗"
        if 281 <= index <= 285:
            return "猫"
        if 7 <= index <= 24:
            return "鸟类"
        if 30 <= index <= 32 or 33 <= index <= 37:
            return "两栖/爬行动物"
        if 107 <= index <= 149:
            return "昆虫/无脊椎动物"
        return "动物"
    if 924 <= index <= 969:
        return "食品/食材"
    if index in {404, 405, 895}:
        return "飞机"
    if index in {436, 468, 511, 609, 627, 656, 661, 717, 734, 751, 817, 864}:
        return "汽车/车辆"
    if index in {407, 444, 671, 779}:
        return "自行车/摩托车"

    text = label.lower()
    keyword_groups = (
        ("电子设备", ("computer", "keyboard", "monitor", "screen", "telephone", "camera", "printer")),
        ("服饰用品", ("jersey", "shirt", "jean", "shoe", "sandal", "coat", "hat", "suit")),
        ("家具/家居", ("table", "chair", "sofa", "bed", "lamp", "refrigerator")),
        ("建筑/场所", ("building", "church", "palace", "library", "vault", "arch", "cote")),
        ("乐器", ("guitar", "piano", "violin", "drum", "organ")),
        ("交通工具", ("car", "bus", "truck", "train", "boat", "ship", "plane")),
        ("容器/餐具", ("bottle", "cup", "mug", "plate", "bowl")),
    )
    for category, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return category
    return "日常物品/场景"


def _confidence_level(confidence: float) -> Tuple[str, bool]:
    if confidence >= 0.55:
        return "较高", False
    if confidence >= 0.25:
        return "一般", False
    return "较低", True


def _scene_category(caption: str) -> str:
    """根据描述文本归纳中文场景大类，保持和物体检测的功能区分。"""
    text = caption.lower()
    groups = (
        ("太空/天文场景", ("planet", "earth", "space", "galaxy", "moon", "star")),
        ("人物活动", ("person", "man", "woman", "boy", "girl", "people", "child")),
        ("动物场景", ("dog", "cat", "bird", "horse", "animal", "bear", "elephant")),
        ("自然风景", ("mountain", "river", "lake", "forest", "beach", "ocean", "sunset")),
        ("城市/建筑", ("city", "street", "building", "church", "bridge", "room", "house")),
        ("交通场景", ("car", "bus", "train", "airplane", "truck", "motorcycle", "boat")),
        ("食物/餐饮", ("food", "meal", "plate", "pizza", "cake", "fruit", "table")),
        ("运动场景", ("playing", "sport", "racing", "team", "ball", "tennis", "soccer", "ski", "surf")),
    )
    for category, keywords in groups:
        if any(keyword in text for keyword in keywords):
            return category
    return "日常生活/其他场景"


def _caption_summary_zh(caption: str) -> str:
    """为常见演示场景生成简洁中文摘要，同时保留英文原始描述。"""
    text = caption.lower()
    subject = None
    if any(word in text for word in ("man", "boy")):
        subject = "一名男性"
    elif any(word in text for word in ("woman", "girl")):
        subject = "一名女性"
    elif any(word in text for word in ("person", "people")):
        subject = "人物"

    if subject:
        if "red bull racing suit" in text:
            return f"{subject}身穿红牛赛车服"
        if "racing suit" in text:
            return f"{subject}身穿赛车服"
        if any(word in text for word in ("smiling", "smile", "laughing")):
            return f"{subject}正在微笑"
        if "standing" in text:
            return f"{subject}站在画面中"
        if "sitting" in text:
            return f"{subject}坐在画面中"
        if "holding" in text:
            return f"{subject}手持物品"
        return f"以{subject}为主体的人物场景"
    if any(word in text for word in ("planet", "earth")):
        return "以星球或地球为主体的太空场景"
    if any(word in text for word in ("racing", "race car", "racing team")):
        return "赛车运动相关场景"
    if "dog" in text:
        return "以狗为主体的动物场景"
    if "cat" in text:
        return "以猫为主体的动物场景"
    if any(word in text for word in ("mountain", "river", "lake", "forest", "beach")):
        return "自然风景场景"
    return f"{_scene_category(caption)}"


def translate_caption_to_chinese(caption: str) -> str:
    """使用离线 Marian 模型翻译 BLIP 英文描述，失败时才使用规则摘要。"""
    global _translator_model, _translator_tokenizer, _translator_error
    if not caption:
        return ""

    try:
        if _translator_model is None:
            if _translator_error is not None:
                return _caption_summary_zh(caption)

            from huggingface_hub import snapshot_download
            from transformers import MarianMTModel, MarianTokenizer

            model_name = "Helsinki-NLP/opus-mt-en-zh"
            print("[INFO] 正在加载英译中模型...")
            if _TRANSLATOR_PATH.is_dir():
                model_source = str(_TRANSLATOR_PATH)
                local_only = True
            else:
              try:
                model_source = snapshot_download(
                    model_name,
                    local_files_only=True,
                )
                local_only = True
              except Exception:
                model_source = model_name
                local_only = False

            _translator_tokenizer = MarianTokenizer.from_pretrained(
                model_source,
                local_files_only=local_only,
            )
            _translator_model = MarianMTModel.from_pretrained(
                model_source,
                local_files_only=local_only,
            )
            _translator_model.eval()
            print("[OK] 英译中模型加载完成")

        inputs = _translator_tokenizer(
            [caption],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.inference_mode():
            output_ids = _translator_model.generate(
                **inputs,
                max_length=64,
                num_beams=3,
            )
        translation = _translator_tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()
        # 修正常见品牌和服饰在通用翻译模型中的直译。
        translation = (
            translation
            .replace("红色公牛", "红牛")
            .replace("赛车西装", "赛车服")
        )
        return translation or _caption_summary_zh(caption)
    except Exception as exc:
        _translator_error = str(exc)
        print(f"[WARN] 英译中模型不可用，将使用简要中文描述：{exc}")
        return _caption_summary_zh(caption)


def describe_image(
    image: np.ndarray,
    confidence_threshold: float = 0.4,
) -> Dict:
    """生成整张图片的语义描述；模型不可用时返回可解释的降级状态。"""
    model, processor = get_caption_model()
    if model is None:
        return {
            "available": False,
            "caption_en": None,
            "caption_zh": "整图描述模型暂不可用",
            "scene_category": None,
            "description_confidence": 0.0,
            "accepted": False,
            "error": _caption_error,
        }

    pil_image = _to_rgb_pil(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3,
            no_repeat_ngram_size=2,
            return_dict_in_generate=True,
            output_scores=True,
        )
    output_ids = generation.sequences
    transition_scores = model.compute_transition_scores(
        output_ids,
        generation.scores,
        generation.beam_indices,
        normalize_logits=True,
    )
    valid_scores = transition_scores[transition_scores < 0]
    if valid_scores.numel():
        description_confidence = float(torch.exp(valid_scores.mean()).item())
    else:
        description_confidence = 0.0

    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return {
        "available": True,
        "caption_en": caption,
        "caption_zh": translate_caption_to_chinese(caption),
        "scene_category": _scene_category(caption),
        "description_confidence": description_confidence,
        "accepted": description_confidence >= confidence_threshold,
        "error": None,
    }


def _detect_content_hint(image: np.ndarray):
    """补足 ImageNet 没有“人物”通用类别的缺陷。检测失败不影响主分类。"""
    try:
        from face_detect import detect_faces

        _, faces = detect_faces(image, conf_threshold=0.7)
        if faces:
            return {
                "category_zh": "人物/人像",
                "description": f"检测到 {len(faces)} 张人脸",
                "face_count": len(faces),
            }
    except Exception as exc:
        print(f"[WARN] 人像内容判断已跳过：{exc}")
    return None


def classify_topk(
    image: np.ndarray,
    top_k: int = 5,
    confidence_threshold: float = 0.4,
) -> Dict:
    """
    返回适合界面展示的完整分类结果。

    字段:
        predictions: 按置信度降序的候选列表
        confidence_level: 较高 / 一般 / 较低
        uncertain: 是否应提示用户结果不确定
    """
    if not 1 <= top_k <= 10:
        raise ValueError("top_k 必须在 1 到 10 之间")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold 必须在 0 到 1 之间")

    model, preprocess, labels = get_model()
    input_tensor = preprocess(_to_rgb_pil(image)).unsqueeze(0)
    with torch.inference_mode():
        probabilities = F.softmax(model(input_tensor)[0], dim=0)
        values, indices = torch.topk(probabilities, top_k)

    predictions: List[Dict] = []
    for rank, (value, index_tensor) in enumerate(zip(values, indices), 1):
        index = int(index_tensor.item())
        english = labels[index]
        predictions.append({
            "rank": rank,
            "index": index,
            "category_zh": _coarse_category(index, english),
            "label_zh": _ZH_LABELS.get(english),
            "label_en": english,
            "confidence": float(value.item()),
        })

    level, _ = _confidence_level(predictions[0]["confidence"])
    accepted = predictions[0]["confidence"] >= confidence_threshold
    return {
        "predictions": predictions,
        "description": describe_image(image, confidence_threshold),
        "content_hint": _detect_content_hint(image),
        "confidence_level": level,
        "confidence_threshold": confidence_threshold,
        "accepted": accepted,
        "uncertain": not accepted,
        "model": "ResNet50 / ImageNet-1K",
    }


def classify_image(image: np.ndarray) -> Tuple[str, float]:
    """保留原接口，供旧代码调用；返回 Top-1 显示名称及置信度。"""
    top = classify_topk(image, top_k=1)["predictions"][0]
    name = top["label_zh"] or top["category_zh"]
    return f"{name} ({top['label_en']})", top["confidence"]


if __name__ == "__main__":
    test_img = np.full((224, 224, 3), (200, 100, 50), dtype=np.uint8)
    result = classify_topk(test_img)
    print("Top-5 分类结果：")
    for item in result["predictions"]:
        name = item["label_zh"] or item["category_zh"]
        print(f"{item['rank']}. {name} / {item['label_en']}: {item['confidence']:.2%}")
    print(f"可信度：{result['confidence_level']}")
