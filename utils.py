"""
utils.py — 通用工具函数
供所有模块共享使用
"""

import numpy as np
import os


def ensure_dir(path: str):
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)


def format_detections(detections):
    """格式化检测结果用于显示"""
    if not detections:
        return "未检测到任何物体"
    lines = [f"✅ 检测到 {len(detections)} 个物体："]
    for d in detections:
        lines.append(f"  · {d['label']} — 置信度 {d['confidence']:.1%}")
    return "\n".join(lines)


def resize_if_large(image: np.ndarray, max_size=1920):
    """如果图片太大，等比例缩小以提升处理速度"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        import cv2
        return cv2.resize(image, (new_w, new_h))
    return image
