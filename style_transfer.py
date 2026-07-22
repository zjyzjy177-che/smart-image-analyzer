"""
style_transfer.py — 艺术风格迁移模块
负责人：组员 B（加分项）
功能：素描黑白线条画 + 二次元动漫风格
运行：python style_transfer.py  # 自测
"""

import numpy as np
import cv2


def _pencil_sketch(image: np.ndarray) -> np.ndarray:
    """
    素描黑白线条画（多层融合）
    原理：灰度 → 反相 → 多尺度高斯模糊 → 颜色减淡 → 铅笔纹理
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 轻微对比度增强，突出细节
    gray = cv2.equalizeHist(gray)

    inverted = 255 - gray

    # 细线条层 — 小模糊核捕获细节
    blurred_fine = cv2.GaussianBlur(inverted, (5, 5), sigmaX=0, sigmaY=0)
    fine_lines = cv2.divide(gray, 255 - blurred_fine, scale=256)

    # 阴影层 — 大模糊核产生柔和阴影
    blurred_coarse = cv2.GaussianBlur(inverted, (31, 31), sigmaX=0, sigmaY=0)
    soft_shading = cv2.divide(gray, 255 - blurred_coarse, scale=256)

    # 中景层 — 中等模糊核平衡
    blurred_mid = cv2.GaussianBlur(inverted, (13, 13), sigmaX=0, sigmaY=0)
    mid_layer = cv2.divide(gray, 255 - blurred_mid, scale=256)

    # 三层融合：细线条为主，中加入结构性线条，粗加阴影深度
    result = cv2.addWeighted(fine_lines, 0.5, mid_layer, 0.3, 0)
    result = cv2.addWeighted(result, 0.85, soft_shading, 0.15, 0)

    # 铅笔纹理 — 细微噪声模拟纸面颗粒
    np.random.seed(42)
    grain = np.random.normal(0, 8, (h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), 0)
    result = np.clip(result.astype(np.float32) + grain, 0, 255)

    # 锐化 — 让线条更清晰
    kernel_sharpen = np.array([[-0.5, -1, -0.5],
                                [-1, 7, -1],
                                [-0.5, -1, -0.5]])
    result = cv2.filter2D(result.astype(np.uint8), -1, kernel_sharpen)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def _anime_cartoon(image: np.ndarray) -> np.ndarray:
    """
    二次元动漫风格
    原理：双边滤波去噪 → 自适应阈值提取轮廓 → 叠加平滑色彩
    """
    h, w = image.shape[:2]

    # 多次双边滤波 — 平滑颜色但保留边缘
    smooth = image.copy()
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

    # 灰度 + 中值模糊 + 自适应阈值 提取线条
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9, C=2)

    # 二值边缘膨胀一下让线条更粗
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 颜色量化 — 减少颜色数量，产生动漫色块感
    data = smooth.reshape(-1, 3).astype(np.float32)
    k = 16  # 聚类到16色
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(smooth.shape)

    # 合并：线条遮盖在量化色块上
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    result = np.where(edges_rgb < 128, 0, quantized)

    return result


def transfer_style(content_img: np.ndarray, style_name: str = "素描·黑白线条画"):
    """
    对图片进行风格迁移

    参数:
        content_img: 原始图片 (H, W, 3) RGB numpy array
        style_name: 风格名称

    返回:
        风格化后的图片 numpy array
    """
    print(f"[INFO] Applying style: {style_name}")

    if "素描" in style_name or "线条" in style_name or "sketch" in style_name.lower():
        result = _pencil_sketch(content_img)
    elif "动漫" in style_name or "二次元" in style_name or "anime" in style_name.lower():
        result = _anime_cartoon(content_img)
    else:
        # 默认走素描
        result = _pencil_sketch(content_img)

    print("[OK] Style transfer completed!")
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("Style Transfer Module - Self Test")
    print("=" * 50)

    import os
    test_path = "images/sample/face.jpg"
    if os.path.exists(test_path):
        test_img = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)
        print(f"Test image: {test_path} ({test_img.shape[1]}x{test_img.shape[0]})")
    else:
        test_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        print("No test image found, using random noise")

    for style in ["素描·黑白线条画", "二次元·动漫风格"]:
        print(f"\n--- {style} ---")
        result = transfer_style(test_img, style)
        print(f"  Input: {test_img.shape}, Output: {result.shape}")

    print("\n[OK] Style transfer module test completed!")
