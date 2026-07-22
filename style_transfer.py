"""
style_transfer.py — 艺术风格迁移模块
负责人：组员 B（加分项）
功能：将照片转换为指定艺术风格（梵高、莫奈等）
运行：python style_transfer.py  # 自测

🟢 注意：这是一个「锦上添花」的模块
如果时间不够、或者跑不起来，可以跳过这个功能
不影响项目核心功能（物体检测 + 分类 + OCR + 人脸检测）的完整性
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# ========== 快速风格迁移（推荐方案） ==========
# 使用 PyTorch 官方教程中的快速风格迁移方法
# 不需要训练，直接运行推理

def load_image(img: np.ndarray, size=512):
    """将 numpy 图片转为模型输入的 tensor"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor


def tensor_to_image(tensor):
    """将 tensor 转回 numpy 图片"""
    tensor = tensor.squeeze(0).cpu()
    tensor = tensor.clamp(0, 1)
    img = transforms.ToPILImage()(tensor)
    return np.array(img)


def transfer_style(
    content_img: np.ndarray,
    style_name: str = "梵高·星夜"
):
    """
    对图片进行风格迁移

    参数:
        content_img: 原始图片 numpy array
        style_name: 风格名称（预留参数，未来可扩展）

    返回:
        风格化后的图片 numpy array

    ⚠️ 这是一个简化版本，实际效果有限
    如果要更好的效果，可以：
    1. 使用预训练的 AdaIN 模型
    2. 使用 TensorFlow Hub 的 Arbitrary Image Stylization 模型
    3. 直接调用 API 服务

    TODO / 改进方向:
    - 加载预训练的快速风格迁移模型（如 pytorch 官方的 neural-style 模型）
    - 支持多种风格切换
    - 优化处理速度
    """
    print(f"🎨 正在应用风格: {style_name}")
    print("⚠️ 这是简化版风格迁移，实际项目可替换为预训练模型")

    # 这里给出的是效果展示用的伪代码框架
    # 实际实现时，需要加载一个预训练的风格迁移模型
    # 推荐用这个：pip install fast-neural-style
    # 或者直接下载预训练模型文件

    # ===== TODO: 以下为真实实现 =====
    # 方案1: 使用 fast-neural-style
    # from fast_neural_style import transformer
    # model = transformer.TransformerNetwork()
    # model.load_state_dict(torch.load('models/mosaic.pth'))
    # content = load_image(content_img)
    # output = model(content)
    # result = tensor_to_image(output)

    # 方案2: 简单的颜色映射（效果有限，但有变化）
    result = simple_color_shift(content_img, style_name)

    print("✅ 风格迁移完成！")
    return result


def simple_color_shift(img: np.ndarray, style: str):
    """
    简易"风格迁移"—— 调整颜色色调
    这不是真正的风格迁移，只是让效果「看起来」有变化
    正式版建议替换为真实的预训练模型
    """
    import cv2

    result = img.copy().astype(np.float32)

    if "梵高" in style:
        # 暖色调 + 增加对比度
        result[:, :, 0] *= 1.1  # R
        result[:, :, 2] *= 0.8  # B
    elif "莫奈" in style:
        # 冷色调 + 柔和
        result[:, :, 0] *= 0.8
        result[:, :, 2] *= 1.2
    elif "毕加索" in style:
        # 高饱和 + 蓝色调
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] *= 1.3  # 提高饱和度
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        # 默认 - 怀旧棕色调
        result[:, :, 0] *= 1.0
        result[:, :, 1] *= 0.9
        result[:, :, 2] *= 0.7

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


if __name__ == "__main__":
    """自测：python style_transfer.py"""
    print("=" * 50)
    print("风格迁移模块 — 自测模式")
    print("=" * 50)

    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    test_img[:] = (100, 150, 200)

    result = transfer_style(test_img, "梵高·星夜")
    print(f"输入图片大小: {test_img.shape}")
    print(f"输出图片大小: {result.shape}")
    print("✅ 风格迁移模块测试完成！")
