"""
classifier.py — 图像分类模块
负责人：组员 A
功能：使用预训练 ResNet 模型对图片进行分类，识别图片内容
运行：python classifier.py  # 自测
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# 全局缓存模型
_model = None
_labels = None


def get_model():
    """
    加载预训练 ResNet50 模型
    TODO: 首次加载需要下载模型权重（~100MB），耐心等待
    """
    global _model, _labels
    if _model is None:
        print("🔄 正在加载 ResNet50 分类模型（首次加载需要下载~100MB权重）...")
        _model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        _model.eval()  # 切换到推理模式
        print("✅ 分类模型加载完成！")

        # 加载 ImageNet 类别标签
        import json
        import urllib.request
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        try:
            with urllib.request.urlopen(url) as f:
                _labels = json.loads(f.read().decode())
            print(f"✅ 已加载 {len(_labels)} 个分类标签")
        except:
            # 如果网络不行，使用本地备用标签
            print("⚠️ 无法下载标签，使用基础标签列表")
            _labels = [f"类别_{i}" for i in range(1000)]

    return _model, _labels


def classify_image(image: np.ndarray):
    """
    对输入图片进行分类

    参数:
        image: numpy 数组 (H, W, 3)

    返回:
        label: str 类别名称
        confidence: float 置信度 0~1
    """
    model, labels = get_model()

    # 预处理：将 numpy 转为 tensor
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image).unsqueeze(0)  # 加 batch 维度

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_prob, top_idx = torch.topk(probabilities, 3)

    # 取 Top-1 结果
    label = labels[top_idx[0].item()] if top_idx[0].item() < len(labels) else f"类别_{top_idx[0].item()}"
    confidence = top_prob[0].item()

    return label, confidence


if __name__ == "__main__":
    """自测：python classifier.py"""
    print("=" * 50)
    print("图像分类模块 — 自测模式")
    print("=" * 50)

    # 创建一个测试图片（纯色图）
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)
    test_img[:] = (200, 100, 50)  # 橙色

    print("正在分类...")
    label, conf = classify_image(test_img)
    print(f"🏷️ 分类结果: {label}")
    print(f"📊 置信度: {conf:.2%}")
    print("✅ 分类模块工作正常！")
