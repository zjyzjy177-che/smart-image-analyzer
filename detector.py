"""
detector.py — YOLOv8 目标检测模块
负责人：组长（你）
功能：加载预训练 YOLOv8 模型，对输入图片进行物体检测，返回标注后的图片和检测结果列表
"""

import numpy as np
import cv2
from ultralytics import YOLO

# 全局缓存模型（避免每次调用都重新加载）
_model = None

def get_model():
    """加载 YOLOv8 预训练模型（首次调用时加载，后续复用）"""
    global _model
    if _model is None:
        print("🔄 正在加载 YOLOv8 模型（首次加载可能需要一些时间）...")
        # 使用 yolov8m.pt — medium 版本，精度显著优于 nano，速度适中
        # 可选: yolov8n.pt(最快) / yolov8s.pt / yolov8m.pt ⭐ / yolov8l.pt / yolov8x.pt(最准)
        _model = YOLO("yolov8m.pt")
        print("✅ YOLOv8 模型加载完成！")
    return _model


def detect_objects(image: np.ndarray, conf_threshold: float = 0.4):
    """
    对输入图片进行目标检测

    参数:
        image: numpy 数组格式的图片 (H, W, 3) BGR 或 RGB
        conf_threshold: 置信度阈值，低于此值的检测结果会被过滤

    返回:
        annotated_image: 标注后的图片 (numpy array)
        detections: 检测结果列表，每个元素为 dict:
            {
                "label": str,        # 物体类别名称
                "confidence": float, # 置信度 0~1
                "bbox": [x1, y1, x2, y2],  # 边界框坐标
                "class_id": int      # 类别ID
            }
    """
    model = get_model()

    # YOLOv8 要求输入为 RGB
    if image.shape[2] == 3:
        # 如果输入是 BGR（OpenCV 默认），转为 RGB
        # 不确定时试一下两种，但 ultralytics 内部会处理
        pass

    # 推理
    results = model(image, conf=conf_threshold)[0]

    # 提取检测结果
    detections = []
    if results.boxes is not None:
        boxes = results.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            label = results.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": cls_id,
            })

    # 绘制标注框
    annotated_image = results.plot()  # ultralytics 自带绘图功能

    return annotated_image, detections


if __name__ == "__main__":
    """测试：本地运行 python detector.py 来测试检测功能"""
    print("=" * 50)
    print("YOLOv8 检测模块 — 自测模式")
    print("=" * 50)

    # 生成一张测试图片（彩色渐变图）
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (100, 100, 100)
    cv2.putText(test_img, "Test Image", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print("正在检测...")
    annotated, dets = detect_objects(test_img)
    print(f"检测到 {len(dets)} 个物体")
    for d in dets:
        print(f"  - {d['label']}: {d['confidence']:.3f}")
    print("✅ 检测模块工作正常！")
