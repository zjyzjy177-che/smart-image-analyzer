"""
face_detect.py — 人脸检测模块
负责人：组员 B
功能：使用 OpenCV 的 DNN 或 Haar Cascade 检测人脸
运行：python face_detect.py  # 自测
"""

import numpy as np
import cv2


def get_detector():
    """
    加载人脸检测器

    使用 OpenCV 的 DNN 模型（精度更高），需要下载模型文件
    如果 DNN 模型下载失败，自动降级为 Haar Cascade（无需下载，但精度稍低）
    """
    import os
    import urllib.request

    # 尝试加载 DNN 模型
    proto_path = "models/deploy.prototxt"
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

    if os.path.exists(proto_path) and os.path.exists(model_path):
        print("🔄 加载 DNN 人脸检测器...")
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        return ("dnn", net)
    else:
        # 降级为 Haar Cascade
        print("🔄 DNN 模型不存在，使用 OpenCV 内置 Haar Cascade...")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            return ("haar", cascade)
        else:
            raise FileNotFoundError("无法加载人脸检测模型！")


_detector = None


def detect_faces(image: np.ndarray):
    """
    检测图片中的人脸

    参数:
        image: numpy 数组 (H, W, 3)，RGB 格式

    返回:
        result_img: 标注了人脸框的图片
        faces: 人脸坐标列表 [(x, y, w, h), ...]
    """
    global _detector
    if _detector is None:
        _detector = get_detector()

    method, detector = _detector
    result_img = image.copy()
    faces = []

    h, w = image.shape[:2]

    if method == "dnn":
        # DNN 方法
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        detector.setInput(blob)
        detections = detector.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                faces.append((x1, y1, x2 - x1, y2 - y1))
                cv2.rectangle(result_img, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(result_img, f"Face {confidence:.1%}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

    else:
        # Haar Cascade 方法
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detected = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, fw, fh) in detected:
            faces.append((x, y, fw, fh))
            cv2.rectangle(result_img, (x, y), (x + fw, y + fh),
                          (0, 255, 0), 2)
            cv2.putText(result_img, "Face",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

    return result_img, faces


if __name__ == "__main__":
    """自测：python face_detect.py"""
    print("=" * 50)
    print("人脸检测模块 — 自测模式")
    print("=" * 50)

    # 创建一张含有人脸的测试图（画个椭圆模拟人脸）
    test_img = np.zeros((400, 400, 3), dtype=np.uint8)
    test_img[:] = (200, 200, 200)
    cv2.ellipse(test_img, (200, 200), (80, 100), 0, 0, 360,
                (220, 180, 150), -1)  # 肤色椭圆

    print("正在检测人脸...")
    result, faces = detect_faces(test_img)
    print(f"检测到 {len(faces)} 张人脸")
    for f in faces:
        print(f"  - 位置: ({f[0]}, {f[1]}), 大小: {f[2]}x{f[3]}")

    if len(faces) == 0:
        print("⚠️ 测试图可能太简单，换一张真实照片试试")

    print("✅ 人脸检测模块测试完成！")
