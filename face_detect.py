"""
face_detect.py — 人脸检测模块
负责人：组员 B
功能：使用 MTCNN 检测人脸，降级方案为 OpenCV DNN / Haar Cascade
运行：python face_detect.py  # 自测
"""

import numpy as np
import cv2


def get_detector():
    """
    加载人脸检测器，优先级：MTCNN > DNN > Haar Cascade
    """
    import os

    # 1) 尝试 MTCNN（精度最高，内置多尺度检测）
    try:
        from facenet_pytorch import MTCNN
        print("[INFO] Loading MTCNN face detector...")
        mtcnn = MTCNN(keep_all=True, device="cpu")
        return ("mtcnn", mtcnn)
    except ImportError:
        pass

    # 2) 尝试 DNN
    proto_path = "models/deploy.prototxt"
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
    if os.path.exists(proto_path) and os.path.exists(model_path):
        print("[INFO] Loading DNN face detector...")
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        return ("dnn", net)

    # 3) 降级为 Haar Cascade
    print("[INFO] Using OpenCV Haar Cascade...")
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        cascade = cv2.CascadeClassifier(cascade_path)
        return ("haar", cascade)

    raise FileNotFoundError("Cannot load face detection model!")


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

    if method == "mtcnn":
        # --- MTCNN（推荐路径）---
        # MTCNN 内置图像金字塔，对多尺度/侧脸/小脸均表现优异
        boxes, probs = detector.detect(image)

        if boxes is not None and probs is not None:
            for box, prob in zip(boxes, probs):
                if prob is not None and prob > 0.9:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(result_img, (fx, fy), (fx + fw, fy + fh),
                          (0, 255, 0), 2)
            label_y = fy - 10 if fy > 20 else fy + fh + 15
            cv2.putText(result_img, "Face", (fx, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    elif method == "dnn":
        # --- DNN 多尺度检测 ---
        scales = [1.0, 1.5, 2.0]
        all_detections = []

        for scale in scales:
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            if scaled_w > 3000 or scaled_h > 3000:
                continue

            scaled_img = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                    (scaled_w, scaled_h))
            blob = cv2.dnn.blobFromImage(scaled_img, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0))
            detector.setInput(blob)
            detections = detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.35:
                    box = detections[0, 0, i, 3:7] * np.array([scaled_w, scaled_h,
                                                                 scaled_w, scaled_h])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = int(x1 / scale), int(y1 / scale)
                    x2, y2 = int(x2 / scale), int(y2 / scale)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    all_detections.append((x1, y1, x2 - x1, y2 - y1,
                                           confidence))

        all_detections.sort(key=lambda x: x[4], reverse=True)
        while all_detections:
            best = all_detections.pop(0)
            faces.append((best[0], best[1], best[2], best[3]))
            keep = []
            for d in all_detections:
                if _box_iou(best[:4], d[:4]) < 0.4:
                    keep.append(d)
            all_detections = keep

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(result_img, (fx, fy), (fx + fw, fy + fh),
                          (0, 255, 0), 2)
            label_y = fy - 10 if fy > 20 else fy + fh + 15
            cv2.putText(result_img, "Face", (fx, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    else:
        # --- Haar Cascade ---
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


def _box_iou(a, b):
    """计算两个框的 IoU"""
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[0] + a[2], b[0] + b[2])
    yb = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0


if __name__ == "__main__":
    """自测：python face_detect.py"""
    print("=" * 50)
    print("Face Detection Module - Self Test")
    print("=" * 50)

    import os
    test_image_path = "images/sample/face.jpg"
    if os.path.exists(test_image_path):
        test_img = cv2.imread(test_image_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        print(f"Using test image: {test_image_path} ({test_img.shape[1]}x{test_img.shape[0]})")
    else:
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        test_img[:] = (200, 200, 200)
        cv2.ellipse(test_img, (200, 200), (80, 100), 0, 0, 360,
                    (220, 180, 150), -1)
        print("No test photo found, using synthetic test image")

    print("Detecting faces...")
    result, faces = detect_faces(test_img)
    print(f"Detected {len(faces)} face(s)")
    for f in faces:
        print(f"  - Position: ({f[0]}, {f[1]}), Size: {f[2]}x{f[3]}")

    if len(faces) == 0:
        print("[WARN] Test image too simple, try a real photo instead")

    print("[OK] Face detection module test completed!")
