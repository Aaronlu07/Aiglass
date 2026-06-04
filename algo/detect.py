import os
import cv2
import numpy as np
from typing import List, Dict

# Global model instance to avoid reloading
_yolo_model = None
_last_device = "cpu"

# ================= 配置区域 =================
# 将模型路径指向你刚刚用 train_yolov8s_strong.py 训练出来的新权重
YOLO_MODEL_PATH = r"d:\FFOutput\ke\aiglass\runs\detect\runs\detect\train_yolov8s_strong\weights\best.pt"
YOLO_CONF = 0.55      # 置信度阈值（提高阈值以过滤类似“把人识别成垃圾桶”的低置信度误检）
YOLO_IOU = 0.45       # NMS IOU 阈值
YOLO_MAXDET = 100     # 每张图最多检测目标数

# ================= 动态分辨率控制 =================
# YOLO_IMGSZ 被移除了固定值，现在它会根据客户端传来的真实图片大小动态调整。
# 这样做既能保证极高的识别率，又能节省算力。
# ==========================================
STAIRS_ENABLED = os.getenv("AIGLASS_STAIRS_ENABLED", "1") == "1"
STAIRS_SCORE_TH = float(os.getenv("AIGLASS_STAIRS_SCORE_TH", "0.60"))
STAIRS_MIN_LINES = int(os.getenv("AIGLASS_STAIRS_MIN_LINES", "6"))

def _get_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def get_last_device():
    return _last_device

def _is_valid_model(path: str):
    try:
        if not os.path.exists(path):
            return False
        if os.path.isdir(path):
            return False
        size = os.path.getsize(path)
        return size > 1024 * 1024
    except Exception:
        return False

def _project_root():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))

def _candidate_paths():
    rels = [
        os.path.join("runs", "detect", "train2", "weights", "best.engine"),
        os.path.join("runs", "detect", "train2", "weights", "best.onnx"),
        os.path.join("runs", "detect", "train2", "weights", "best.pt"),
        os.path.join("runs", "detect", "train", "weights", "best.engine"),
        os.path.join("runs", "detect", "train", "weights", "best.onnx"),
        os.path.join("runs", "detect", "train", "weights", "best.pt"),
        "best.engine",
        "best.onnx",
        "best.pt"
    ]
    root = _project_root()
    out = []
    for p in rels:
        out.append(p)
        out.append(os.path.join(root, p))
    return out

def _pick_model_path():
    model_path = os.getenv("AIGLASS_YOLO_MODEL", "")
    if model_path:
        model_path = os.path.abspath(model_path)
        if _is_valid_model(model_path):
            return model_path
    for p in _candidate_paths():
        if _is_valid_model(p):
            return os.path.abspath(p)
    return "yolov8n"

def _get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    
    try:
        from ultralytics import YOLO
        import torch
        
        # 使用配置区域指定的新模型路径
        model_path = YOLO_MODEL_PATH
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Trying to fallback to yolov8n.pt...")
            model_path = "yolov8n.pt" # 最后的倔强
            
        model = YOLO(model_path)
        
        # Warmup (可选)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, verbose=False, device=_get_device(), imgsz=640)
        
        _yolo_model = model
        print(f"[Detect] YOLO model loaded successfully from {model_path} on {_get_device()}")
        return model
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None


def _horizontal_lines(gray: np.ndarray):
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=50, minLineLength=40, maxLineGap=10)
    selected = []
    if lines is None:
        return selected
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) < 10.0 or abs(abs(ang) - 180.0) < 10.0:
            selected.append((x1, y1, x2, y2))
    return selected


def _stair_score(lines: List[tuple], h: int):
    if len(lines) < 4:
        return 0.0
    ys = sorted([(min(y1, y2), max(y1, y2)) for _, y1, _, y2 in lines], key=lambda t: (t[0] + t[1]) * 0.5)
    centers = np.array([(a + b) * 0.5 for a, b in ys], dtype=np.float32)
    centers = centers.reshape(-1, 1)
    if len(centers) < 4:
        return 0.0
    diffs = np.diff(centers, axis=0).flatten()
    if len(diffs) == 0:
        return 0.0
    mean = float(np.mean(diffs))
    var = float(np.var(diffs))
    uniform = 1.0 / (1.0 + var)
    density = min(1.0, len(lines) / 12.0)
    range_ok = 1.0 if (centers.max() - centers.min()) > h * 0.25 else 0.0
    return max(0.0, min(1.0, 0.5 * uniform + 0.3 * density + 0.2 * range_ok))


def local_detect(enhanced: Dict, use_yolo: bool = False) -> List[Dict]:
    bgr = enhanced.get("image")
    if bgr is None:
        return []
    
    results = []
    h, w = bgr.shape[:2]

    # 1. Try YOLO Detection if requested (Simulating Cloud Inference)
    if use_yolo:
        model = _get_yolo_model()
        if model:
            try:
                device = _get_device()
                global _last_device
                _last_device = device
                use_half = device != "cpu"
                
                # 为了不改变原图尺寸，我们将用于推理的图片拷贝一份
                infer_img = bgr.copy()
                
                # 硬件旋转适配：如果收到的是横屏图片（宽大于高），则逆时针旋转 90 度
                # 这样可以让 YOLO 模型看到正立的人和车，大幅提高准确率
                rotated = False
                if infer_img.shape[1] > infer_img.shape[0]:
                    infer_img = cv2.rotate(infer_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated = True
                    h_infer, w_infer = infer_img.shape[:2]
                else:
                    h_infer, w_infer = h, w

                # Run inference
                # 动态计算推理分辨率：取图像长边并向上取整为 32 的倍数
                # 这样可以保证原生分辨率输入模型，绝对不改变它的长宽比和清晰度
                dynamic_imgsz = max(h_infer, w_infer)
                dynamic_imgsz = ((dynamic_imgsz + 31) // 32) * 32

                yolo_res = model(
                    infer_img,
                    verbose=False,
                    device=device,
                    imgsz=dynamic_imgsz,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    max_det=YOLO_MAXDET,
                    half=use_half,
                )
                for r in yolo_res:
                    boxes = r.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        
                        # Normalize coordinates using infer_img dimensions
                        cx_infer = (x1 + x2) / 2.0 / w_infer
                        cy_infer = (y1 + y2) / 2.0 / h_infer
                        bw_infer = (x2 - x1) / w_infer
                        bh_infer = (y2 - y1) / h_infer
                        
                        if rotated:
                            # 映射回原始未旋转的图像坐标系（客户端假设收到的是原始坐标）
                            cx = 1.0 - cy_infer
                            cy = cx_infer
                            bw = bh_infer
                            bh = bw_infer
                        else:
                            cx = cx_infer
                            cy = cy_infer
                            bw = bw_infer
                            bh = bh_infer
                        
                        results.append({
                            "cls": cls_name,
                            "score": conf,
                            "cx": cx,
                            "cy": cy,
                            "w": bw,
                            "h": bh
                        })
            except Exception as e:
                print(f"YOLO inference failed: {e}")
    else:
        # 2. Legacy Stair Detection (Simulating Edge Lightweight Detection)
        if not STAIRS_ENABLED:
            return results
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lines = _horizontal_lines(gray)
        if len(lines) < STAIRS_MIN_LINES:
            return results
        score = _stair_score(lines, h)
        if score >= STAIRS_SCORE_TH:
            xs = []
            ys = []
            for x1, y1, x2, y2 in lines:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            if xs and ys:
                x_min = max(0, min(xs))
                x_max = min(w - 1, max(xs))
                y_min = max(0, min(ys))
                y_max = min(h - 1, max(ys))
                cx = (x_min + x_max) * 0.5 / float(w)
                cy = (y_min + y_max) * 0.5 / float(h)
                bw = (x_max - x_min) / float(w)
                bh = (y_max - y_min) / float(h)
                results.append({"cls": "stairs", "score": score, "cx": cx, "cy": cy, "w": bw, "h": bh})

    return results
