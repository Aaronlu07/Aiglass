import cv2
import numpy as np
import os


class Dataset:
    def __init__(self, source: str = ""):
        self.source = source or ""
        self.cap = None
        self.mode = "synthetic"
        if self.source and self.source.lower().endswith((".mp4", ".avi", ".mov")):
            self.cap = cv2.VideoCapture(self.source)
            if self.cap and self.cap.isOpened():
                self.mode = "video"
            else:
                self.cap = None
                self.mode = "synthetic"
        elif self.source and self.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img = cv2.imread(self.source)
            if img is not None:
                self.mode = "image"
                self.image = img
            else:
                self.mode = "synthetic"
        else:
            idx = 0
            try:
                if self.source.strip():
                    idx = int(self.source.strip())
            except Exception:
                idx = 0
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not (self.cap and self.cap.isOpened()):
                self.cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if not (self.cap and self.cap.isOpened()):
                self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if self.cap and self.cap.isOpened():
                self.mode = "camera"
                self._apply_camera_prefs()
            else:
                self.cap = None
                self.mode = "synthetic"

    def _apply_camera_prefs(self):
        cap = self.cap
        if cap is None:
            return
        try:
            w = int(os.getenv("AIGLASS_CAM_WIDTH", "640"))
            h = int(os.getenv("AIGLASS_CAM_HEIGHT", "480"))
            fps = float(os.getenv("AIGLASS_CAM_FPS", "30"))
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            if w > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
            if h > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(fps))
        except Exception:
            pass

    def next_frame(self):
        if self.mode == "camera":
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            return self._synthetic_frame()
        if self.mode == "video":
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            return self._synthetic_frame()
        if self.mode == "image":
            return self.image.copy()
        return self._synthetic_frame()

    def _synthetic_frame(self, w: int = 640, h: int = 480):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(40, h - 40, 30):
            cv2.line(img, (40, i), (w - 40, i), (200, 200, 200), 2)
        for i in range(0, w, 80):
            cv2.line(img, (i, h - 60), (i + 40, h - 40), (180, 180, 180), 2)
        return img
