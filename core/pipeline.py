import time
from dataclasses import dataclass
from aiglass.data.dataset import Dataset
from aiglass.algo.enhance import enhance_frame
from aiglass.algo.detect import local_detect, get_last_device
from aiglass.models.api import CloudModelClient
from aiglass.semantic.analysis import summarize
from aiglass.tts.tts import speak


@dataclass
class PipelineConfig:
    gamma: float = 1.2
    upload_interval_ms: int = 500
    speak_enabled: bool = True
    use_cloud: bool = False
    server_url: str = "http://127.0.0.1:8000"
    dataset_source: str = ""


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._running = False
        self.ds = Dataset(config.dataset_source)
        self.client = CloudModelClient(config.server_url)

    def capture(self):
        return self.ds.next_frame()

    def enhance(self, frame):
        # Use Retinex if Cloud Mode is enabled (Simulating powerful cloud enhancement)
        # Otherwise use simple Gamma/CLAHE (Simulating edge lightweight enhancement)
        return enhance_frame(frame, gamma=self.config.gamma, use_retinex=self.config.use_cloud)

    def detect(self, enhanced):
        if self.config.use_cloud:
            # Cloud Inference: Calls YOLO
            payload = {"enhanced": enhanced}
            return self.client.infer(payload)
        # Edge Inference: Calls lightweight algorithm (e.g. Stairs/Lines)
        return local_detect(enhanced, use_yolo=False)

    def speak(self, text):
        if not self.config.speak_enabled:
            return False
        return speak(text)

    def step(self, frame_id: int):
        f = self.capture()
        t0 = time.perf_counter()
        e = self.enhance(f)
        t1 = time.perf_counter()
        d = self.detect(e)
        t2 = time.perf_counter()
        t = summarize(d)
        s = self.speak(t)
        last_device = get_last_device()
        device_name = "cpu" if last_device in (None, "cpu") else "cuda"
        gpu_available = False
        gpu_name = ""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        perf = {
            "enhance_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "device": device_name,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name
        }
        return {"frame_id": frame_id, "frame": f, "enhanced": e, "detections": d, "text": t, "spoken": s, "perf": perf}

    def run_once(self):
        return self.step(int(time.time() * 1000))

    def self_test(self):
        out = []
        for i in range(3):
            out.append(self.step(i))
        return {"ok": True, "samples": out}
