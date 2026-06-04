import os
import sys
import asyncio
import time
import uvicorn
import base64
import socket
import threading
import cv2
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Header
from fastapi.responses import HTMLResponse
from typing import Optional
from typing import List
import json

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from aiglass.core.pipeline import Pipeline, PipelineConfig
from aiglass.algo.enhance import enhance_frame
from aiglass.algo.detect import get_last_device
from aiglass.tts.tts import speak

app = FastAPI()

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Glass Web Viewer</title>
    <meta charset="utf-8">
    <style>
        body { font-family: system-ui; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .card { background: #1e293b; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
        .stats { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; font-weight: bold; margin-bottom: 15px; }
        .video-container { position: relative; width: 100%; max-width: 640px; margin: 0 auto; background: #000; border-radius: 8px; overflow: hidden; }
        #video-img { width: 100%; display: block; }
        .overlay { position: absolute; border: 2px solid #22c55e; box-sizing: border-box; pointer-events: none; }
        .label { position: absolute; background: #22c55e; color: #000; font-size: 12px; font-weight: bold; padding: 2px 4px; top: -20px; left: -2px; white-space: nowrap; }
        .logs { background: #0f172a; border: 1px solid #334155; padding: 10px; height: 150px; overflow-y: auto; border-radius: 8px; font-family: monospace; font-size: 12px; }
        .log-line { margin-bottom: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>AI Glass Web Viewer</h2>
        </div>
        <div class="card">
            <div class="stats">
                <span id="stat-fps">FPS: -</span>
                <span id="stat-infer">Infer: - ms</span>
                <span id="stat-dev">Device: -</span>
                <span id="stat-ip">ESP32 IP: -</span>
            </div>
            <div class="video-container" id="vc">
                <img id="video-img" alt="Waiting for stream..." />
                <div id="overlays"></div>
            </div>
        </div>
        <div class="card">
            <h4>Alerts / Logs</h4>
            <div class="logs" id="logs"></div>
        </div>
    </div>
    <script>
        const img = document.getElementById('video-img');
        const overlays = document.getElementById('overlays');
        const logs = document.getElementById('logs');
        let currentUrl = null;
        
        function connect() {
            const wsUrl = `ws://${location.host}/ws/client`;
            const ws = new WebSocket(wsUrl);
            ws.binaryType = 'blob';
            
            ws.onopen = () => log('Connected to server');
            ws.onclose = () => { log('Disconnected. Reconnecting...'); setTimeout(connect, 2000); };
            ws.onerror = (e) => log('WS Error');
            
            ws.onmessage = (e) => {
                if (e.data instanceof Blob) {
                    if (currentUrl) URL.revokeObjectURL(currentUrl);
                    currentUrl = URL.createObjectURL(e.data);
                    img.src = currentUrl;
                } else {
                    try {
                        const data = JSON.parse(e.data);
                        if (data.type === 'meta') {
                            document.getElementById('stat-fps').innerText = `FPS: ${data.perf.fps ? data.perf.fps.toFixed(1) : '-'}`;
                            document.getElementById('stat-infer').innerText = `Infer: ${data.perf.infer_ms ? data.perf.infer_ms.toFixed(1) : '-'} ms`;
                            document.getElementById('stat-dev').innerText = `Device: ${data.perf.device || '-'}`;
                            document.getElementById('stat-ip').innerText = `ESP32 IP: ${data.camera_ip || '-'}`;
                            
                            // Draw boxes
                            overlays.innerHTML = '';
                            const w = img.clientWidth;
                            const h = img.clientHeight;
                            if (w > 0 && h > 0 && data.detections) {
                                data.detections.forEach(d => {
                                    const bw = d.w * w;
                                    const bh = d.h * h;
                                    const bx = (d.cx * w) - bw/2;
                                    const by = (d.cy * h) - bh/2;
                                    
                                    const box = document.createElement('div');
                                    box.className = 'overlay';
                                    box.style.left = bx + 'px';
                                    box.style.top = by + 'px';
                                    box.style.width = bw + 'px';
                                    box.style.height = bh + 'px';
                                    
                                    const label = document.createElement('div');
                                    label.className = 'label';
                                    label.innerText = `${d.label || d.class_name || 'obj'} ${(d.conf || d.score || 0).toFixed(2)}`;
                                    box.appendChild(label);
                                    overlays.appendChild(box);
                                });
                            }
                            
                            if (data.text) {
                                log(`Alert: ${data.text}`);
                            }
                        }
                    } catch (err) {}
                }
            };
        }
        
        function log(msg) {
            const div = document.createElement('div');
            div.className = 'log-line';
            div.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
            logs.appendChild(div);
            logs.scrollTop = logs.scrollHeight;
            while(logs.childNodes.length > 50) logs.removeChild(logs.firstChild);
        }
        
        connect();
    </script>
</body>
</html>
"""

@app.get("/")
async def get_index():
    return HTMLResponse(html_content)

OPTIMIZE_ONLY = os.getenv("AIGLASS_OPTIMIZE_ONLY", "0") == "1"
OPT_GRAY = os.getenv("AIGLASS_OPT_GRAY", "0") == "1"
OPT_WIDTH = int(os.getenv("AIGLASS_OPT_WIDTH", "0"))  # 修改为0，保持原生分辨率，不缩小画面以保证最高画质
OPT_JPEG_QUALITY = int(os.getenv("AIGLASS_OPT_JPEG_QUALITY", "80"))  # 提高传输到前端的画质
OPT_GAMMA = float(os.getenv("AIGLASS_OPT_GAMMA", "1.2"))
OPT_MIN_INTERVAL = float(os.getenv("AIGLASS_OPT_MIN_INTERVAL", "0.033"))
FAST_ENHANCE = os.getenv("AIGLASS_FAST_ENHANCE", "1") == "1"
DET_WIDTH = int(os.getenv("AIGLASS_DET_WIDTH", "0"))  # 修改为0，保持原生分辨率给YOLO推理，极大提升准确率
DET_GRAY = os.getenv("AIGLASS_DET_GRAY", "0") == "1"
DET_CROP_RATIO = min(1.0, max(0.4, float(os.getenv("AIGLASS_DET_CROP_RATIO", "1.0"))))  # 修改为1.0，不裁剪画面，防止丢失边缘目标
ENHANCE_EVERY_N = max(1, int(os.getenv("AIGLASS_ENHANCE_EVERY_N", "1")))
DET_MIN_INTERVAL = max(0.03, float(os.getenv("AIGLASS_DET_MIN_INTERVAL", "0.12")))
DET_FORCE_INTERVAL = max(DET_MIN_INTERVAL, float(os.getenv("AIGLASS_DET_FORCE_INTERVAL", "0.45")))
DET_CHANGE_THRESHOLD = max(0.0, float(os.getenv("AIGLASS_DET_CHANGE_THRESHOLD", "6.0")))
MIN_SCORE_DEFAULT = float(os.getenv("AIGLASS_MIN_SCORE_DEFAULT", "0.35"))
MIN_SCORE_LIGHT = float(os.getenv("AIGLASS_MIN_SCORE_LIGHT", "0.45"))
MIN_SCORE_PERSON = float(os.getenv("AIGLASS_MIN_SCORE_PERSON", "0.35"))
DISCOVERY_PORT = int(os.getenv("AIGLASS_DISCOVERY_PORT", "8899"))
DISCOVERY_MAGIC = "AIGLASS_DISCOVER"
DISCOVERY_REPLY_PREFIX = "AIGLASS_SERVER"
MQTT_SUB_TOPIC = os.getenv("AIGLASS_MQTT_SUB", "aiglass/telemetry")
MQTT_RESULT_PREFIX = os.getenv("AIGLASS_MQTT_RESULT_PREFIX", "aiglass/device")
HTTP_DEVICE_HEADER = "x-aiglass-device"

def _gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {"available": True, "name": torch.cuda.get_device_name(0)}
        return {"available": False, "name": ""}
    except Exception:
        return {"available": False, "name": ""}

GPU_INFO = _gpu_info()
WAKE_WORDS = [w.strip() for w in os.getenv("AIGLASS_WAKE_WORDS", "小镜,小助理,你好眼镜").split(",") if w.strip()]
VOICE_ARM_WINDOW_S = float(os.getenv("AIGLASS_VOICE_ARM_WINDOW_S", "12"))
VOICE_REPLY_COOLDOWN_S = float(os.getenv("AIGLASS_VOICE_REPLY_COOLDOWN_S", "3"))

conversation_lock = threading.Lock()
conversation_stream = []
assistant_state = {
    "armed_until": 0.0,
    "active_target_en": "",
    "active_target_cn": "",
    "last_reply_ts": 0.0,
}
latest_detection_snapshot = []

TARGET_MAP = {
    "人": ("person", "人"),
    "行人": ("person", "行人"),
    "车辆": ("car", "车辆"),
    "汽车": ("car", "汽车"),
    "卡车": ("truck", "卡车"),
    "自行车": ("bicycle", "自行车"),
    "摩托车": ("motorcycle", "摩托车"),
    "公交车": ("bus", "公交车"),
    "红灯": ("red_light", "红灯"),
    "绿灯": ("green_light", "绿灯"),
    "斑马线": ("crosswalk", "斑马线"),
    "盲道": ("blind_road", "盲道"),
    "路障": ("roadblock", "路障"),
    "垃圾桶": ("ashcan", "垃圾桶"),
    "消防栓": ("fire_hydrant", "消防栓"),
    "锥桶": ("reflective_cone", "反光锥"),
    "路锥": ("reflective_cone", "反光锥"),
    "警示柱": ("warning_column", "警示柱"),
    "树": ("tree", "树"),
    "杆子": ("pole", "杆子"),
    "柱子": ("pole", "柱子"),
    "狗": ("dog", "狗"),
    "标志牌": ("sign", "标志牌"),
    "三轮车": ("tricycle", "三轮车"),
}

def _append_conversation(role: str, text: str):
    if not text:
        return
    with conversation_lock:
        conversation_stream.append({
            "role": role,
            "text": text,
            "ts": time.time(),
        })
        if len(conversation_stream) > 30:
            del conversation_stream[:-30]

def _conversation_tail():
    with conversation_lock:
        return list(conversation_stream[-12:])

def _assistant_meta():
    now = time.time()
    return {
        "wake_words": WAKE_WORDS,
        "armed": assistant_state["armed_until"] > now,
        "active_target_en": assistant_state["active_target_en"],
        "active_target_cn": assistant_state["active_target_cn"],
    }

def _safe_to_dict(x):
    if isinstance(x, dict):
        return x
    for attr in ("to_dict", "model_dump", "__dict__"):
        try:
            v = getattr(x, attr, None)
        except Exception:
            v = None
        if callable(v):
            try:
                d = v()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        elif isinstance(v, dict):
            return v
    return {"_raw": str(x)}

def _extract_sentence(event_obj):
    d = _safe_to_dict(event_obj)
    cands = [d]
    for k in ("output", "data", "result"):
        v = d.get(k)
        if isinstance(v, dict):
            cands.append(v)
    for obj in cands:
        sent = obj.get("sentence")
        if isinstance(sent, dict):
            text = sent.get("text")
            is_end = sent.get("sentence_end")
            if is_end is not None:
                is_end = bool(is_end)
            return text, is_end
    for obj in cands:
        text = obj.get("text")
        if isinstance(text, str):
            return text, None
    return None, None

def _contains_wake_word(text: str):
    return any(w in text for w in WAKE_WORDS)

def _strip_wake_word(text: str):
    out = text
    for w in WAKE_WORDS:
        out = out.replace(w, "")
    return out.strip(" ，。,.!！?")

def _find_target_from_text(text: str):
    for key, value in TARGET_MAP.items():
        if key in text:
            return value
    return None, None

def _extract_target_with_llm(text: str):
    api_key = os.getenv("AIGLASS_LLM_API_KEY", "sk-RXrfKQ3dNpaOjLcUGNtfJK9hKPmTjToAbF2fswC4OOGHVCZa")
    base_url = os.getenv("AIGLASS_LLM_BASE_URL", "https://www.chatnode.org/v1")
    model = os.getenv("AIGLASS_LLM_MODEL", "deepseek-v3.1")
    
    system_prompt = """
    你是一个视障辅助眼镜的智能助手。用户的语音指令可能包含要寻找的目标。
    请从用户的语音文本中提取出他们想要寻找的物品名称。
    如果你识别到了要找的物品，请返回其英文类别名和中文类别名，以 JSON 格式返回：{"target_en": "英文名", "target_cn": "中文名"}。
    如果是通用的寻物指令但没有明确目标，或者不是寻物指令，请返回 {"target_en": "", "target_cn": ""}。
    目前系统支持的物品类别英文名和中文名映射如下：
    person: 人/行人
    car: 车辆/汽车
    truck: 卡车
    bicycle: 自行车
    motorcycle: 摩托车
    bus: 公交车
    red_light: 红灯
    green_light: 绿灯
    crosswalk: 斑马线
    blind_road: 盲道
    roadblock: 路障
    ashcan: 垃圾桶
    fire_hydrant: 消防栓
    reflective_cone: 反光锥
    warning_column: 警示柱
    tree: 树
    pole: 杆子/柱子
    dog: 狗
    sign: 标志牌
    tricycle: 三轮车
    如果你发现用户要找的物品在上述列表中，请严格使用对应的英文名。如果不在，请自行翻译一个合适的英文单词作为 target_en。
    只返回 JSON，不要返回其他任何内容。
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1
    }
    
    try:
        res = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=5.0)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        return parsed.get("target_en", ""), parsed.get("target_cn", "")
    except Exception as e:
        print(f"[LLM Target Extraction Failed]: {e}")
        return _find_target_from_text(text)

def _reply_text(text: str):
    assistant_state["last_reply_ts"] = time.time()
    _append_conversation("assistant", text)
    try:
        speak(text)
    except Exception:
        pass
    return text

def _format_position(det: dict):
    cx = float(det.get("cx", 0.5))
    if cx < 0.38:
        return "左前方"
    if cx > 0.62:
        return "右前方"
    return "正前方"

def _summarize_front_scene():
    if not latest_detection_snapshot:
        return "当前没有识别到明显目标。"
    names = []
    for d in latest_detection_snapshot[:5]:
        cls_name = str(d.get("cls") or "目标")
        names.append(cls_name)
    return "当前前方检测到：" + "、".join(names) + "。"

def _process_voice_text(text: str):
    now = time.time()
    cleaned = (text or "").strip()
    if not cleaned:
        return "没有识别到有效语音。"

    _append_conversation("user", cleaned)
    had_wake = _contains_wake_word(cleaned)
    if had_wake:
        assistant_state["armed_until"] = now + VOICE_ARM_WINDOW_S
        cleaned = _strip_wake_word(cleaned)

    armed = assistant_state["armed_until"] > now
    if not had_wake and not armed:
        return ""

    if not cleaned:
        return _reply_text("我在，请说你要找什么。")

    if "停止寻找" in cleaned or "取消寻物" in cleaned or "结束寻物" in cleaned:
        assistant_state["active_target_en"] = ""
        assistant_state["active_target_cn"] = ""
        assistant_state["armed_until"] = 0.0
        return _reply_text("已停止寻物。")

    if "前面有什么" in cleaned or "现在有什么" in cleaned:
        assistant_state["armed_until"] = now + VOICE_ARM_WINDOW_S
        return _reply_text(_summarize_front_scene())

    if "帮我找" in cleaned or "找一下" in cleaned or cleaned.startswith("找"):
        target_en, target_cn = _extract_target_with_llm(cleaned)
        if target_en:
            assistant_state["active_target_en"] = target_en
            assistant_state["active_target_cn"] = target_cn
            assistant_state["armed_until"] = now + VOICE_ARM_WINDOW_S
            return _reply_text(f"好的，我开始帮你找{target_cn}。")
        return _reply_text("我听到你想找东西，但还没有识别出目标名称。")

    if assistant_state["active_target_en"]:
        target_en, target_cn = _extract_target_with_llm(cleaned)
        if target_en:
            assistant_state["active_target_en"] = target_en
            assistant_state["active_target_cn"] = target_cn
            assistant_state["armed_until"] = now + VOICE_ARM_WINDOW_S
            return _reply_text(f"已切换为帮你找{target_cn}。")

    assistant_state["armed_until"] = now + VOICE_ARM_WINDOW_S
    return _reply_text("已收到你的指令，请再具体一些，例如帮我找人、帮我找盲道。")

def _augment_alerts_with_find_mode(detections, alerts, now_ts):
    latest_detection_snapshot[:] = list(detections)
    target_en = assistant_state["active_target_en"]
    target_cn = assistant_state["active_target_cn"]
    if not target_en:
        return alerts
    for d in detections:
        if str(d.get("cls")) != target_en:
            continue
        if now_ts - assistant_state["last_reply_ts"] < VOICE_REPLY_COOLDOWN_S:
            return alerts
        pos = _format_position(d)
        msg = f"已找到{target_cn}，在{pos}。"
        alerts = list(alerts)
        if msg not in alerts:
            alerts.append(msg)
        _reply_text(msg)
        return alerts
    return alerts

def _transcribe_pcm16_bytes(audio_bytes: bytes):
    try:
        from dashscope import audio as dash_audio
    except Exception as e:
        return "", f"dashscope not available: {e}"

    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-2fcb1b369c094eb29ec2e90c2fb7e937").strip()
    if not api_key:
        return "", "missing DASHSCOPE_API_KEY"

    done = threading.Event()
    finals = []
    errors = []

    class BatchCallback:
        def on_open(self):
            pass
        def on_close(self):
            done.set()
        def on_complete(self):
            done.set()
        def on_error(self, err):
            errors.append(str(err))
            done.set()
        def on_result(self, result):
            text, is_end = _extract_sentence(result)
            if text:
                if is_end is True:
                    finals.append(text.strip())
        def on_event(self, event):
            self.on_result(event)

    recognition = None
    try:
        recognition = dash_audio.asr.Recognition(
            api_key=api_key,
            model=os.getenv("AIGLASS_ASR_MODEL", "paraformer-realtime-v2"),
            format="pcm",
            sample_rate=16000,
            callback=BatchCallback(),
        )
        recognition.start()
        chunk_bytes = 640
        for i in range(0, len(audio_bytes), chunk_bytes):
            recognition.send_audio_frame(audio_bytes[i:i + chunk_bytes])
        silence = bytes(chunk_bytes)
        for _ in range(12):
            recognition.send_audio_frame(silence)
        recognition.stop()
        done.wait(6.0)
    except Exception as e:
        errors.append(str(e))
    finally:
        try:
            if recognition is not None:
                recognition.stop()
        except Exception:
            pass

    text = "".join(finals).strip()
    return text, ("; ".join(errors) if errors else "")

def _prepare_detection_frame(frame):
    h, w = frame.shape[:2]
    crop_w = max(32, int(w * DET_CROP_RATIO))
    crop_h = max(32, int(h * DET_CROP_RATIO))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    roi = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    if DET_GRAY:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    det_frame = roi
    if DET_WIDTH > 0 and roi.shape[1] > DET_WIDTH:
        scale = DET_WIDTH / roi.shape[1]
        det_frame = cv2.resize(roi, (DET_WIDTH, int(roi.shape[0] * scale)))
    return det_frame, (x0, y0, crop_w, crop_h, w, h)

def _remap_detections(detections, crop_info):
    x0, y0, crop_w, crop_h, full_w, full_h = crop_info
    mapped = []
    for d in detections:
        item = dict(d)
        item["cx"] = (x0 + float(d.get("cx", 0.5)) * crop_w) / float(full_w)
        item["cy"] = (y0 + float(d.get("cy", 0.5)) * crop_h) / float(full_h)
        item["w"] = float(d.get("w", 0.0)) * crop_w / float(full_w)
        item["h"] = float(d.get("h", 0.0)) * crop_h / float(full_h)
        mapped.append(item)
    return mapped

def _motion_score(frame, prev_small):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64))
    if prev_small is None:
        return 999.0, small
    score = float(np.mean(cv2.absdiff(small, prev_small)))
    return score, small

# Global state
connected_camera: WebSocket = None
connected_clients: List[WebSocket] = []
pipeline = None
mqtt_last = None
mqtt_lock = threading.Lock()
mqtt_thread_started = False
mqtt_client_instance = None
last_camera_ip = None
discovery_thread_started = False
camera_send_lock = asyncio.Lock()

def _direction(cx: float):
    if cx < 0.4:
        return "左侧"
    if cx > 0.6:
        return "右侧"
    return "正前方"

def _is_forward(cx: float):
    return 0.4 <= cx <= 0.6

def _best_by_cls(detections, cls_name):
    best = None
    best_score = -1.0
    for d in detections:
        if d.get("cls") != cls_name:
            continue
        score = float(d.get("score") or 0.0)
        area = float(d.get("w") or 0.0) * float(d.get("h") or 0.0)
        weight = score * (0.5 + area)
        if weight > best_score:
            best_score = weight
            best = d
    return best

def _best_by_classes(detections, cls_list):
    best = None
    best_score = -1.0
    for d in detections:
        if d.get("cls") not in cls_list:
            continue
        score = float(d.get("score") or 0.0)
        area = float(d.get("w") or 0.0) * float(d.get("h") or 0.0)
        weight = score * (0.5 + area)
        if weight > best_score:
            best_score = weight
            best = d
    return best

def _should_emit(state, key, now, cooldown):
    last = state["last_alert_ts"].get(key, 0.0)
    if now - last < cooldown:
        return False
    state["last_alert_ts"][key] = now
    return True

def _build_alerts(detections, state, now):
    alerts = []
    blind_road = _best_by_cls(detections, "blind_road")
    red_light = _best_by_cls(detections, "red_light")
    green_light = _best_by_cls(detections, "green_light")
    obstacle = _best_by_classes(detections, {"car", "truck", "bus", "motorcycle", "bicycle", "tricycle", "person", "roadblock", "warning_column", "reflective_cone", "ashcan"})
    if blind_road:
        cx = float(blind_road.get("cx", 0.5))
        if abs(cx - 0.5) <= 0.18:
            if _should_emit(state, "blind_road_forward", now, 1.5):
                alerts.append("前方盲道")
        else:
            if _should_emit(state, "blind_road_offset", now, 1.5):
                alerts.append(f"盲道偏离，向{_direction(cx)}调整")
    light = None
    if red_light and green_light:
        light = red_light if float(red_light.get("score", 0.0)) >= float(green_light.get("score", 0.0)) else green_light
    else:
        light = red_light or green_light
    if light:
        cls = light.get("cls")
        cx = float(light.get("cx", 0.5))
        name = "红灯" if cls == "red_light" else "绿灯"
        if _is_forward(cx) and cls == "red_light":
            if _should_emit(state, "red_light_forward", now, 1.0):
                alerts.append("前方红灯")
        if state["last_light"] != cls:
            state["last_light"] = cls
            if _should_emit(state, "light_change", now, 1.0):
                alerts.append(f"{name}变化")
    if obstacle:
        cx = float(obstacle.get("cx", 0.5))
        area = float(obstacle.get("w", 0.0)) * float(obstacle.get("h", 0.0))
        if _is_forward(cx) and area >= 0.02:
            if _should_emit(state, "obstacle", now, 1.0):
                alerts.append("前方有障碍物")
    return alerts

def _filter_detections(detections):
    out = []
    for d in detections or []:
        cls = d.get("cls") or ""
        score = float(d.get("score") or 0.0)
        if cls in ("red_light", "green_light"):
            if score < MIN_SCORE_LIGHT:
                continue
        elif cls == "person":
            if score < MIN_SCORE_PERSON:
                continue
        else:
            if score < MIN_SCORE_DEFAULT:
                continue
        out.append(d)
    return out

def get_pipeline():
    global pipeline
    if pipeline is None:
        cfg = PipelineConfig(use_cloud=True)
        pipeline = Pipeline(cfg)
    return pipeline

def _local_ip_for_peer(peer_ip: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((peer_ip, 9))
        return sock.getsockname()[0]
    except Exception:
        return ""
    finally:
        sock.close()

def start_discovery_listener(server_port: int):
    global discovery_thread_started
    if discovery_thread_started:
        return

    def run():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", DISCOVERY_PORT))
            while True:
                try:
                    payload, addr = sock.recvfrom(512)
                    msg = payload.decode(errors="ignore").strip()
                    if not msg.startswith(DISCOVERY_MAGIC):
                        continue
                    local_ip = _local_ip_for_peer(addr[0])
                    if not local_ip:
                        continue
                    reply = f"{DISCOVERY_REPLY_PREFIX} {local_ip} {server_port} /ws/camera"
                    sock.sendto(reply.encode(), addr)
                except Exception as e:
                    print(f"Discovery listener error: {e}")
                    time.sleep(0.2)
        finally:
            sock.close()

    threading.Thread(target=run, daemon=True).start()
    discovery_thread_started = True

def start_mqtt_listener():
    global mqtt_thread_started, mqtt_client_instance
    if mqtt_thread_started:
        return
    try:
        import paho.mqtt.client as mqtt
    except Exception as e:
        print(f"MQTT disabled: {e}")
        return

    host = os.getenv("AIGLASS_MQTT_HOST", "127.0.0.1")
    port = int(os.getenv("AIGLASS_MQTT_PORT", "1883"))
    topic = MQTT_SUB_TOPIC

    def on_message(client, userdata, msg):
        global mqtt_last
        payload = msg.payload.decode(errors="ignore")
        try:
            data = json.loads(payload)
        except Exception:
            data = {"raw": payload}
        with mqtt_lock:
            mqtt_last = {"topic": msg.topic, "data": data, "ts": time.time()}

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(host, port, 30)
    client.subscribe(topic)
    client.loop_start()
    mqtt_client_instance = client
    mqtt_thread_started = True

def publish_mqtt_result(device_id: str, payload: dict):
    client = mqtt_client_instance
    if client is None:
        return
    topic = f"{MQTT_RESULT_PREFIX}/{device_id}/result"
    try:
        client.publish(topic, json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        print(f"MQTT publish failed: {e}")

def _decode_frame_bytes(data: bytes):
    nparr = np.frombuffer(data, np.uint8)
    raw_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if raw_frame is None:
        raw_frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if raw_frame is not None:
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
    return raw_frame

def _encode_preview_image(image):
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), OPT_JPEG_QUALITY])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

def _process_http_frame(raw_frame, device_id: str, source_ip: str):
    p = get_pipeline()
    now = time.monotonic()
    h, w = raw_frame.shape[:2]
    disp_frame = raw_frame
    if OPT_WIDTH > 0 and w > OPT_WIDTH:
        scale = OPT_WIDTH / w
        disp_frame = cv2.resize(raw_frame, (OPT_WIDTH, int(h * scale)))
    t0 = time.perf_counter()
    if FAST_ENHANCE:
        enhanced_res = enhance_frame(disp_frame, gamma=OPT_GAMMA, use_retinex=False)
    else:
        enhanced_res = p.enhance(disp_frame)
    enhance_ms = (time.perf_counter() - t0) * 1000.0
    enhanced_img = enhanced_res["image"]
    metrics = {"brightness": enhanced_res["value"], "sharpness": enhanced_res["sharpness"]}
    det_frame, crop_info = _prepare_detection_frame(raw_frame)
    t1 = time.perf_counter()
    det_res = p.detect({"image": det_frame, "value": metrics["brightness"], "sharpness": metrics["sharpness"]})
    infer_ms = (time.perf_counter() - t1) * 1000.0
    detections = _remap_detections(det_res, crop_info)
    alerts = _build_alerts(detections, {"last_light": None, "last_alert_ts": {}}, now)
    alerts = _augment_alerts_with_find_mode(detections, alerts, now)
    text = "；".join(alerts)
    payload = {
        "device_id": device_id,
        "camera_ip": source_ip,
        "detections": detections,
        "metrics": metrics,
        "perf": {
            "fps": 0.0,
            "enhance_ms": enhance_ms,
            "infer_ms": infer_ms,
            "device": "cpu" if get_last_device() in (None, "cpu") else "cuda",
            "gpu_available": GPU_INFO["available"],
            "gpu_name": GPU_INFO["name"]
        },
        "alerts": alerts,
        "text": text,
        "image": _encode_preview_image(enhanced_img),
        "conversation": _conversation_tail(),
        "assistant": _assistant_meta(),
    }
    publish_mqtt_result(device_id, payload)
    return payload

async def _broadcast_payload(payload: dict):
    to_remove = []
    for client in connected_clients:
        try:
            await client.send_json(payload)
        except Exception as e:
            print(f"Client error: {e}")
            to_remove.append(client)
    for client in to_remove:
        if client in connected_clients:
            connected_clients.remove(client)

    # 【恢复代码】如果有音频数据，将其下发给 ESP32-S3 设备端进行播放
    tts_audio = payload.get("tts_audio") if isinstance(payload, dict) else None
    device_ws = connected_camera
    if tts_audio and device_ws:
        try:
            await device_ws.send_bytes(tts_audio)
        except Exception as e:
            print(f"[WebSocket] 发送音频到设备端失败: {e}")

@app.post("/api/device/infer")
async def device_infer(request: Request, x_aiglass_device: Optional[str] = Header(default=None)):
    data = await request.body()
    if not data:
        return {"ok": False, "error": "empty body"}
    raw_frame = _decode_frame_bytes(data)
    if raw_frame is None:
        return {"ok": False, "error": "invalid image"}
    device_id = (x_aiglass_device or request.query_params.get("device_id") or "esp32").strip() or "esp32"
    source_ip = request.client.host if request.client else ""
    payload = _process_http_frame(raw_frame, device_id, source_ip)
    payload["ok"] = True
    payload["mqtt"] = mqtt_last
    await _broadcast_payload(payload)
    return payload

@app.post("/api/device/cmd")
async def device_cmd(request: Request):
    global connected_camera
    try:
        data = await request.json()
    except Exception:
        data = {}
    cmd = str((data or {}).get("cmd") or "").strip()
    if not cmd:
        return {"ok": False, "error": "empty cmd"}
    ws = connected_camera
    if ws is None:
        return {"ok": False, "error": "camera not connected"}
    try:
        async with camera_send_lock:
            await ws.send_text(cmd)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/device/audio")
async def device_audio(request: Request):
    """
    发送二进制音频数据给 ESP32-CAM 播放。
    请求体必须是 16kHz 16-bit 单声道的 RAW/PCM 二进制数据。
    """
    global connected_camera
    try:
        audio_data = await request.body()
    except Exception:
        audio_data = b""
    if not audio_data:
        return {"ok": False, "error": "empty audio data"}
    
    ws = connected_camera
    if ws is None:
        return {"ok": False, "error": "camera not connected"}
    
    try:
        async with camera_send_lock:
            await ws.send_bytes(audio_data)
        return {"ok": True, "bytes_sent": len(audio_data)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/device/audio_upload")
async def device_audio_upload(request: Request):
    audio_data = await request.body()
    if not audio_data:
        return {"ok": False, "error": "empty audio data"}

    text, err = _transcribe_pcm16_bytes(audio_data)
    reply = ""
    if text:
        reply = _process_voice_text(text)
    elif err:
        _append_conversation("system", f"ASR失败: {err}")

    payload = {
        "ok": True,
        "transcript": text,
        "reply": reply,
        "conversation": _conversation_tail(),
        "assistant": _assistant_meta(),
        "error": err,
    }
    await _broadcast_payload(payload)
    return payload

@app.websocket("/ws/camera")
async def camera_endpoint(websocket: WebSocket):
    global connected_camera, last_camera_ip
    if connected_camera is not None and connected_camera is not websocket:
        try:
            await connected_camera.close()
        except Exception:
            pass
    await websocket.accept()
    connected_camera = websocket
    try:
        last_camera_ip = websocket.client.host
    except Exception:
        last_camera_ip = None
    print("Camera connected!")
    
    p = None if OPTIMIZE_ONLY else get_pipeline()
    min_interval = OPT_MIN_INTERVAL
    last_sent = 0.0
    frame_idx = 0
    last_detections = []
    last_enhanced_img = None
    last_metrics = {"brightness": 0.0, "sharpness": 0.0}
    alert_state = {"last_light": None, "last_alert_ts": {}}
    last_fps = 0.0
    last_perf = {"enhance_ms": 0.0, "infer_ms": 0.0, "motion": 0.0}
    last_det_ts = 0.0
    last_det_small = None
    
    try:
        while True:
            # Receive bytes (JPEG) from ESP32
            data = await websocket.receive_bytes()
            
            # Decode image
            nparr = np.frombuffer(data, np.uint8)
            raw_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if raw_frame is None:
                raw_frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if raw_frame is not None:
                    raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
            
            if raw_frame is not None:
                if not connected_clients:
                    continue
                now = time.monotonic()
                if now - last_sent < min_interval:
                    continue
                h, w = raw_frame.shape[:2]
                disp_frame = raw_frame
                if OPT_WIDTH > 0 and w > OPT_WIDTH:
                    scale = OPT_WIDTH / w
                    disp_frame = cv2.resize(raw_frame, (OPT_WIDTH, int(h * scale)))
                if OPTIMIZE_ONLY:
                    if OPT_GRAY:
                        gray = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2GRAY)
                        disp_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    t0 = time.perf_counter()
                    enhanced_res = enhance_frame(disp_frame, gamma=OPT_GAMMA, use_retinex=False)
                    last_perf["enhance_ms"] = (time.perf_counter() - t0) * 1000.0
                    enhanced_img = enhanced_res["image"]
                    last_metrics = {"brightness": enhanced_res["value"], "sharpness": enhanced_res["sharpness"]}
                    detections = []
                    alerts = []
                    last_perf["infer_ms"] = 0.0
                else:
                    if frame_idx % ENHANCE_EVERY_N == 0:
                        t0 = time.perf_counter()
                        if FAST_ENHANCE:
                            enhanced_res = enhance_frame(disp_frame, gamma=OPT_GAMMA, use_retinex=False)
                        else:
                            enhanced_res = p.enhance(disp_frame)
                        last_perf["enhance_ms"] = (time.perf_counter() - t0) * 1000.0
                        enhanced_img = enhanced_res["image"]
                        last_enhanced_img = enhanced_img
                        last_metrics = {"brightness": enhanced_res["value"], "sharpness": enhanced_res["sharpness"]}
                    else:
                        enhanced_img = last_enhanced_img if last_enhanced_img is not None else disp_frame
                    det_frame, crop_info = _prepare_detection_frame(raw_frame)
                    motion_score, det_small = _motion_score(det_frame, last_det_small)
                    last_perf["motion"] = motion_score
                    det_due = (now - last_det_ts) >= DET_MIN_INTERVAL
                    det_force = (now - last_det_ts) >= DET_FORCE_INTERVAL
                    scene_changed = motion_score >= DET_CHANGE_THRESHOLD
                    if last_det_ts == 0.0 or det_force or (det_due and scene_changed):
                        last_det_small = det_small
                        last_det_ts = now
                        t1 = time.perf_counter()
                        det_res = p.detect({"image": det_frame, "value": last_metrics["brightness"], "sharpness": last_metrics["sharpness"]})
                        last_perf["infer_ms"] = (time.perf_counter() - t1) * 1000.0
                        last_detections = _remap_detections(det_res, crop_info)
                    detections = _filter_detections(last_detections)
                    alerts = _build_alerts(detections, alert_state, now)
                    alerts = _augment_alerts_with_find_mode(detections, alerts, now)
                
                # Prepare result to send to client
                # We send the ENHANCED image + Metadata

                # Encode enhanced image back to JPEG for transmission to client
                _, buffer = cv2.imencode('.jpg', enhanced_img, [int(cv2.IMWRITE_JPEG_QUALITY), OPT_JPEG_QUALITY])
                jpg_bytes = buffer.tobytes()
                frame_id = int(time.time() * 1000)
                
                frame_idx += 1
                dt = now - last_sent if last_sent > 0 else 0.0
                if dt > 0:
                    fps = 1.0 / dt
                    last_fps = fps if last_fps == 0.0 else (last_fps * 0.9 + fps * 0.1)
                last_sent = now

                # Broadcast to all connected clients
                to_remove = []
                with mqtt_lock:
                    mqtt_payload = mqtt_last
                for client in connected_clients:
                    try:
                        # Send JPEG bytes + JSON metadata (avoid base64 overhead)
                        last_device = get_last_device()
                        device_name = "cpu" if last_device in (None, "cpu") else "cuda"
                        await client.send_bytes(jpg_bytes)
                        await client.send_json({
                            "type": "meta",
                            "frame_id": frame_id,
                            "detections": detections,
                            "metrics": last_metrics,
                            "perf": {
                                "fps": last_fps,
                                "enhance_ms": last_perf["enhance_ms"],
                                "infer_ms": last_perf["infer_ms"],
                                "device": device_name,
                                "gpu_available": GPU_INFO["available"],
                                "gpu_name": GPU_INFO["name"]
                            },
                            "mqtt": mqtt_payload,
                            "alerts": alerts,
                            "text": "；".join(alerts),
                            "camera_ip": last_camera_ip,
                            "conversation": _conversation_tail(),
                            "assistant": _assistant_meta()
                        })
                    except Exception as e:
                        print(f"Client error: {e}")
                        to_remove.append(client)
                
                for c in to_remove:
                    connected_clients.remove(c)
                    
    except WebSocketDisconnect:
        print("Camera disconnected")
    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        if connected_camera is websocket:
            connected_camera = None
        try:
            await websocket.close()
        except Exception:
            pass

@app.websocket("/ws/client")
async def client_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")
    try:
        while True:
            # Keep connection alive, maybe receive commands
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        try:
            await websocket.close()
        except Exception:
            pass

def run_server(host="0.0.0.0", port=8000):
    if not OPTIMIZE_ONLY:
        start_mqtt_listener()
    start_discovery_listener(port)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
