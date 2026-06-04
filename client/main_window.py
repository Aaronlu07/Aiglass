from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QPlainTextEdit, QGroupBox, QLineEdit, QCheckBox, QSpinBox, QComboBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtCore import QUrl
import time
import socket
import cv2
import json
import base64
import numpy as np
import threading
import urllib.request
import winsound
import math
import os
import zipfile
import wave
import io

from aiglass.core.pipeline import Pipeline, PipelineConfig
from aiglass.tts.tts import speak
import requests
import dashscope
from dashscope.audio.asr import Recognition

# 这里需要填入你的阿里云 DashScope API Key
# 注意：这只是个测试 Key，请务必替换成你自己的！
dashscope.api_key = " " 

def recognize_audio_dashscope(audio_file_path):
    """使用阿里云通义实验室的 Paraformer 语音识别 (国内直连，速度极快)"""
    try:
        recognition = Recognition(model='paraformer-realtime-v1',
                                  format='wav',
                                  sample_rate=16000,
                                  callback=None)
        result = recognition.call(audio_file_path)
        if result.status_code == 200:
            # 解析阿里云返回的句子
            sentences = result.get_sentence()
            if sentences and len(sentences) > 0:
                text = "".join([s['text'] for s in sentences])
                return text
            return ""
        else:
            print(f"ASR Error: {result.message}")
            return ""
    except Exception as e:
        print(f"ASR Exception: {e}")
        return ""

def call_doubao_vision(text, image_bgr):
    # 将图像尺寸缩小一半，极大地减少 base64 体积，加快网络上传速度，防止超时
    h, w = image_bgr.shape[:2]
    resized_img = cv2.resize(image_bgr, (w // 2, h // 2))
    
    # Convert image to base64 with lower JPEG quality (80 instead of default 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, buffer = cv2.imencode('.jpg', resized_img, encode_param)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    img_url = f"data:image/jpeg;base64,{img_b64}"
    
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Authorization": "Bearer ark-c8890dd1-3d07-4548-a95a-d8caf22bf2d0-7d016",
        "Content-Type": "application/json"
    }
    data = {
         "model": "doubao-seed-2-0-pro-260215", 
         "messages": [ 
             {
                 "role": "system",
                 "content": "你是一个智能导盲眼镜的视觉助手，名字叫“小镜”。请根据用户提供的第一人称视角照片和用户的语音指令，给出快速、精准、简短的回答。不要有多余的废话和客套话，重点是帮助视障用户寻物、避障或理解前方环境。如果是寻物指令，请明确指出物品的位置但注意你的看到的内容是左右镜像记得左右反着说（如：左前方、正前方、右下方）。"
             },
             { 
                 "role": "user", 
                 "content": [ 
                     { 
                         "type": "image_url", 
                         "image_url": {"url": img_url}
                     }, 
                     { 
                         "type": "text", 
                         "text": text 
                     } 
                 ] 
             } 
         ] 
    }
    
    # 增加网络超时时间至 30 秒，并保持重试机制
    for attempt in range(3):
        try:
            res = requests.post(url, headers=headers, json=data, timeout=30)
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content']
            else:
                if attempt == 2:
                    return f"大模型请求失败: {res.status_code} {res.text}"
                time.sleep(1) # 等待一秒后重试
        except requests.exceptions.Timeout:
            if attempt == 2:
                return "大模型处理太慢导致网络超时了，请稍后再试一次。"
            time.sleep(1)
        except Exception as e:
            if attempt == 2:
                return "抱歉，我现在连不上云端网络，请检查您的网络连接或稍后再试。"
            time.sleep(1)

class BackgroundAudioThread(QThread):
    log_signal = Signal(str)
    chat_signal = Signal(str, str)
    wake_state_signal = Signal(bool) # True 唤醒, False 睡眠
    trigger_vision_signal = Signal(str) # 发送语音文本，请求主线程触发大模型

    def __init__(self):
        super().__init__()
        self._stop = False
        # 增加大量与“小镜”谐音、形近或容易被ASR误识别的唤醒词，大幅提高唤醒成功率
        self.wake_words = [
            "小镜", "小静", "小金", "晓静", "小鸡", "小菁", "小敬", "小景", 
            "小晶", "小经", "小精", "小斤", "小进", "孝敬", "肖静", "笑晶",
            "小琴", "小青", "小晴", "小轻", "小庆", "小亲", "小请",
            "小丁", "小叮", "小顶", "小鼎", "小钉",
            "小星", "小新", "小信", "小鑫", "小欣", "小馨",
            "叫镜", "叫静", "叫金", "小叽", "小吉", "小极", "小急", "小剂", "小挤"
        ]

    def run(self):
        self._stop = False
        import speech_recognition as sr
        
        # --- Vosk 离线模型初始化 ---
        self.log_signal.emit("🔄 正在初始化完全离线语音模型 (Vosk)...")
        model_path = "model"
        if not os.path.exists(model_path):
            self.log_signal.emit("⬇️ 未检测到中文离线模型，正在自动下载 (约 40MB)，请稍候...")
            try:
                url = 'https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip'
                zip_path = 'vosk-model-small-cn.zip'
                urllib.request.urlretrieve(url, zip_path)
                self.log_signal.emit("📦 下载完成，正在解压...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall()
                os.rename('vosk-model-small-cn-0.22', 'model')
                os.remove(zip_path)
                self.log_signal.emit("✅ 离线模型安装成功！")
            except Exception as e:
                self.log_signal.emit(f"❌ 离线模型下载失败: {e}")
        
        try:
            from vosk import Model, KaldiRecognizer
            vosk_model = Model("model")
            self.log_signal.emit("✅ Vosk 离线模型加载成功，从此彻底告别网络超时！")
        except Exception as e:
            vosk_model = None
            self.log_signal.emit(f"⚠️ Vosk 离线模型加载失败: {e}，将回退到 Google 在线识别。")
            
        def recognize_vosk(wav_data):
            if not vosk_model:
                return ""
            try:
                # 转换 wav_data (16kHz 16bit Mono) 给 Vosk
                wf = wave.open(io.BytesIO(wav_data), "rb")
                rec = KaldiRecognizer(vosk_model, wf.getframerate())
                rec.SetWords(False)
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    rec.AcceptWaveform(data)
                result = json.loads(rec.FinalResult())
                return result.get("text", "").replace(" ", "")
            except Exception as e:
                print(f"Vosk error: {e}")
                return ""
        # -----------------------------

        r = sr.Recognizer()
        # 针对国内用户环境进行优化：降低固定阈值，关闭动态调整，防止自适应过高导致喊破喉咙都没反应
        r.energy_threshold = 500 
        r.dynamic_energy_threshold = False

        with sr.Microphone(sample_rate=16000) as source: # Vosk 需要 16000 采样率
            # 稍作噪音适应
            r.adjust_for_ambient_noise(source, duration=0.5)
            self.log_signal.emit(f"🎙️ [后台录音] 麦克风已就绪 (阈值:{r.energy_threshold})，正在监听唤醒词“小镜”...")
            
            while not self._stop:
                try:
                    # 1. 监听唤醒词
                    self.wake_state_signal.emit(False)
                    # timeout=1 表示如果1秒内没声音就进入下一次循环，不阻塞
                    audio = r.listen(source, timeout=1, phrase_time_limit=3)
                    
                    self.log_signal.emit("⏳ 听到声音，正在转写...")
                    # 优先尝试本地离线 Vosk 模型，失败再回退
                    try:
                        if vosk_model:
                            text = recognize_vosk(audio.get_wav_data())
                        else:
                            text = r.recognize_sphinx(audio, language='zh-CN')
                    except Exception as e_sphinx:
                        # 如果没有安装 sphinx/vosk 模型，则使用原生的 google 免费接口
                        try:
                            text = r.recognize_google(audio, language='zh-CN')
                        except sr.RequestError:
                            self.log_signal.emit("⚠️ 离线模型缺失且网络超时！请安装离线模型或开启代理。")
                            time.sleep(2)
                            continue
                    
                    if not text:
                        continue
                        
                    self.log_signal.emit(f"👂 [环境音] {text}")
                    
                    is_woken = False
                    for w in self.wake_words:
                        if w in text:
                            is_woken = True
                            break
                            
                    if is_woken:
                        # 触发唤醒
                        self.log_signal.emit("🔔 [唤醒] 检测到唤醒词，进入对话模式！")
                        self.wake_state_signal.emit(True)
                        winsound.Beep(1000, 150) # 滴声提示用户开始说话
                        
                        # 2. 监听用户正式指令
                        self.log_signal.emit("🎤 [聆听指令] 请说话...")
                        audio_cmd = r.listen(source, timeout=5, phrase_time_limit=10)
                        
                        self.log_signal.emit("⏳ 录音完成，正在识别指令...")
                        try:
                            if vosk_model:
                                cmd_text = recognize_vosk(audio_cmd.get_wav_data())
                            else:
                                cmd_text = r.recognize_sphinx(audio_cmd, language='zh-CN')
                        except Exception:
                            try:
                                cmd_text = r.recognize_google(audio_cmd, language='zh-CN')
                            except sr.RequestError:
                                self.log_signal.emit("⚠️ 离线模型缺失且网络超时！请安装离线模型或开启代理。")
                                continue
                        
                        if not cmd_text:
                            self.log_signal.emit("⚠️ 未听清指令，请重新唤醒。")
                            continue
                            
                        self.log_signal.emit(f"🗣️ [用户指令] {cmd_text}")
                        self.chat_signal.emit("user", cmd_text)
                        
                        # 通知主线程拿着最新的图像去请求豆包
                        self.trigger_vision_signal.emit(cmd_text)
                        
                        # 为了避免马上又监听到环境音，稍微睡一会儿
                        time.sleep(1)

                except sr.WaitTimeoutError:
                    pass # 超时正常，继续循环
                except sr.UnknownValueError:
                    pass # 没听清正常，继续循环
                except sr.RequestError as e:
                    self.log_signal.emit("⚠️ Google语音服务网络超时！请确保电脑已开启代理或VPN加速。")
                    time.sleep(2)
                except Exception as e:
                    if not self._stop:
                        self.log_signal.emit(f"⚠️ 麦克风监听异常: {e}")
                    time.sleep(1)

    def stop(self):
        self._stop = True

class DoubaoVisionThread(QThread):
    log_signal = Signal(str)
    chat_signal = Signal(str, str)
    
    def __init__(self, text, image_bgr):
        super().__init__()
        self.text = text
        self.image_bgr = image_bgr
        
    def run(self):
        try:
            self.log_signal.emit("🚀 正在请求视觉大模型...")
            response_text = call_doubao_vision(self.text, self.image_bgr)
            self.log_signal.emit(f"🤖 [小睛] {response_text}")
            self.chat_signal.emit("assistant", response_text)
            
            # 使用系统的 TTS 发声（会通过电脑音响或连接到电脑的耳机播放）
            try:
                import pyttsx3
                import pythoncom
                import winsound
                import os
                
                pythoncom.CoInitialize() # 必须在后台线程初始化 COM
                engine = pyttsx3.init()
                # 调慢一点语速，让视障用户听得更清楚
                rate = engine.getProperty('rate')
                engine.setProperty('rate', rate - 30)
                
                tts_wav = "temp_doubao_reply.wav"
                if os.path.exists(tts_wav):
                    try:
                        os.remove(tts_wav)
                    except:
                        pass
                        
                # 采用先保存为 wav 再通过 winsound 播放的机制，彻底解决多线程环境下的无声 Bug
                engine.save_to_file(response_text, tts_wav)
                engine.runAndWait()
                
                # 额外延迟一小会儿，确保底层文件句柄已经完全释放
                time.sleep(0.1)
                
                if os.path.exists(tts_wav):
                    # 确保路径是绝对路径，防止 winsound 找不到文件
                    abs_wav_path = os.path.abspath(tts_wav)
                    winsound.PlaySound(abs_wav_path, winsound.SND_FILENAME)
            except Exception as tts_e:
                self.log_signal.emit(f"⚠️ TTS 语音播报失败: {tts_e}")
                # 回退使用旧版 speak
                speak(response_text)
                
        except Exception as e:
            self.log_signal.emit(f"❌ 大模型线程异常: {e}")

try:
    from websockets.sync.client import connect
except ImportError:
    pass # Will handle later

def send_pcm_beep_to_esp32(freq=1000, duration_ms=150, sample_rate=16000):
    """
    生成 16-bit Mono 16kHz 的 PCM 音频数据并通过 HTTP 发送给 ESP32-CAM。
    """
    try:
        samples = int(sample_rate * duration_ms / 1000)
        audio_data = bytearray()
        for i in range(samples):
            # 生成正弦波，幅度设置为 12000 (16-bit max is 32767)
            t = i / sample_rate
            sample = int(math.sin(2 * math.pi * freq * t) * 12000)
            # 转成 16-bit 小端字节序
            audio_data.extend(sample.to_bytes(2, byteorder='little', signed=True))
        
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/device/audio",
            data=bytes(audio_data),
            headers={'Content-Type': 'application/octet-stream'}
        )
        with urllib.request.urlopen(req, timeout=1.0) as response:
            pass
    except Exception as e:
        print(f"Failed to send PCM audio to ESP32: {e}")

def play_and_send_beeps(freq1, dur1, freq2, dur2):
    # 设备扬声器已损坏，仅在 PC 端播放提示音
    winsound.Beep(freq1, dur1)
    time.sleep(0.1)
    winsound.Beep(freq2, dur2)

def play_and_send_wav(wav_path, pcm_path):
    # 设备扬声器已损坏，仅在 PC 端播放 wav (异步)
    if os.path.exists(wav_path):
        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

class NetworkThread(QThread):
    status = Signal(str)
    metrics = Signal(dict)

    def __init__(self, url="ws://127.0.0.1:8000/ws/client"):
        super().__init__()
        self.url = url
        self._stop = False
        self._ws = None

    def run(self):
        self._stop = False
        self.status.emit("connecting")
        try:
            from websockets.sync.client import connect
            with connect(self.url) as websocket:
                self._ws = websocket
                self.status.emit("connected")
                last_img = None
                while not self._stop:
                    try:
                        message = websocket.recv()
                        if isinstance(message, (bytes, bytearray)):
                            nparr = np.frombuffer(message, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is None:
                                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            last_img = img
                            continue

                        data = json.loads(message)

                        b64_img = data.get("image")
                        if b64_img:
                            try:
                                img_bytes = base64.b64decode(b64_img)
                                nparr = np.frombuffer(img_bytes, np.uint8)
                                last_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            except Exception:
                                pass

                        alerts = data.get("alerts") or []
                        text = data.get("text") or "；".join([a for a in alerts if a])
                        m = {
                            "frame_id": int(data.get("frame_id") or time.time() * 1000),
                            "enhanced": {
                                "image": last_img,
                                "value": data.get("metrics", {}).get("brightness", 0),
                            },
                            "detections": data.get("detections", []),
                            "perf": data.get("perf") or {},
                            "text": text,
                            "camera_ip": data.get("camera_ip"),
                            "spoken": False,
                            "conversation": data.get("conversation") or [],
                            "assistant": data.get("assistant") or {},
                        }
                        self.metrics.emit(m)
                    except Exception as e:
                        print(f"WS Error: {e}")
                        break
        except Exception as e:
            self.status.emit(f"error: {str(e)}")
        self._ws = None
        self.status.emit("stopped")

    def stop(self):
        self._stop = True
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


class PipelineThread(QThread):
    status = Signal(str)
    metrics = Signal(dict)

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._stop = False

    def run(self):
        self._stop = False
        self.status.emit("running")
        last_ts = 0.0
        last_fps = 0.0
        while not self._stop:
            try:
                m = self.pipeline.run_once()
                now = time.perf_counter()
                if last_ts > 0:
                    dt = now - last_ts
                    if dt > 0:
                        fps = 1.0 / dt
                        last_fps = fps if last_fps == 0.0 else (last_fps * 0.9 + fps * 0.1)
                last_ts = now
                perf = m.get("perf") or {}
                if last_fps > 0:
                    perf["fps"] = last_fps
                m["perf"] = perf
                self.metrics.emit(m)
            except Exception as e:
                self.status.emit(f"error: {str(e)}")
                break
            time.sleep(0.03)
        self.status.emit("stopped")

    def stop(self):
        self._stop = True


class MainWindow(QMainWindow):
    log_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        # --- 全局应用鲜艳且清晰的 QSS 样式 (针对论文截图优化) ---
        self.setStyleSheet("""
            QWidget {
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
                font-size: 16pt; 
                font-weight: bold;
                color: #1e293b;
                background-color: #f8fafc;
            }
            QGroupBox {
                font-weight: 900;
                font-size: 18pt;
                border: 3px solid #3b82f6;
                border-radius: 8px;
                margin-top: 24px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #2563eb;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                font-weight: 900;
                font-size: 16pt;
                border-radius: 6px;
                padding: 12px 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton#startBtn {
                background-color: #10b981; /* 鲜艳的绿色 */
            }
            QPushButton#startBtn:hover {
                background-color: #059669;
            }
            QPushButton#stopBtn {
                background-color: #ef4444; /* 鲜艳的红色 */
            }
            QPushButton#stopBtn:hover {
                background-color: #dc2626;
            }
            QPushButton#wakeBtn {
                background-color: #8b5cf6; /* 鲜艳的紫色 */
                font-size: 18pt;
                font-weight: 900;
                padding: 15px;
            }
            QPushButton#wakeBtn:hover {
                background-color: #7c3aed;
            }
            QLabel {
                font-weight: 900;
                font-size: 16pt;
                color: #0f172a;
            }
            QPlainTextEdit {
                background-color: #ffffff;
                border: 2px solid #cbd5e1;
                border-radius: 6px;
                font-family: 'Consolas', 'Microsoft YaHei', monospace;
                font-size: 15pt;
                font-weight: bold;
                color: #000000;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                font-size: 15pt;
                font-weight: bold;
                padding: 8px;
                border: 2px solid #94a3b8;
                border-radius: 4px;
                background-color: #ffffff;
                color: #000000;
            }
            QCheckBox {
                font-size: 16pt;
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
            }
        """)
    def _default_server_url(self):
        ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
        return f"http://{ip}:8000"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 盲人辅助眼镜系统 (演示版)")
        self.resize(1400, 900)
        self.gammaBox = QDoubleSpinBox()
        self.gammaBox.setRange(0.5, 3.0)
        self.gammaBox.setSingleStep(0.1)
        self.gammaBox.setValue(1.2)
        self.serverEdit = QLineEdit(self._default_server_url())
        self.cloudCheck = QCheckBox("启用云端推理")
        self.netModeCheck = QCheckBox("接收ESP32流 (网络模式)") # New Checkbox
        self.deviceFpsBox = QSpinBox()
        self.deviceFpsBox.setRange(5, 30)
        self.deviceFpsBox.setValue(25)  # 提高默认帧率以增加流畅度
        self.deviceFsBox = QComboBox()
        self.deviceFsBox.addItems(["QQVGA(160x120)", "QVGA(320x240)", "VGA(640x480)", "SVGA(800x600)", "XGA(1024x768)", "HD(1280x720)"])
        self.deviceFsBox.setCurrentIndex(4)  # 默认使用 XGA(1024x768) 以保证极高的画质和识别准确率
        self.deviceGrayCheck = QCheckBox("灰度")
        self.applyDeviceBtn = QPushButton("下发参数")
        self.sourceEdit = QLineEdit("0")
        self.startBtn = QPushButton("启动系统")
        self.startBtn.setObjectName("startBtn")
        self.stopBtn = QPushButton("停止系统")
        self.stopBtn.setObjectName("stopBtn")
        self.wakeBtn = QPushButton("🎤 唤醒对话")
        self.wakeBtn.setObjectName("wakeBtn")
        self.statusLabel = QLabel("就绪")
        self.chat_thread = None # Initialize chat_thread attribute
        self.cameraIpLabel = QLabel("未知")
        self.logView = QPlainTextEdit()
        self.logView.setReadOnly(True)
        self.logView.setUndoRedoEnabled(False)
        self.logView.document().setMaximumBlockCount(200)
        self.dialogView = QPlainTextEdit()
        self.dialogView.setReadOnly(True)
        self.dialogView.setUndoRedoEnabled(False)
        self.dialogView.document().setMaximumBlockCount(120)
        self.findTargetLabel = QLabel("寻物目标: 无")
        self.wakeStateLabel = QLabel("唤醒状态: 待唤醒")
        self.log_signal.connect(self.logView.appendPlainText)
        self._last_log_ts = 0.0
        self._last_log_text = ""
        self._last_render_ts = 0.0
        self._camera_connected = False
        self._last_alert_text = ""
        self._last_alert_ts = 0.0
        self._last_conversation_size = 0
        top = QWidget()
        self.setCentralWidget(top)
        outer = QVBoxLayout(top)
        
        # System Info Header
        header = QLabel("基于YOLO与大模型的端云协同盲人辅助系统")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 26px; font-weight: bold; margin: 10px; color: #1d4ed8;")
        outer.addWidget(header)

        ctrlGroup = QGroupBox("系统控制台")
        ctrlLayout = QHBoxLayout(ctrlGroup)
        ctrlLayout.addWidget(QLabel("图像增强强度 (Gamma):"))
        ctrlLayout.addWidget(self.gammaBox)
        ctrlLayout.addWidget(QLabel("云端地址:"))
        ctrlLayout.addWidget(self.serverEdit)
        ctrlLayout.addWidget(QLabel("摄像头源:"))
        ctrlLayout.addWidget(self.sourceEdit)
        ctrlLayout.addWidget(self.cloudCheck)
        ctrlLayout.addWidget(self.netModeCheck) # Add to layout
        ctrlLayout.addWidget(QLabel("ESP32 FPS:"))
        ctrlLayout.addWidget(self.deviceFpsBox)
        ctrlLayout.addWidget(QLabel("分辨率:"))
        ctrlLayout.addWidget(self.deviceFsBox)
        ctrlLayout.addWidget(self.deviceGrayCheck)
        ctrlLayout.addWidget(self.applyDeviceBtn)
        ctrlLayout.addWidget(QLabel("ESP32 IP:"))
        ctrlLayout.addWidget(self.cameraIpLabel)
        ctrlLayout.addWidget(self.startBtn)
        ctrlLayout.addWidget(self.stopBtn)
        ctrlLayout.addWidget(self.wakeBtn)
        ctrlLayout.addWidget(QLabel("当前状态:"))
        ctrlLayout.addWidget(self.statusLabel)
        outer.addWidget(ctrlGroup)
        perfGroup = QGroupBox("性能统计")
        perfLayout = QHBoxLayout(perfGroup)
        self.fpsLabel = QLabel("FPS: -")
        self.enhanceLabel = QLabel("增强: - ms")
        self.inferLabel = QLabel("推理: - ms")
        self.deviceLabel = QLabel("设备: -")
        self.gpuLabel = QLabel("GPU: -")
        perfLayout.addWidget(self.fpsLabel)
        perfLayout.addWidget(self.enhanceLabel)
        perfLayout.addWidget(self.inferLabel)
        perfLayout.addWidget(self.deviceLabel)
        perfLayout.addWidget(self.gpuLabel)
        outer.addWidget(perfGroup)
        previewGroup = QGroupBox("实时视觉反馈 (模拟端侧/云侧处理结果)")
        previewLayout = QVBoxLayout(previewGroup)
        self.imageView = QLabel()
        self.imageView.setAlignment(Qt.AlignCenter)
        self.imageView.setFixedSize(960, 540)
        previewLayout.addWidget(self.imageView)
        outer.addWidget(previewGroup)
        voiceGroup = QGroupBox("语音对话与寻物状态")
        voiceLayout = QVBoxLayout(voiceGroup)
        voiceInfo = QHBoxLayout()
        voiceInfo.addWidget(self.wakeStateLabel)
        voiceInfo.addSpacing(20)
        voiceInfo.addWidget(self.findTargetLabel)
        voiceInfo.addStretch(1)
        voiceLayout.addLayout(voiceInfo)
        voiceLayout.addWidget(self.dialogView)
        outer.addWidget(voiceGroup)
        logGroup = QGroupBox("日志")
        logLayout = QVBoxLayout(logGroup)
        logLayout.addWidget(self.logView)
        outer.addWidget(logGroup)
        self.thread = None
        self.pipeline = None
        self.bg_audio_thread = None # 后台音频监听线程
        
        self.startBtn.clicked.connect(self.on_start)
        self.stopBtn.clicked.connect(self.on_stop)
        self.wakeBtn.clicked.connect(self.on_manual_wake)
        self.applyDeviceBtn.clicked.connect(self.on_apply_device_config)
        self.gammaBox.valueChanged.connect(self.on_gamma_changed)
        self.serverEdit.textChanged.connect(self.on_gamma_changed)
        self.sourceEdit.textChanged.connect(self.on_gamma_changed)
        self.cloudCheck.stateChanged.connect(self.on_gamma_changed)

    def _post_json(self, url: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            body = resp.read()
        try:
            return json.loads(body.decode("utf-8", errors="ignore"))
        except Exception:
            return {"ok": False, "raw": body.decode("utf-8", errors="ignore")}

    def _send_device_cmd(self, cmd: str):
        base = (self.serverEdit.text() or "").strip().rstrip("/")
        if not base.startswith("http"):
            base = "http://" + base
        url = base + "/api/device/cmd"
        return self._post_json(url, {"cmd": cmd})

    def on_apply_device_config(self):
        if not self.netModeCheck.isChecked():
            self.log_signal.emit("请先勾选：接收ESP32流 (网络模式)")
            return

        fps = int(self.deviceFpsBox.value())
        fs_index = int(self.deviceFsBox.currentIndex())
        gray = 1 if self.deviceGrayCheck.isChecked() else 0
        cmds = [f"FPS={fps}", f"FS={fs_index}", f"GRAY={gray}"]

        def run():
            for c in cmds:
                try:
                    r = self._send_device_cmd(c)
                    if not r or not r.get("ok"):
                        self.log_signal.emit(f"下发失败: {c} -> {r}")
                    else:
                        self.log_signal.emit(f"已下发: {c}")
                except Exception as e:
                    self.log_signal.emit(f"下发异常: {c} -> {e}")

        threading.Thread(target=run, daemon=True).start()

    def on_manual_wake(self):
        # 用户点击按钮手动唤醒，发出滴声，并在后台线程插入模拟的唤醒指令
        winsound.Beep(1000, 150)
        self.log_signal.emit("🔔 [手动唤醒] 已触发，请在后台线程中继续说话...")
        # 为了兼容性，不直接操作后台线程，只发信号让它准备接收
        
    def on_wake_state_change(self, is_woken):
        self.wakeStateLabel.setText(f"唤醒状态: {'已唤醒' if is_woken else '待唤醒'}")
        
    def on_trigger_vision(self, text):
        if self.chat_thread is not None and self.chat_thread.isRunning():
            self.log_signal.emit("⚠️ 大模型请求正在进行中，请稍后再试...")
            return
            
        if not hasattr(self, '_last_bgr') or self._last_bgr is None:
            self.log_signal.emit("⚠️ 没有获取到当前摄像头画面，无法请求视觉模型！")
            return
            
        self.chat_thread = DoubaoVisionThread(text, self._last_bgr.copy())
        self.chat_thread.log_signal.connect(self.log_signal.emit)
        self.chat_thread.chat_signal.connect(self.on_chat_message)
        self.chat_thread.start()

    def on_chat_message(self, role, text):
        prefix = {"user": "你", "assistant": "助手"}.get(role, role)
        self.dialogView.appendPlainText(f"{prefix}: {text}")

    def on_gamma_changed(self, v):
        if self.netModeCheck.isChecked():
            self.pipeline = None
            return
        cfg = PipelineConfig(gamma=float(self.gammaBox.value()), server_url=self.serverEdit.text(), use_cloud=self.cloudCheck.isChecked(), dataset_source=self.sourceEdit.text())
        self.pipeline = Pipeline(cfg)

    def on_start(self):
        if self.thread is not None and self.thread.isRunning():
            return
            
        # 启动后台唤醒监听线程
        if self.bg_audio_thread is None or not self.bg_audio_thread.isRunning():
            self.bg_audio_thread = BackgroundAudioThread()
            self.bg_audio_thread.log_signal.connect(self.log_signal.emit)
            self.bg_audio_thread.chat_signal.connect(self.on_chat_message)
            self.bg_audio_thread.wake_state_signal.connect(self.on_wake_state_change)
            self.bg_audio_thread.trigger_vision_signal.connect(self.on_trigger_vision)
            self.bg_audio_thread.start()

        if self.netModeCheck.isChecked():
            # Network Mode
            base_url = self.serverEdit.text().strip().rstrip("/")
            if base_url.startswith("http"):
                url = base_url.replace("http", "ws") + "/ws/client"
            else:
                url = f"ws://{base_url}/ws/client"
            
            self.thread = NetworkThread(url)
            self.thread.status.connect(self.on_status)
            self.thread.metrics.connect(self.on_metrics)
            self.thread.start()
        else:
            # Local Mode
            if self.pipeline is None:
                self.on_gamma_changed(self.gammaBox.value())
            self.thread = PipelineThread(self.pipeline)
            self.thread.status.connect(self.on_status)
            self.thread.metrics.connect(self.on_metrics)
            self.thread.start()

    def on_stop(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(2000)
            self.thread = None
            
        if self.bg_audio_thread is not None:
            self.bg_audio_thread.stop()
            self.bg_audio_thread.wait(2000)
            self.bg_audio_thread = None
            
        # 安全地停止大模型请求线程（如果还在跑的话）
        if hasattr(self, 'chat_thread') and self.chat_thread is not None and self.chat_thread.isRunning():
            self.chat_thread.wait(2000)
            self.chat_thread = None
            
        self._camera_connected = False
        self._last_alert_text = ""
        self._last_alert_ts = 0.0
        self._last_conversation_size = 0
        self.dialogView.clear()
        self.findTargetLabel.setText("寻物目标: 无")
        self.wakeStateLabel.setText("唤醒状态: 待唤醒")

    def on_status(self, s: str):
        self.statusLabel.setText(s)
        self.logView.appendPlainText(s)

    def on_metrics(self, m: dict):
        now = time.perf_counter()
        camera_ip = m.get("camera_ip")
        if camera_ip:
            self.cameraIpLabel.setText(camera_ip)
            if not self._camera_connected:
                self._camera_connected = True
                # 连接成功时，发送“配网成功”语音
                threading.Thread(target=play_and_send_wav, args=("d:/FFOutput/ke/aiglass/配网成功.wav", "d:/FFOutput/ke/aiglass/配网成功_16k.pcm"), daemon=True).start()

        assistant = m.get("assistant") or {}
        active_target_cn = assistant.get("active_target_cn") or ""
        armed = bool(assistant.get("armed"))
        self.findTargetLabel.setText(f"寻物目标: {active_target_cn or '无'}")
        self.wakeStateLabel.setText(f"唤醒状态: {'已唤醒' if armed else '待唤醒'}")

        conversation = m.get("conversation") or []
        if len(conversation) < self._last_conversation_size:
            self.dialogView.clear()
            self._last_conversation_size = 0
        if len(conversation) > self._last_conversation_size:
            for item in conversation[self._last_conversation_size:]:
                role = item.get("role", "system")
                prefix = {"user": "你", "assistant": "助手", "system": "系统"}.get(role, role)
                text = item.get("text", "")
                if text:
                    self.dialogView.appendPlainText(f"{prefix}: {text}")
            self._last_conversation_size = len(conversation)

        # 检查是否有新的语音提醒内容，或者同一个提醒经过了一段时间 (例如 5 秒)
        alert_text = m.get("text", "")
        if alert_text:
            is_new = (alert_text != self._last_alert_text)
            is_time_to_repeat = (now - self._last_alert_ts) > 5.0
            
            if is_new or is_time_to_repeat:
                self._last_alert_text = alert_text
                self._last_alert_ts = now
                
                if "人" in alert_text:
                    # 遇到人时，播放并发送“避让行人”专属语音
                    threading.Thread(target=play_and_send_wav, args=("d:/FFOutput/ke/aiglass/避让行人.wav", "d:/FFOutput/ke/aiglass/避让行人_16k.pcm"), daemon=True).start()
                else:
                    # 有其他新提醒时，发出滴声，并使用自带TTS朗读
                    threading.Thread(target=play_and_send_beeps, args=(2000, 100, 2000, 100), daemon=True).start()
                    speak(alert_text)
        else:
            # 前方无障碍物时，清空上次记录，确保下次出现立即播报
            self._last_alert_text = ""

        perf = m.get("perf") or {}
        fps = perf.get("fps")
        if fps is not None:
            self.fpsLabel.setText(f"FPS: {fps:.1f}")
        enhance_ms = perf.get("enhance_ms")
        if enhance_ms is not None:
            self.enhanceLabel.setText(f"增强: {enhance_ms:.1f} ms")
        infer_ms = perf.get("infer_ms")
        if infer_ms is not None:
            self.inferLabel.setText(f"推理: {infer_ms:.1f} ms")
        device_name = perf.get("device")
        if device_name:
            self.deviceLabel.setText(f"设备: {device_name}")
        gpu_available = perf.get("gpu_available")
        gpu_name = perf.get("gpu_name") or ""
        if gpu_available is not None:
            gpu_text = "可用" if gpu_available else "不可用"
            if gpu_name:
                gpu_text = f"{gpu_text}({gpu_name})"
            self.gpuLabel.setText(f"GPU: {gpu_text}")
            
        # 隐藏频繁刷屏且无实际意义的内部日志
        # txt = f"id={m['frame_id']} e={m['enhanced']['value']:.3f} det={len(m['detections'])} text={m['text']} speak={m['spoken']}"
        # if txt != self._last_log_text or (now - self._last_log_ts) >= 0.5:
        #     self.logView.appendPlainText(txt)
        #     self._last_log_ts = now
        #     self._last_log_text = txt
            
        img = m.get("enhanced", {}).get("image")
        if img is not None and (now - self._last_render_ts) >= 0.033:
            self._last_bgr = img.copy() # 保存最新一帧的 BGR 数据给大模型使用
            # 客户端显示层：仅负责将画面旋转为正向（逆时针90度）供用户观看
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            dets = m.get("detections") or []
            vis = img.copy()
            h, w = vis.shape[:2]
            for d in dets:
                # 恢复之前推导出的完美坐标映射逻辑
                # 因为画面被逆时针旋转了90度，所以从云端（原始未旋转图）传来的坐标，
                # 必须经过以下映射才能贴合在旋转后的画面上：
                orig_cx = d.get("cx", 0.5) or 0.5
                orig_cy = d.get("cy", 0.5) or 0.5
                orig_w = d.get("w", 0.3) or 0.3
                orig_h = d.get("h", 0.3) or 0.3
                
                # 逆时针旋转 90 度的坐标映射
                new_cx = orig_cy
                new_cy = 1.0 - orig_cx
                new_w = orig_h
                new_h = orig_w
                
                cx = int(new_cx * w)
                cy = int(new_cy * h)
                bw = int(new_w * w)
                bh = int(new_h * h)
                x1 = max(0, cx - bw // 2)
                y1 = max(0, cy - bh // 2)
                x2 = min(w - 1, cx + bw // 2)
                y2 = min(h - 1, cy + bh // 2)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{d.get('cls','obj')}: {d.get('score',0.0):.2f}"
                cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if m.get("text"):
                cv2.putText(vis, m["text"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.imageView.width() if self.imageView.width() > 0 else w, self.imageView.height() if self.imageView.height() > 0 else h, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.imageView.setPixmap(pix)
            self._last_render_ts = now

    def closeEvent(self, event):
        self.on_stop()
        super().closeEvent(event)

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
