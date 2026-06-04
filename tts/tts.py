import pyttsx3
import threading
import time

_engine = None
_lock = threading.Lock()
_last_spoken = ""
_last_speak_time = 0

def _init_engine():
    global _engine
    if _engine is None:
        try:
            _engine = pyttsx3.init()
            _engine.setProperty('rate', 150)    # 语速
            _engine.setProperty('volume', 1.0)  # 音量
            # 尝试设置为中文语音
            voices = _engine.getProperty('voices')
            for voice in voices:
                if 'zh' in voice.id.lower() or 'chinese' in voice.name.lower():
                    _engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            print(f"TTS init error: {e}")
            _engine = False
    return _engine

def _speak_thread(text: str):
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except ImportError:
        pass
    
    engine = _init_engine()
    if engine:
        with _lock:
            try:
                import winsound
                import os
                
                tts_wav = "temp_sys_reply.wav"
                if os.path.exists(tts_wav):
                    try:
                        os.remove(tts_wav)
                    except:
                        pass
                        
                engine.save_to_file(text, tts_wav)
                engine.runAndWait()
                
                # 额外延迟，防止文件被锁死
                import time
                time.sleep(0.1)
                
                if os.path.exists(tts_wav):
                    abs_wav_path = os.path.abspath(tts_wav)
                    winsound.PlaySound(abs_wav_path, winsound.SND_FILENAME)
            except Exception as e:
                print(f"TTS say error: {e}")

def speak(text: str) -> bool:
    global _last_spoken, _last_speak_time
    if not text or len(text.strip()) == 0:
        return False
        
    now = time.time()
    # 简单的防抖：同一句话在 3 秒内不重复播报，避免一直逼逼叨
    if text == _last_spoken and (now - _last_speak_time) < 3.0:
        return False
        
    _last_spoken = text
    _last_speak_time = now
    
    # 异步播报，不阻塞主线程和推理线程
    threading.Thread(target=_speak_thread, args=(text,), daemon=True).start()
    return True
