import os
import sys
import threading
import requests
import numpy as np
import sounddevice as sd
import time
import json  # ADDED

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLabel, QSlider,
    QSizePolicy, QFileDialog, QMessageBox, QLineEdit  # ADDED
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QSettings,QEvent
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QTextCursor

# ======================
# CONFIG
# ======================

KOBOLD_BASE = "http://localhost:5002"
TTS_BASE = "http://localhost:9000"

SAMPLE_RATE = 44100
CHANNELS = 1

WAVE_SAMPLES = 2048
WAVE_TIMER_MS = 30
MAX_HISTORY_MESSAGES = 20  # 10 turns

# ======================
# STT CONFIG
# ======================

STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_BLOCKSIZE = 1024
PTT_KEY = Qt.Key_Alt  # push-to-talk key

LEFT_ALT_SCANCODE = 0x38  # Windows left alt

# ======================
# AUDIO PLAYER
# ======================

class PCMPlayer:
    def __init__(self):
        self.stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=1024
        )
        self.stream.start()

        self.last_audio = np.zeros(WAVE_SAMPLES, dtype=np.int16)
        self.last_rms = 0.0

        self.auto_gain = False
        self.agc_mode = "rms"  # or "peak"
        self.agc_target_rms = 0.05
        self.agc_gain = 1.0
        self.agc_max_gain = 1.6
        self.agc_smoothing = 0.015

    def play(self, pcm_bytes, gain=1.5, limit=0.95):
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if self.agc_mode == "rms":
            level = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        else:  # peak
            level = float(np.max(np.abs(audio)) + 1e-12)

        self.last_rms = 0.97 * self.last_rms + 0.03 * level

        
        if self.auto_gain:
            desired = self.agc_target_rms / max(self.last_rms, 1e-9)
            desired = min(desired, self.agc_max_gain)
            self.agc_gain += (desired - self.agc_gain) * self.agc_smoothing
            gain *= self.agc_gain

        audio *= gain
        audio = np.clip(audio, -limit, limit)
        out = (audio * 32767).astype(np.int16)

        if len(out) >= WAVE_SAMPLES:
            self.last_audio = out[-WAVE_SAMPLES:]
        else:
            keep = WAVE_SAMPLES - len(out)
            self.last_audio = np.concatenate([self.last_audio[-keep:], out])

        self.stream.write(out.tobytes())

    def close(self):
        self.stream.stop()
        self.stream.close()

# ======================
# STT ENGINE
# ======================

from faster_whisper import WhisperModel
import queue

class STTEngine:
    def __init__(self, model="base", device="auto"):
        self.model = WhisperModel(
            model,
            device=device,
            compute_type="float16" if device != "cpu" else "int8"
        )

        self.audio_q = queue.Queue()
        self.recording = False
        self.stream = None

    def start(self):
        if self.stream:
            return

        self.recording = True
        self.audio_q.queue.clear()

        self.stream = sd.InputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=STT_CHANNELS,
            dtype="float32",
            blocksize=STT_BLOCKSIZE,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_q.put(indata.copy())

    def transcribe(self):
        chunks = []
        while not self.audio_q.empty():
            chunks.append(self.audio_q.get())

        if not chunks:
            return ""

        audio = np.concatenate(chunks, axis=0).flatten()
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True
        )

        return " ".join(seg.text.strip() for seg in segments)


# ======================
# STATUS LED
# ======================

class StatusLED(QWidget):
    def __init__(self, diameter=10):
        super().__init__()
        self._color = QColor("#2a2a2a")
        self._diameter = diameter
        self.setFixedSize(diameter, diameter)

    def set_color(self, color_hex):
        self._color = QColor(color_hex)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(0, 0, self._diameter, self._diameter)


# ======================
# WAVE DISPLAY
# ======================

class WaveDisplay(QWidget):
    def __init__(self, player, app_ref):
        super().__init__()
        self.player = player
        self.app_ref = app_ref
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(WAVE_TIMER_MS)

        self._smoothed = np.zeros(WAVE_SAMPLES, dtype=np.float32)
        self._smooth_alpha = 0.25

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        bg = QColor("#141414")
        panel = QColor("#1a1a1a")
        border = QColor("#2e7d4f")
        neon = QColor("#4cff7a")
        dim = QColor("#2f9d57")

        p.fillRect(self.rect(), bg)

        pad = 14
        r = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad)
        p.setPen(QPen(border, 2))
        p.setBrush(panel)
        p.drawRoundedRect(r, 16, 16)

        audio = self.player.last_audio.astype(np.float32) / 32768.0
        self._smoothed = (1 - self._smooth_alpha) * self._smoothed + self._smooth_alpha * audio

        mid_y = r.center().y()
        p.setPen(QPen(dim, 1))
        p.drawLine(int(r.left()) + 10, int(mid_y), int(r.right()) - 10, int(mid_y))

        left, right = r.left() + 10, r.right() - 10
        amp = 0.42 * (r.height() - 20)

        xs = np.linspace(left, right, len(self._smoothed))
        ys = mid_y - (self._smoothed * amp)

        step = max(1, len(xs) // max(600, self.width()))
        p.setPen(QPen(neon, 2))
        for i in range(step, len(xs), step):
            p.drawLine(QPointF(xs[i - step], ys[i - step]), QPointF(xs[i], ys[i]))

        if self.app_ref.debug_enabled:
            p.setPen(neon)
            p.setFont(QFont("", 10))
            y = int(r.top()) + 22
            for line in (
                f"BUSY: {self.app_ref.busy}",
                f"HISTORY: {len(self.app_ref.chat_history)} msgs",
                f"KOBOLD: {self.app_ref.ollama_tps:.1f} tk/s" if self.app_ref.ollama_tps else "KOBOLD: --",
                f"TTS TTFB: {int(self.app_ref.tts_ttfb * 1000)} ms" if self.app_ref.tts_ttfb else "TTS TTFB: --",
            ):
                p.drawText(int(r.left()) + 16, y, line)
                y += 16
# ======================
# TEXT EDIT
# ======================

class SendTextEdit(QTextEdit):
    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.parent_app = parent_app




# ======================
# MAIN APP
# ======================

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ecko v0.2")
        self.resize(860, 700)

        self.settings = QSettings("Ecko", "Ecko-GUI-v0.2")

        self.player = PCMPlayer()
        self.user_gain = 1.5
        self.busy = False
        self.debug_enabled = False

        self.ollama_tps = 0.0
        self.tts_ttfb = 0.0

        self.pipeline_sem = threading.Semaphore(1)
        self.chat_history = []
        
        self.stt = STTEngine(model="base", device="cpu")
        
        self.mic_enabled = False
        self.ptt_enabled = False
        self.ptt_active = False

        self.build_ui()
        
        QApplication.instance().installEventFilter(self)
        
        self.initial_load()
        

    def get_character_dir(self):
        path = os.path.join(os.getcwd(), "characters")
        os.makedirs(path, exist_ok=True)
        return path
    
    def on_character_selected(self, index):
        if index <= 0:
            return

        name = self.char_box.currentText()
        path = os.path.join(self.get_character_dir(), name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.apply_character_data(data)
            self.reset_chat()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
   

    def load_characters(self):
        self.char_box.blockSignals(True)
        self.char_box.clear()
        self.char_box.addItem("— Select Character —")

        char_dir = self.get_character_dir()
        for name in sorted(os.listdir(char_dir)):
            if name.lower().endswith(".json"):
                self.char_box.addItem(name)

        self.char_box.blockSignals(False)    
    def build_prompt(self):
        prompt = ""

        for msg in self.chat_history:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt += f"[SYSTEM]\n{content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += "Assistant:"
        return prompt
    def get_kv_scaling(self):
        if not self.kv_scale_btn.isChecked():
            return None

        try:
            val = float(self.kv_scale_input.text())
            return val
        except ValueError:
            return None
    def toggle_mic(self, enabled):
        self.mic_enabled = enabled

        # Force PTT to follow mic state
        if self.ptt_btn.isChecked() != enabled:
            self.ptt_btn.blockSignals(True)
            self.ptt_btn.setChecked(enabled)
            self.ptt_btn.blockSignals(False)

        self.ptt_enabled = enabled

        if not enabled:
            self.ptt_active = False
            self.stt.stop()
            return

    # Mic ON always requires PTT (temporary behavior)
    # Actual recording starts only when Alt is held


            
    def toggle_ptt(self, enabled):
        # PTT is temporarily forced by Mic toggle
        self.ptt_enabled = self.mic_enabled


        
    def eventFilter(self, obj, event):
        # ---------- ENTER TO SEND ----------
        if (
            event.type() == QEvent.KeyPress
            and obj is self.text_box
            and self.enter_send_btn.isChecked()
            and event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and event.modifiers() == Qt.NoModifier
        ):
            event.accept()
            self.on_send()
            return True

        # ---------- PTT HANDLING ----------
        if self.mic_enabled and self.ptt_enabled:
            if event.type() == QEvent.KeyPress:
                if event.key() == PTT_KEY and not self.ptt_active:
                    self.ptt_active = True
                    self.stt.start()

            elif event.type() == QEvent.KeyRelease:
                if event.key() == PTT_KEY and self.ptt_active:
                    self.ptt_active = False
                    self.stt.stop()
                    self.finish_stt()

        return super().eventFilter(obj, event)



        return False




  
    # ---------- UI ----------

    def build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = QHBoxLayout()
        self.ollama_led = StatusLED()
        self.tts_led = StatusLED()
        
        top.addWidget(self.tts_led)
        top.addWidget(QLabel("Echo-TTS"))
        top.addSpacing(10)
        top.addWidget(self.ollama_led)
        top.addWidget(QLabel("KoboldCPP"))
        top.addStretch(1)

        self.agc_mode_box = QComboBox()
        self.agc_mode_box.addItems(["RMS", "Peak"])
        self.agc_mode_box.setFixedWidth(80)
        top.addWidget(self.agc_mode_box)

        self.agc_mode_box.currentTextChanged.connect(
            lambda t: setattr(self.player, "agc_mode", t.lower())
        )

        self.agc_btn = QPushButton("Gain")
        self.agc_btn.setCheckable(True)
        self.agc_btn.toggled.connect(lambda s: setattr(self.player, "auto_gain", s))
        top.addWidget(self.agc_btn)

        top.addWidget(QLabel("VOL"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(50, 300)
        self.vol_slider.setValue(150)
        self.vol_slider.setFixedWidth(160)
        self.vol_slider.valueChanged.connect(lambda v: setattr(self, "user_gain", v / 100))
        top.addWidget(self.vol_slider)

        self.debug_btn = QPushButton("Debug")
        self.debug_btn.setCheckable(True)
        self.debug_btn.toggled.connect(lambda s: setattr(self, "debug_enabled", s))
        top.addWidget(self.debug_btn)

        self.model_box = QComboBox()
        self.voice_box = QComboBox()
        self.model_box.setFixedWidth(180)
        self.voice_box.setFixedWidth(140)

        self.model_box.currentTextChanged.connect(lambda t: self.settings.setValue("last_model", t))
        self.voice_box.currentTextChanged.connect(lambda t: self.settings.setValue("last_voice", t))

        self.model_box.setEditable(True)
        self.model_box.lineEdit().setReadOnly(True)
        
        top.addWidget(QLabel("Model"))
        top.addWidget(self.model_box)
        top.addWidget(QLabel("Voice"))
        top.addWidget(self.voice_box)

        root.addLayout(top)

        self.wave = WaveDisplay(self.player, self)
        root.addWidget(self.wave, 1)

        self.text_box = SendTextEdit(self)
        self.text_box.setPlaceholderText("Type text to speak…")
        self.text_box.setMinimumHeight(120)
        root.addWidget(self.text_box)
        # ---- KV Scaling controls ----
        self.kv_scale_btn = QPushButton("KV Scale")
        self.kv_scale_btn.setCheckable(True)
        top.addWidget(self.kv_scale_btn)

        self.kv_scale_input = QLineEdit("1.25")
        self.kv_scale_input.setFixedWidth(60)
        self.kv_scale_input.setAlignment(Qt.AlignCenter)
        self.kv_scale_input.setToolTip("speaker_kv_scale")
        top.addWidget(self.kv_scale_input)
        self.kv_scale_input.setEnabled(False)
        self.kv_scale_btn.toggled.connect(self.kv_scale_input.setEnabled)


        # ---- bottom bar (MODIFIED only to add buttons) ----
        bottom = QHBoxLayout()
        self.enter_send_btn = QPushButton("Enter to Send")
        self.enter_send_btn.setCheckable(True)
        bottom.addWidget(self.enter_send_btn)

        bottom.addSpacing(12)

        self.save_char_btn = QPushButton("Save Character")
        self.save_char_btn.clicked.connect(self.save_character)
        bottom.addWidget(self.save_char_btn)

        self.char_box = QComboBox()
        self.char_box.setFixedWidth(200)
        self.char_box.currentIndexChanged.connect(self.on_character_selected)
        bottom.addWidget(QLabel("Character"))
        bottom.addWidget(self.char_box)


        self.reset_chat_btn = QPushButton("Reset Chat")
        self.reset_chat_btn.clicked.connect(self.reset_chat)
        bottom.addWidget(self.reset_chat_btn)

        self.mic_btn = QPushButton("🎤 Mic")
        self.mic_btn.setCheckable(True)
        bottom.addWidget(self.mic_btn)

        self.ptt_btn = QPushButton("PTT")
        self.ptt_btn.setCheckable(True)
        bottom.addWidget(self.ptt_btn)
        
        self.mic_btn.toggled.connect(self.toggle_mic)
        self.ptt_btn.toggled.connect(self.toggle_ptt)

        bottom.addStretch(1)

        self.send_btn = QPushButton("Speak ▶")
        self.send_btn.clicked.connect(self.on_send)
        bottom.addWidget(self.send_btn)
        root.addLayout(bottom)

        self.sys_toggle = QPushButton("System Prompt ▼")
        self.sys_toggle.setCheckable(True)
        self.sys_toggle.toggled.connect(self.toggle_system_prompt)
        root.addWidget(self.sys_toggle)

        self.system_prompt = QTextEdit()
        self.system_prompt.setVisible(False)
        self.system_prompt.setMinimumHeight(80)
        root.addWidget(self.system_prompt)

        self.setStyleSheet("""
            QWidget { background-color: #141414; color: #4cff7a; }
            QTextEdit, QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #2e7d4f;
                border-radius: 10px;
                padding: 6px 8px;
            }
            QPushButton {
                background-color: #1e1e1e;
                border: 1px solid #2e7d4f;
                border-radius: 10px;
                padding: 8px 12px;
            }
            QPushButton:checked {
                background-color: #234b34;
                border: 1px solid #4cff7a;
            }
            QSlider::groove:horizontal {
                border: 1px solid #2e7d4f;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #4cff7a;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4cff7a;
                border: 1px solid #2e7d4f;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)

    def toggle_system_prompt(self, expanded):
        self.system_prompt.setVisible(expanded)
        self.sys_toggle.setText("System Prompt ▲" if expanded else "System Prompt ▼")
    # ---------- LOAD ----------

    def initial_load(self):
        self.load_kobold_status()
        self.load_voices()
        self.load_characters()


        self.vol_slider.setValue(int(self.settings.value("volume", 150)))
        self.user_gain = self.vol_slider.value() / 100
        self.agc_btn.setChecked(self.settings.value("agc", False, type=bool))
        self.debug_btn.setChecked(self.settings.value("debug", False, type=bool))
        self.enter_send_btn.setChecked(self.settings.value("enter_send", False, type=bool))
        self.system_prompt.setPlainText(self.settings.value("system_prompt", ""))

        size = self.settings.value("window_size")
        if size:
            self.resize(size)

    def load_kobold_status(self):
        try:
            r = requests.get(f"{KOBOLD_BASE}/api/v1/model", timeout=3)
            model_name = r.json().get("result", "Unknown model")

            self.model_box.blockSignals(True)
            self.model_box.clear()
            self.model_box.addItem(model_name)
            self.model_box.blockSignals(False)

            self.ollama_led.set_color("#4cff7a")
        except Exception:
            self.ollama_led.set_color("#3a1414")


    def load_voices(self):
        self.voice_box.blockSignals(True)
        self.voice_box.clear()
        try:
            r = requests.get(f"{TTS_BASE}/v1/voices", timeout=5)
            voices = sorted(v["id"] for v in r.json().get("data", []))
            self.voice_box.addItems(voices)
            last = self.settings.value("last_voice", "")
            if last in voices:
                self.voice_box.setCurrentText(last)
            self.tts_led.set_color("#4cff7a")
        except Exception:
            self.tts_led.set_color("#3a1414")
        finally:
            self.voice_box.blockSignals(False)
    # =========================
    # CHARACTER PRESETS (ADDED)
    # =========================

    def get_character_data(self):
        return {
            "system_prompt": self.system_prompt.toPlainText(),
            "voice": self.voice_box.currentText(),
            "volume": self.vol_slider.value(),
            "agc": self.agc_btn.isChecked(),
            "agc_mode": self.agc_mode_box.currentText(),

            # ---- KV scaling ----
            "kv_scale_enabled": self.kv_scale_btn.isChecked(),
            "kv_scale_value": self.kv_scale_input.text(),
        }

    def apply_character_data(self, data):
        if "system_prompt" in data:
            self.system_prompt.setPlainText(data["system_prompt"])

        if "voice" in data:
            idx = self.voice_box.findText(data["voice"])
            if idx != -1:
                self.voice_box.setCurrentIndex(idx)

        if "volume" in data:
            self.vol_slider.setValue(int(data["volume"]))

        if "agc" in data:
            self.agc_btn.setChecked(bool(data["agc"]))
            
        if "agc_mode" in data:
            self.agc_mode_box.setCurrentText(data["agc_mode"])


        # ---- KV scaling ----
        if "kv_scale_enabled" in data:
            self.kv_scale_btn.setChecked(bool(data["kv_scale_enabled"]))

        if "kv_scale_value" in data:
            self.kv_scale_input.setText(str(data["kv_scale_value"]))



    def save_character(self):
        base_dir = os.path.join(os.getcwd(), "characters")
        os.makedirs(base_dir, exist_ok=True)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Character",
            base_dir,
            "Character Preset (*.json)"
        )
        if not path:
            return

        # ---- FORCE .json EXTENSION ----
        if not path.lower().endswith(".json"):
            path += ".json"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_character_data(), f, indent=2)
            self.load_characters()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_character(self):
        base_dir = os.path.join(os.getcwd(), "characters")
        os.makedirs(base_dir, exist_ok=True)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Character",
            base_dir,
            "Character Preset (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.apply_character_data(data)
            self.reset_chat()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def reset_chat(self):
        if QMessageBox.question(
            self,
            "Reset Chat",
            "Clear conversation history?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self.chat_history.clear()
            self.ollama_tps = 0.0
            self.tts_ttfb = 0.0


# ---------- ACTION ----------
    def on_send(self):
        if not self.pipeline_sem.acquire(blocking=False):
            return

        text = self.text_box.toPlainText().strip()
        if not text:
            self.pipeline_sem.release()
            return

        QTimer.singleShot(0, self.text_box.clear)
        threading.Thread(target=self.run_pipeline, args=(text,), daemon=True).start()


    def finish_stt(self):
        text = self.stt.transcribe().strip()
        if not text:
            return

        self.text_box.setLineWrapMode(QTextEdit.NoWrap)

        # Split text into characters for gradual typing
        self._stt_text = text
        self._stt_index = 0

        def type_step():
            if self._stt_index < len(self._stt_text):
                self.text_box.moveCursor(QTextCursor.End)
                self.text_box.insertPlainText(self._stt_text[self._stt_index])
                self._stt_index += 1
                QTimer.singleShot(30, type_step)  # 30 ms per character
            else:
                self.text_box.moveCursor(QTextCursor.End)
                QTimer.singleShot(800, self.on_send)  # send after typing done

        type_step()




    def run_pipeline(self, text):
        self.busy = True
        self.ollama_tps = 0.0
        self.tts_ttfb = 0.0

        try:
            if not self.chat_history:
                sys_text = self.system_prompt.toPlainText().strip()
                if sys_text:
                    self.chat_history.append({"role": "system", "content": sys_text})

            self.chat_history.append({"role": "user", "content": text})

            if len(self.chat_history) > MAX_HISTORY_MESSAGES:
                self.chat_history = self.chat_history[-MAX_HISTORY_MESSAGES:]

            # ===== KOBOLDCPP GENERATION =====
            prompt = self.build_prompt()

            start = time.perf_counter()
            r = requests.post(
                f"{KOBOLD_BASE}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 300,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop_sequence": ["User:"]
                },
                timeout=120
            )

            elapsed = time.perf_counter() - start
            reply = r.json()["results"][0]["text"].strip()
            tokens_est = max(1, len(reply) // 4)  # ~4 chars per token
            self.ollama_tps = tokens_est / max(elapsed, 1e-6)

            # ===== END KOBOLDCPP =====
            self.chat_history.append({"role": "assistant", "content": reply})

            tts_start = time.perf_counter()
            first = True

            # ---- build TTS payload ----
            tts_payload = {
                "input": reply,
                "voice": self.voice_box.currentText(),
                "stream": True
            }

            kv_scale = self.get_kv_scaling()
            if kv_scale is not None:
                tts_payload["extra_body"] = {
                    "speaker_kv_scale": kv_scale,
                    "speaker_kv_min_t": 0.9,
                    "speaker_kv_max_layers": 24
                }

            # ---- TTS request ----
            with requests.post(
                f"{TTS_BASE}/v1/audio/speech",
                json=tts_payload,
                stream=True,
                timeout=120
            ) as resp:
                for chunk in resp.iter_content(4096):
                    if chunk:
                        if first:
                            self.tts_ttfb = time.perf_counter() - tts_start
                            first = False
                        self.player.play(chunk, self.user_gain)

        finally:
            self.busy = False
            self.pipeline_sem.release()


    def save_settings(self):
        self.settings.setValue("volume", self.vol_slider.value())
        self.settings.setValue("agc", self.agc_btn.isChecked())
        self.settings.setValue("debug", self.debug_btn.isChecked())
        self.settings.setValue("enter_send", self.enter_send_btn.isChecked())
        self.settings.setValue("system_prompt", self.system_prompt.toPlainText())
        self.settings.setValue("last_model", self.model_box.currentText())
        self.settings.setValue("last_voice", self.voice_box.currentText())
        self.settings.setValue("window_size", self.size())

    def closeEvent(self, event):
        self.save_settings()
        self.player.close()
        event.accept()


# ======================
# ENTRY
# ======================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())

