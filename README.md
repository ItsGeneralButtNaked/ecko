# ECKO

ECKO is a lightweight desktop interface for:

* **Echo-TTS** streaming speech synthesis
* **KoboldCPP** text generation

## Why did you make this?

Ive been messing around with LLMs and TTS for a short while. Echo got released and there wasn't anything else similar available. All the other solutions I tried only connect to the wav output, which on my level of hardware means a grim wait for the audio to generate. Additionally, they come with so many other features that are not required for basic turn-based TTS interaction.

So I just built my own! As you can see in the demo videos it is blazing fast without any performance tweaks even on archaic hardware.

## Features

* Live streaming TTS playback
* Per-character presets (system prompt, voice, volume, AGC, KV scaling)
* RMS or Peak Auto Gain Control
* Real-time waveform visualization
* Persistent settings

---

## Requirements

* **Python 3.11**
* A running **KoboldCPP** server
  [https://github.com/LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp)
* A running **Echo-TTS-API** server
  [https://github.com/KevinAHM/echo-tts-api](https://github.com/KevinAHM/echo-tts-api)

---

## Server Setup

This application requires **two local servers** to be running before launch:

* **KoboldCPP** (LLM text generation)
* **Echo-TTS-API** (streaming text-to-speech)

The GUI connects to these services using the following default endpoints:

```python
KOBOLD_BASE = "http://localhost:5001"
TTS_BASE    = "http://localhost:8000"
```

You can change these in the script if your servers run on different hosts or ports.

---

## Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ItsGeneralButtNaked/echo-tts-gui-kobold.git
cd echo-tts-gui-kobold
```

### 2️⃣ Create a virtual environment

#### Option A — Conda

```bash
conda create -n echo-tts-gui python=3.11
conda activate echo-tts-gui
```

#### Option B — Standard Python venv

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python echo-tts-gui-kobold.py
```

---

## Linux Notes

You may need **PortAudio** and XCB cursor support:

```bash
sudo apt install portaudio19-dev libxcb-cursor0
```

---

## Extra Waffle 🍩

I also have an **Ollama-based version** that works, but it’s currently behind on a few features.
This isn’t meant to be a deep or comprehensive tool — it’s just a quick and easy way to play around with the amazing **Echo-TTS**.

[https://github.com/jordandare/echo-tts](https://github.com/jordandare/echo-tts)

---

## Features / Updates

1. I’m not fully happy with the Auto Gain yet — it definitely needs some tweaks, but it’s useful to have.
2. KV scale could use more exposed values (possibly an **Advanced** tab).
3. General UI cleanup.
4. Speech to text

---

## Platform Support

* **Linux:** Tested and supported
* **Windows:** ❗ *Testing coming soon*

Windows support code may exist, but the application has **not yet been tested on Windows**.
Proper validation and cleanup will be done once Windows testing is completed.
