from flask import Flask, render_template_string, jsonify
import torch
import sounddevice as sd
import numpy as np
from collections import deque
import librosa
import torch.nn as nn
import csv
from datetime import datetime

# -------------------------------
# FLASK APP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# LOAD MODEL
# -------------------------------
# -------------------------------
# LOAD MODEL (PyTorch 2.6 FIX)
# -------------------------------
import sklearn.preprocessing
from torch.serialization import add_safe_globals

add_safe_globals([sklearn.preprocessing._label.LabelEncoder])

checkpoint = torch.load("model/alm_model.pth", map_location="cpu", weights_only=False)
speech_enc = checkpoint["speech_encoder"]
sound_enc = checkpoint["sound_encoder"]


# -------------------------------
# AUDIO CONFIG
# -------------------------------
sd.default.device = None  # Auto select mic
SR = 16000
DURATION = 3

# -------------------------------
# MEMORY (Last 5 events)
# -------------------------------
memory = deque(maxlen=5)

# -------------------------------
# MODEL DEFINITION
# -------------------------------
class ALMNet(nn.Module):
    def __init__(self, input_size, speech_classes, sound_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.speech_head = nn.Linear(128, speech_classes)
        self.sound_head = nn.Linear(128, sound_classes)

    def forward(self, x):
        z = self.shared(x)
        return self.speech_head(z), self.sound_head(z)

model = ALMNet(40, len(speech_enc.classes_), len(sound_enc.classes_))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# -------------------------------
# CONTEXT ENGINE (FIXED)
# -------------------------------
def context_engine(speech, sound, s_conf, n_conf):
    # Save to memory
    memory.append({
        "speech": speech,
        "sound": sound,
        "s_conf": s_conf,
        "n_conf": n_conf
    })

    # Count recent events
    siren_count = sum(1 for m in memory if m["sound"] == "siren")
    noise_count = sum(1 for m in memory if m["sound"] not in ["silence", "normal"])
    stop_count = sum(1 for m in memory if m["speech"] == "stop")

    # Confidence gate
    if s_conf < 40 or n_conf < 40:
        return "Low confidence — system unsure, please repeat"

    # Pattern reasoning
    if siren_count >= 3:
        return "🚨 Emergency mode: Multiple sirens detected recently"

    if stop_count >= 2 and noise_count >= 2:
        return "🛑 Obstacle alert: Stop command with surrounding noise"

    if speech == "go" and sound in ["silence", "normal"]:
        return "✅ Safe to proceed: Calm environment"

    return "Normal environment"

# -------------------------------
# WEB UI
# -------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>ALM Dashboard</title>
<style>
body { font-family: Arial; background:#111; color:#eee; text-align:center; }
.card { background:#222; padding:20px; margin:20px; border-radius:10px; }
button { padding:15px 30px; font-size:18px; border:none; border-radius:10px; cursor:pointer; }
.bar { height:20px; background:lime; margin-top:10px; transition: width 0.5s; }
</style>
</head>
<body>
<h1>🎤 ALM Live Dashboard</h1>
<button onclick="runALM()">Start Listening</button>

<div class="card">
<h2>Speech</h2>
<p id="speech">---</p>
<div id="speechBar" class="bar" style="width:0%"></div>
</div>

<div class="card">
<h2>Sound</h2>
<p id="sound">---</p>
<div id="soundBar" class="bar" style="width:0%"></div>
</div>

<div class="card">
<h2>Context</h2>
<p id="context">---</p>
</div>

<script>
function runALM() {
  fetch('/run')
    .then(res => res.json())
    .then(data => {
      document.getElementById("speech").innerText = data.speech + " (" + data.speech_conf + "%)";
      document.getElementById("sound").innerText = data.sound + " (" + data.sound_conf + "%)";
      document.getElementById("context").innerText = data.context;

      document.getElementById("speechBar").style.width = data.speech_conf + "%";
      document.getElementById("soundBar").style.width = data.sound_conf + "%";
    });
}
</script>
</body>
</html>
"""

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/run")
def run_alm():
    print("🔥 /run endpoint hit", flush=True)

    print("🎤 Recording audio...", flush=True)
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
    sd.wait()
    audio = audio.flatten()
    print("✅ Recording finished", flush=True)

    features = extract_mfcc(audio)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outS, outN = model(x)

        s_probs = torch.softmax(outS, dim=1)[0]
        n_probs = torch.softmax(outN, dim=1)[0]

        s_idx = torch.argmax(s_probs).item()
        n_idx = torch.argmax(n_probs).item()

        speech = speech_enc.inverse_transform([s_idx])[0]
        sound = sound_enc.inverse_transform([n_idx])[0]

        s_conf = int(s_probs[s_idx].item() * 100)
        n_conf = int(n_probs[n_idx].item() * 100)

    # FIXED CALL
    context = context_engine(speech, sound, s_conf, n_conf)

    # -------------------------------
    # LOGGING SYSTEM
    # -------------------------------
    with open("logs.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            speech,
            sound,
            s_conf,
            n_conf,
            context
        ])

    return jsonify({
        "speech": speech,
        "sound": sound,
        "speech_conf": s_conf,
        "sound_conf": n_conf,
        "context": context
    })

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
