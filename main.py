import flask
from flask import Flask, render_template_string
import speech_recognition as sr
import sounddevice as sd
import numpy as np

from emotion_detector import detect_emotion
from sound_detector import detect_sound

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>DL Audio AI Dashboard</title>
</head>
<body>
    <h1>🎤 DL Audio AI System</h1>
    <p>Click the button and speak</p>
    <form method="post">
        <button type="submit">Start Listening</button>
    </form>
    {% if text %}
        <h2>You Said: {{ text }}</h2>
        <h2>Emotion: {{ emotion }}</h2>
        <h2>Sound: {{ sound }}</h2>
    {% endif %}
</body>
</html>
"""

def get_volume(seconds=1, fs=44100):
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, blocking=True)
    return float(np.linalg.norm(audio))

@app.route("/", methods=["GET", "POST"])
def home():
    text = None
    emotion = None
    sound = None

    if flask.request.method == "POST":
        r = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5, phrase_time_limit=5)

        try:
            text = r.recognize_google(audio)
            emotion = detect_emotion(text)

            volume = get_volume()
            sound = detect_sound(volume)

        except:
            text = "Could not recognize"
            emotion = "N/A"
            sound = "N/A"

    return render_template_string(HTML, text=text, emotion=emotion, sound=sound)

if __name__ == "__main__":
    app.run(debug=True)
