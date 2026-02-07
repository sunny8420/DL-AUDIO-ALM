import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    volume = np.linalg.norm(indata)
    print("Mic Level:", volume)

print("🎤 Speak now...")
with sd.InputStream(callback=callback):
    input("Press Enter to stop...")
