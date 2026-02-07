import os
import librosa
import numpy as np

OUT = "model/features"
os.makedirs(OUT, exist_ok=True)

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def process_folder(folder, tag):
    X, y = [], []
    total = 0

    print(f"\nProcessing {folder}...")

    for root, dirs, files in os.walk(folder):
        for file in files:
            # Handles .wav, .WAV, .Wave etc
            if file.lower().endswith(".wav"):
                path = os.path.join(root, file)
                try:
                    feat = extract_mfcc(path)
                    label = os.path.basename(root)
                    X.append(feat)
                    y.append(label)
                    total += 1
                except Exception as e:
                    print("Skip:", path)

    print(f"Total samples for {tag}: {total}")
    np.save(f"{OUT}/{tag}_X.npy", np.array(X))
    np.save(f"{OUT}/{tag}_y.npy", np.array(y))

process_folder("dataset/speech", "speech")
process_folder("dataset/sound", "sound")
