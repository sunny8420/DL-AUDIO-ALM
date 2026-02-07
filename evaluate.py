import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# -------------------------------
# LOAD LOGS
# -------------------------------
df = pd.read_csv("logs.csv")

# -------------------------------
# BASIC CHECK
# -------------------------------
required_cols = ["true_speech", "true_sound", "speech", "sound"]
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing column: {col}")
        print("Make sure logs.csv header is:")
        print("timestamp,speech,sound,speech_conf,sound_conf,context,true_speech,true_sound")
        exit()

# Drop rows where true labels are empty
df = df.dropna(subset=["true_speech", "true_sound"])

# -------------------------------
# ACCURACY
# -------------------------------
speech_acc = accuracy_score(df["true_speech"], df["speech"])
sound_acc = accuracy_score(df["true_sound"], df["sound"])

print("\n🎯 MODEL PERFORMANCE")
print("Speech Accuracy:", round(speech_acc * 100, 2), "%")
print("Sound Accuracy:", round(sound_acc * 100, 2), "%")

# -------------------------------
# CONFUSION MATRICES
# -------------------------------
speech_labels = sorted(df["true_speech"].unique())
sound_labels = sorted(df["true_sound"].unique())

speech_cm = confusion_matrix(
    df["true_speech"], df["speech"], labels=speech_labels
)

sound_cm = confusion_matrix(
    df["true_sound"], df["sound"], labels=sound_labels
)

# -------------------------------
# PLOT & SAVE
# -------------------------------
plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(speech_cm, display_labels=speech_labels).plot(cmap="Blues")
plt.title("Speech Confusion Matrix")
plt.savefig("speech_confusion_matrix.png")

plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(sound_cm, display_labels=sound_labels).plot(cmap="Greens")
plt.title("Sound Confusion Matrix")
plt.savefig("sound_confusion_matrix.png")

print("\n📊 FILES GENERATED:")
print(" - speech_confusion_matrix.png")
print(" - sound_confusion_matrix.png")
