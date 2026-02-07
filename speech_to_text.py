import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

print("🎙️ Speak something...")
with mic as source:
    audio = r.listen(source)

try:
    text = r.recognize_google(audio)
    print("You said:", text)
except:
    print("Could not recognize")
