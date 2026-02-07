def detect_emotion(text):
    if "help" in text.lower():
        return "🚨 EMERGENCY"
    elif "happy" in text.lower():
        return "😊 HAPPY"
    else:
        return "😐 NORMAL"
