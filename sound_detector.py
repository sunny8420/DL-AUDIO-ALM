def detect_sound(volume):
    if volume > 0.8:
        return "🚨 VERY LOUD SOUND (Possible Emergency)"
    elif volume > 0.3:
        return "⚠️ Medium Noise"
    else:
        return "😐 Silence / Normal"
