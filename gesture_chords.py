import cv2
import mediapipe as mp
import numpy as np
import pygame
import urllib.request
import os
import time
import wave
import pyaudio


# ── Download model ───────────────────────────────────────────
model_path = "../audio-control-system/hand_landmarker.task"
if not os.path.exists(model_path):
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading hand tracking model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            model_path
        )

# ── Audio setup ───────────────────────────────────────────────
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
SAMPLE_RATE = 44100

def generate_tone(frequency, duration=1.5, volume=0.3, wave="sine"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    if wave == "sine":
        tone = np.sin(2 * np.pi * frequency * t)
    elif wave == "soft":
        tone = np.sin(2 * np.pi * frequency * t) * 0.7
        tone += np.sin(4 * np.pi * frequency * t) * 0.2
        tone += np.sin(6 * np.pi * frequency * t) * 0.1
    # Fade in and out
    fade = int(SAMPLE_RATE * 0.1)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    tone = (tone * volume * 32767).astype(np.int16)
    return tone

def make_chord(frequencies, duration=1.5, volume=0.25):
    chord = np.zeros(int(SAMPLE_RATE * duration))
    for freq in frequencies:
        chord += generate_tone(freq, duration, volume / len(frequencies), "soft")
    chord = np.clip(chord, -32767, 32767).astype(np.int16)
    stereo = np.column_stack((chord, chord))
    sound = pygame.sndarray.make_sound(stereo)
    return sound

# ── Chord definitions (frequencies in Hz) ────────────────────
CHORDS = {
    "Open_Palm": {
        "name": "C Major",
        "notes": [261.63, 329.63, 392.00],  # C E G
        "color": (0, 255, 150),
        "vibe": "Bright & Open"
    },
    "Closed_Fist": {
        "name": "A Minor",
        "notes": [220.00, 261.63, 329.63],  # A C E
        "color": (100, 100, 255),
        "vibe": "Dark & Moody"
    },
    "Victory": {
        "name": "G Major",
        "notes": [196.00, 246.94, 293.66],  # G B D
        "color": (255, 220, 0),
        "vibe": "Hopeful"
    },
    "ILoveYou": {
        "name": "E Minor",
        "notes": [164.81, 196.00, 246.94],  # E G B
        "color": (255, 80, 80),
        "vibe": "Dramatic"
    },
    "Thumb_Up": {
        "name": "F Major",
        "notes": [174.61, 220.00, 261.63],  # F A C
        "color": (255, 160, 50),
        "vibe": "Warm & Resolved"
    },
    "Pointing_Up": {
        "name": "B Minor",
        "notes": [246.94, 293.66, 369.99],  # B D F#
        "color": (180, 100, 255),
        "vibe": "Melancholic"
    },
    "Thumb_Down": {
        "name": "F Minor",
        "notes": [174.61, 207.65, 261.63],  # F Ab C
        "color": (80, 180, 255),
        "vibe": "Deeply Emotional"
    },
}

# Pre-generate all chord sounds
print("Generating chord sounds...")
chord_sounds = {}
for gesture, data in CHORDS.items():
    chord_sounds[gesture] = make_chord(data["notes"])
    print(f"  ✓ {data['name']} ready")

# ── MediaPipe setup ───────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download gesture model
gesture_model_path = "gesture_recognizer.task"
if not os.path.exists(gesture_model_path):
    print("Downloading gesture recognition model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        gesture_model_path
    )
    print("Done!")

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ── State ─────────────────────────────────────────────────────
last_gesture = None
last_play_time = 0
COOLDOWN = 1.0  # seconds between triggers

# ── Main loop ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("\n🖐️  Gesture Chords ready!")
print("✋ Open Palm     = C Major")
print("✊ Fist          = A Minor")
print("✌️  Peace/Victory = G Major")
print("🤟 ILoveYou      = E Minor")
print("👍 Thumbs Up     = F Major")
print("👆 Pointing Up   = B Minor")
print("👎 Thumb Down    = F Minor")
print("\nPress Q to quit.")

current_chord_info = None
# ── Recording state ───────────────────────────────────────────
# ── Recording state ───────────────────────────────────────────
is_recording = False
recorded_frames = []
chord_frames = []
pa = pyaudio.PyAudio()
mic_stream = pa.open(format=pyaudio.paInt16, channels=1,
                     rate=44100, input=True,
                     frames_per_buffer=1024)

with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        if result.gestures and result.hand_landmarks:
            gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score
            hand = result.hand_landmarks[0]

            # Get bounding box
            xs = [lm.x * w for lm in hand]
            ys = [lm.y * h for lm in hand]
            x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20
            x2, y2 = int(max(xs)) + 20, int(max(ys)) + 20

            if gesture in CHORDS:
                chord_data = CHORDS[gesture]
                color = chord_data["color"]
                current_chord_info = chord_data

                # Play chord if new gesture or cooldown passed
                now = time.time()
                if gesture != last_gesture or (now - last_play_time) > COOLDOWN:
                    chord_sounds[gesture].play()
                    last_play_time = now
                    last_gesture = gesture

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Chord label above box
                label = f"{chord_data['name']} ({confidence:.2f})"
                cv2.rectangle(frame, (x1, y1-30), (x1+len(label)*11, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

                # Draw hand dots
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, color, -1)

            else:
                # Unknown gesture
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150,150,150), 2)
                cv2.putText(frame, gesture, (x1, y1-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 2)
                last_gesture = None

        else:
            last_gesture = None
            current_chord_info = None

        # Bottom info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-80), (w, h), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Recording indicator
        if is_recording:
            cv2.circle(frame, (w-30, 30), 12, (0,0,255), -1)
            cv2.putText(frame, "REC", (w-60, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if current_chord_info:
            cv2.putText(frame, f"♪  {current_chord_info['name']}",
                       (15, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                       current_chord_info["color"], 2)
            cv2.putText(frame, current_chord_info["vibe"],
                       (15, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       current_chord_info["color"], 1)
        else:
            cv2.putText(frame, "Show a hand gesture to play a chord",
                       (15, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (150,150,150), 1)

        cv2.imshow("Gesture Chords", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not is_recording:
                is_recording = True
                recorded_frames = []
                print("🔴 Recording started!")
            else:
                is_recording = False
                # Save wav file
                filename = f"recording_{int(time.time())}.wav"
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(44100)
                    wf.writeframes(b''.join(recorded_frames))
                print(f"✅ Saved: {filename}")

        # Capture mic if recording
        if is_recording:
            try:
                mic_data = mic_stream.read(1024, exception_on_overflow=False)
                recorded_frames.append(mic_data)
            except:
                pass

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()