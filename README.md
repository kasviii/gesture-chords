# Gesture-Controlled Chord Player

A real-time computer vision project that uses hand gesture recognition to trigger musical chords. Make a hand gesture in front of your webcam and hear a chord play instantly.


## What This Actually Is (First Attempt)

Built in one day on a completely free stack. No DAW, no MIDI hardware, no custom training.

**What it actually does:**
- Detects 7 hand gestures in real time via webcam using MediaPipe's built-in gesture recognizer
- Plays a corresponding chord through synthesized audio when a gesture is detected
- Draws a colored bounding box around your hand with the chord name and confidence score
- Shows the chord name and vibe label at the bottom of the screen
- Press **R** to record your session (mic only — chord audio capture is a known limitation, fixed in v2)

**Gesture to Chord mapping:**

| Gesture | Chord | Vibe |
|---|---|---|
| Open Palm | C Major | Bright & Open |
| Closed Fist | A Minor | Dark & Moody |
| Peace/Victory | G Major | Hopeful |
| ILoveYou | E Minor | Dramatic |
| Thumb Up | F Major | Warm & Resolved |
| Pointing Up | B Minor | Melancholic |
| Thumb Down | F Minor | Deeply Emotional |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| MediaPipe | Gesture recognition (CNN-based) |
| OpenCV | Webcam feed and visual overlay |
| Pygame | Chord sound synthesis and playback |
| NumPy | Audio signal generation |
| PyAudio | Microphone recording |

**Total cost: $0**

---

## How To Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run:**
```bash
python gesture_chords.py
```

The gesture recognition model downloads automatically on first run.

**3. Controls:**
- Show any of the 7 gestures to play a chord
- Press R to start/stop recording
- Press Q to quit

---

## Known Limitations

- Recording captures mic only, not the chord sounds.
- Chord synthesis is basic sine wave based, not a real instrument sound
- Gesture detection occasionally misfires on similar-looking hand shapes
