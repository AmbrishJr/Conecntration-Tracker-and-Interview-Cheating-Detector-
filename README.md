# Concentration Tracker 

A real-time concentration tracking system built using **MediaPipe** and **OpenCV**. This tool evaluates a user's attentiveness based on eye blinks, gaze direction, and head pose — ideal for applications like study monitoring, e-learning, or productivity enhancement!.

## Features

- **Eye Blink Detection**  
  Calculates Eye Aspect Ratio (EAR) to detect blinks and periods of eye closure.

- **Gaze Detection**  
  Estimates if the user is looking straight or away using iris landmarks.

- **Head Pose Estimation**  
  Evaluates user orientation based on nose position relative to screen center.

- **Concentration Score**  
  Computes a weighted score combining gaze, head pose, and blinking behavior.

- **Live Visual Feedback**  
  Real-time UI overlay on webcam feed showing concentration level, blink status, and distraction counter.

- **Full Face Mesh Overlay**  
  A live 468-point FaceMesh wireframe is drawn over the detected face so you can
  see exactly what landmarks the tracker is using.

- **Distraction Tracking**  
  Counts how many frames the user is not paying attention and issues warnings if needed.

## Sample Output

The video feed displays:
- A full face mesh wireframe drawn over the detected face
- A concentration percentage bar
- Blink detection alerts
- Distraction count
- ACTIVE / DISTRACTED indicator
- FPS counter

## Tech Stack

- Python 3.x
- OpenCV
- MediaPipe (FaceMesh)
- NumPy

## How It Works

1. **Face landmarks** are detected using MediaPipe FaceMesh (468 landmarks per face
   and rendered as a live wireframe overlay).
2. **EAR (Eye Aspect Ratio)** is used to detect blinks.
3. **Iris position** is used to assess gaze direction.
4. **Nose position** is used to infer head pose.
5. A **composite concentration score** is calculated as: score = 0.4 * gaze + 0.4 * head_pose + 0.2 * (not blinking)
6. A **visual feedback system** shows user concentration in real time.

## Run the Project

```bash
git clone https://github.com/AmbrishJr/Concentration-Tracker-Project-.git
cd Concentration-Tracker-Project-

# recommended: use a virtual environment
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python concen_tracker.py
```

Press `q` while the window is focused to quit.

> **Compatibility:** MediaPipe removed the legacy `mp.solutions` API in newer
> versions (0.10.18+). All scripts go through `mp_face.py`, a shim that draws the
> full face mesh on both current MediaPipe (auto-downloads `models/face_landmarker.task`
> on first run; uses a vendored copy of the mesh topology so the overlay keeps
> working after upgrades) and older versions that still ship `mp.solutions`.

## Interview Integrity Checker

Detects likely cheating during an interview by analyzing head pose, gaze
direction, eye movements, and blink behavior from a single webcam.

```bash
python interview_integrity.py
```

This performs a short ~15s calibration (look straight at the camera; press
ESC to skip), then tracks the session live and writes an integrity report to
`interview_report.txt` when you exit with `q` or ESC. The same face mesh wireframe
is drawn during tracking. Run a quick sanity check of the detection math with:

```bash
python interview_integrity.py --selftest
```

### What it flags

- Sustained looking left/right (side monitor or notes) and looking down (desk/phone)
- Rapid left-right eye scanning while the head stays forward (possible script/autocue)
- Face leaving the frame for 3+ seconds
- A second person appearing in the frame
- Elevated blink rate vs. calibration baseline (soft signal)

### Accuracy notes

Single-webcam gaze estimation is inherently approximate. This detector relies
on a per-user calibration baseline, temporal smoothing, and dwell-time
thresholds so brief natural glances are ignored and only sustained behavior is
flagged. Treat the integrity score as a screening signal, not proof of
misconduct. Tune the `CONFIG` thresholds at the top of the script after testing
in your own environment.

