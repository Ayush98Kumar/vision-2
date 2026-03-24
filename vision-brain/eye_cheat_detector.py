"""
Eye Cheat Detector  (Asymmetric Direction Fix)
===============================================
Python 3.11 + MediaPipe 0.10.14

Install:
    pip install mediapipe opencv-python numpy

Run:
    python eye_cheat_detector.py

Fix: Separate thresholds for looking LEFT vs RIGHT,
     because eyes naturally move more in one direction.
     Downward gaze (writing) is still allowed.
     Press C to run live calibration and auto-set thresholds.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
LOOK_AWAY_THRESHOLD_SEC = 1.0

LOOK_LEFT_THRESHOLD  = 0.13   # iris x < (0.5 - this)  → looking left
LOOK_RIGHT_THRESHOLD = 0.13   # iris x > (0.5 + this)  → looking right  ← looser
UPWARD_THRESHOLD     = 0.18   # iris y < (0.5 - this)  → looking up (downward allowed)

WINDOW_TITLE = "Eye Cheat Detector  |  C=Calibrate  Q=Quit"
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────

LEFT_IRIS  = [474, 475, 476, 477]
LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249,
              263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_EYE  = [33,   7, 163, 144, 145, 153, 154, 155,
              133, 173, 157, 158, 159, 160, 161, 246]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def get_iris_position(lm, iris_ids, eye_ids, w, h):
    iris_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in iris_ids])
    cx, cy   = iris_pts.mean(axis=0)
    eye_pts  = np.array([(lm[i].x * w, lm[i].y * h) for i in eye_ids])
    x_min, y_min = eye_pts.min(axis=0)
    x_max, y_max = eye_pts.max(axis=0)
    ew = x_max - x_min + 1e-6
    eh = y_max - y_min + 1e-6
    return (cx - x_min) / ew, (cy - y_min) / eh

def is_looking_away(rx, ry):
    looking_left  = (0.5 - rx) > LOOK_LEFT_THRESHOLD
    looking_right = (rx - 0.5) > LOOK_RIGHT_THRESHOLD
    looking_up    = (0.5 - ry) > UPWARD_THRESHOLD
    return looking_left or looking_right or looking_up

def draw_eye(frame, lm, iris_ids, eye_ids, w, h, color):
    ipts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in iris_ids])
    cx, cy = ipts.mean(axis=0).astype(int)
    r = max(int(np.linalg.norm(ipts[0] - ipts[2]) / 2), 5)
    cv2.circle(frame, (cx, cy), r, color, 2)
    cv2.circle(frame, (cx, cy), 2, color, -1)
    epts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_ids])
    cv2.polylines(frame, [cv2.convexHull(epts)], True, color, 1)

def draw_direction_indicator(frame, rx, ry, x_offset, y_offset, label):
    bw, bh = 80, 60
    cx, cy = x_offset + bw // 2, y_offset + bh // 2
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + bw, y_offset + bh), (50, 50, 50), -1)
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + bw, y_offset + bh), (120, 120, 120), 1)

    dot_x = int(x_offset + rx * bw)
    dot_y = int(y_offset + ry * bh)
    dot_x = max(x_offset + 4, min(x_offset + bw - 4, dot_x))
    dot_y = max(y_offset + 4, min(y_offset + bh - 4, dot_y))

    away = is_looking_away(rx, ry)
    dot_color = (0, 0, 255) if away else (0, 255, 100)
    cv2.circle(frame, (dot_x, dot_y), 5, dot_color, -1)

    cv2.line(frame, (cx, y_offset), (cx, y_offset + bh), (80, 80, 80), 1)
    cv2.line(frame, (x_offset, cy), (x_offset + bw, cy), (80, 80, 80), 1)

    cv2.putText(frame, label, (x_offset, y_offset - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

class EyeCheatDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[INFO] Downloading {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.look_away_starts = {}

    def process_frame(self, frame, username="default"):
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        res = self.landmarker.detect(mp_image)

        looking_away = True
        
        if res.face_landmarks:
            lm = res.face_landmarks[0]

            lx, ly = get_iris_position(lm, LEFT_IRIS,  LEFT_EYE,  w, h)
            rx, ry = get_iris_position(lm, RIGHT_IRIS, RIGHT_EYE, w, h)

            left_away  = is_looking_away(lx, ly)
            right_away = is_looking_away(rx, ry)
            looking_away = left_away or right_away

        now = time.time()
        if looking_away:
            if username not in self.look_away_starts:
                self.look_away_starts[username] = now
            elapsed = now - self.look_away_starts[username]
            
            if elapsed >= LOOK_AWAY_THRESHOLD_SEC:
                self.look_away_starts[username] = now
                return "Eyes Off Screen"
        else:
            if username in self.look_away_starts:
                del self.look_away_starts[username]
            
        return None

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("=" * 60)
    print("  EYE CHEAT DETECTOR  (Asymmetric Direction Fix)")
    print(f"  Alert fires after {LOOK_AWAY_THRESHOLD_SEC}s  |  Downward gaze allowed")
    print("  Press C to calibrate (Not supported in refactored test mode yet) |  Press Q to quit")
    print("=" * 60)

    detector = EyeCheatDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        
        flag = detector.process_frame(frame)
        if flag:
            print("\n" + "=" * 60)
            print("       !!!  CHEATING DETECTED  !!!")
            print(f"       Eyes off screen for {LOOK_AWAY_THRESHOLD_SEC} seconds")
            print("=" * 60 + "\n")

        # For visual testing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res   = detector.landmarker.detect(mp_image)
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            lx, ly = get_iris_position(lm, LEFT_IRIS,  LEFT_EYE,  w, h)
            rx, ry = get_iris_position(lm, RIGHT_IRIS, RIGHT_EYE, w, h)
            draw_eye(frame, lm, LEFT_IRIS,  LEFT_EYE,  w, h, (0, 255, 0))
            draw_eye(frame, lm, RIGHT_IRIS, RIGHT_EYE, w, h, (0, 255, 0))
            draw_direction_indicator(frame, lx, ly, 10,      h - 130, "L eye")
            draw_direction_indicator(frame, rx, ry, 10 + 95, h - 130, "R eye")

        if detector.look_away_start:
            elapsed = time.time() - detector.look_away_start
            fill = int(w * min(elapsed / LOOK_AWAY_THRESHOLD_SEC, 1.0))
            cv2.rectangle(frame, (0, 40), (fill, 52), (0, 0, 255), -1)
        else:
            cv2.rectangle(frame, (0, 40), (w, 52), (0, 80, 0), -1)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
