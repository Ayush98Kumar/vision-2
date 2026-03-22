"""
Real-Time Phone Detector using OpenCV + YOLOv8
==============================================
Requirements:
    pip install ultralytics opencv-python

Run:
    python phone.py
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO

# ─── Config ────────────────────────────────────────────────
MODEL_PATH   = "yolov8l.pt"   # downloads automatically on first run
CAMERA_INDEX = 0              # 0 = default webcam
CONFIDENCE   = 0.30           # minimum confidence threshold (0.0 - 1.0)
TARGET_CLASS = "cell phone"   # COCO class name to detect

# Bounding box colors (BGR)
COLOR_BOX   = (0, 0, 220)     # red box
COLOR_LABEL = (0, 0, 220)     # red label background
COLOR_TEXT  = (255, 255, 255) # white text
COLOR_SAFE  = (50, 200, 50)   # green status when no phone
COLOR_ALERT = (0, 0, 220)     # red status when phone found
# ────────────────────────────────────────────────────────────


def draw_box(frame, x1, y1, x2, y2, label, confidence):
    """Draw bounding box and label on frame."""
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

    # Label background
    text = f"{label}  {confidence:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), COLOR_LABEL, -1)

    # Label text
    cv2.putText(frame, text, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)


def draw_status(frame, phone_detected, count):
    """Draw status bar at top of frame."""
    h, w = frame.shape[:2]

    if phone_detected:
        status_text  = "  PHONE DETECTED"
        bar_color    = (0, 0, 180)
        status_color = COLOR_ALERT
    else:
        status_text  = "  Scanning..."
        bar_color    = (30, 30, 30)
        status_color = COLOR_SAFE

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 36), bar_color, -1)
    cv2.putText(frame, status_text, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Detection counter (top right)
    counter_text = f"Detections: {count}"
    (cw, _), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, counter_text, (w - cw - 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Bottom hint
    cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "  Press Q to quit", (4, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

class PhoneDetector:
    def __init__(self, model_path='yolov8l.pt'):
        """
        Initializes the YOLOv8-large model for phone detection.
        Requires 'ultralytics' to be installed.
        """
        try:
            self.model = YOLO(model_path)
            # Create a small dummy run to warm up the model
            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy_img, verbose=False, imgsz=320)
            print("[INFO] PhoneDetector YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load PhoneDetector Model: {e}")
            self.model = None
            return

        self.last_flagged = {} # Dictionary to store last flagged time for each username

    def process_frame(self, frame, username="default") -> str | None:
        """
        Processes a BGR image frame and returns an alert string if a phone is detected.
        """
        if self.model is None:
            return None

        # Check cooldown for the current username
        current_time = time.time()
        if username in self.last_flagged and (current_time - self.last_flagged[username]) < 15:
            return None # Still in cooldown period

        # Run inference at reduced resolution for speed (320px instead of full frame)
        results = self.model.predict(frame, conf=0.4, iou=0.45, verbose=False, imgsz=320)[0]
        detected = False
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id].lower()
            if 'phone' in class_name or 'cell' in class_name:
                detected = True
                break

        if detected:
            self.last_flagged[username] = current_time
            return "Phone Detected"

        return None

def main():
    detector = PhoneDetector()
    if detector.model is None:
        return

    print("Opening webcam...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index={CAMERA_INDEX})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    total_detections = 0
    print("Running — press Q in the window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        flag = detector.process_frame(frame)
        phone_found = flag is not None
        if phone_found:
            total_detections += 1

        draw_status(frame, phone_found, total_detections)
        cv2.imshow("Phone Detector — YOLOv8 + OpenCV", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Total detections: {total_detections}")


if __name__ == "__main__":
    main()