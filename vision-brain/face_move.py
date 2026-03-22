import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import Counter
import urllib.request
import os

# ==========================================
# 1. Auto-Download MediaPipe Model
# ==========================================
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )

# ==========================================
# 2. Movement Tracker
# ==========================================
class MovementTracker:
    def __init__(self, flag_threshold=3, time_window=15, stable_frames=4, decay_timeout=10):
        self.history = []
        self.flag_threshold = flag_threshold
        self.time_window = time_window

        self.stable_frames_required = stable_frames
        self.pending_direction = "Center"
        self.pending_frame_count = 0
        self.current_direction = "Center"

        self.decay_timeout = decay_timeout
        self.last_seen_time = {}

        self.cooldown_until = 0
        self.frames_to_show_flag = 0
        self.max_display_frames = 30

    def log_movement(self, direction):
        current_time = time.time()

        self.history = [(t, d) for t, d in self.history if current_time - t <= self.time_window]

        for d in list(self.last_seen_time.keys()):
            if current_time - self.last_seen_time[d] > self.decay_timeout:
                self.history = [(t, hd) for t, hd in self.history if hd != d]
                del self.last_seen_time[d]
                print(f"[{time.strftime('%H:%M:%S')}] FORGIVEN: {d} reset")

        show_cheater = False
        if self.frames_to_show_flag > 0:
            self.frames_to_show_flag -= 1
            show_cheater = True

        if current_time < self.cooldown_until:
            self.pending_direction = "Center"
            self.pending_frame_count = 0
            return "CHEATER" if show_cheater else ""

        if direction == self.pending_direction:
            self.pending_frame_count += 1
        else:
            self.pending_direction = direction
            self.pending_frame_count = 1

        if self.pending_frame_count >= self.stable_frames_required:
            if self.pending_direction != self.current_direction:
                self.current_direction = self.pending_direction
                if self.current_direction != "Center":
                    self.history.append((current_time, self.current_direction))
                    self.last_seen_time[self.current_direction] = current_time
                    count_so_far = sum(1 for _, d in self.history if d == self.current_direction)
                    print(f"[{time.strftime('%H:%M:%S')}] Looked {self.current_direction} "
                          f"({count_so_far}/{self.flag_threshold})")

        direction_counts = Counter(d for _, d in self.history)
        for dir_name, count in direction_counts.items():
            if count >= self.flag_threshold:
                self.frames_to_show_flag = self.max_display_frames
                show_cheater = True
                self.history = []
                self.last_seen_time = {}
                self.current_direction = "Center"
                self.pending_direction = "Center"
                self.pending_frame_count = 0
                self.cooldown_until = current_time + 2.0
                print(f"*** CHEATER DETECTED ({dir_name}) — 2s cooldown ***")
                break

        return "CHEATER" if show_cheater else ""


# ==========================================
# 3. Head Direction Detection
# ==========================================
def get_head_direction_and_box(landmarks, img_w, img_h):
    nose        = landmarks[1]
    left_eye    = landmarks[33]
    right_eye   = landmarks[263]
    forehead    = landmarks[10]
    chin        = landmarks[152]

    eye_center_x = (left_eye.x + right_eye.x) / 2.0
    eye_width    = abs(right_eye.x - left_eye.x)
    if eye_width == 0:
        return "Center", None
    yaw_offset = (nose.x - eye_center_x) / eye_width

    face_height = abs(chin.y - forehead.y)
    if face_height == 0:
        return "Center", None
    chin_ratio = (chin.y - nose.y) / face_height

    left_cheek, right_cheek = landmarks[234], landmarks[454]
    bbox = (
        int(left_cheek.x  * img_w),
        int(forehead.y    * img_h),
        int(right_cheek.x * img_w),
        int(chin.y        * img_h),
    )

    if   yaw_offset >  0.25: return "Left",   bbox   
    elif yaw_offset < -0.25: return "Right",  bbox   
    elif chin_ratio <  0.32: return "Down",   bbox   
    elif chin_ratio >  0.52: return "Up",     bbox   
    else:                    return "Center", bbox

# ==========================================
# FaceMoveDetector Class (for app.py)
# ==========================================
class FaceMoveDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.trackers = {}

    def process_frame(self, image, username="default"):
        if username not in self.trackers:
            # stable_frames=3 (~0.75s at 4 FPS) / flag_threshold=3 (3 sustained glances within 15s)
            self.trackers[username] = MovementTracker(flag_threshold=3, time_window=15, stable_frames=3, decay_timeout=10)
        img_h, img_w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = self.detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            direction, _ = get_head_direction_and_box(landmarks, img_w, img_h)
            status_text = self.trackers[username].log_movement(direction)
            if status_text == "CHEATER":
                return "Suspicious Head Movement"
        return None

# ==========================================
# 4. Main Loop
# ==========================================
def main():
    detector = FaceMoveDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Proctor system running. Press Q to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        flag = detector.process_frame(image)
        
        if flag:
            img_h, img_w, _ = image.shape
            text_size = cv2.getTextSize("CHEATER!", cv2.FONT_HERSHEY_DUPLEX, 2.5, 6)[0]
            text_x = (img_w - text_size[0]) // 2
            text_y = (img_h + text_size[1]) // 2
            cv2.putText(image, "CHEATER!", (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 255), 6)

        image = cv2.flip(image, 1)
        cv2.imshow('AI Exam Proctor', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
