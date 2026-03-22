import cv2
import mediapipe as mp
import math
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def calculate_distance(p1, p2, w, h):
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

def calculate_distance_3d(p1, p2, w, h):
    """Calculates 3D Euclidean distance using MediaPipe's roughly scaled Z coordinate."""
    dx = (p1.x - p2.x) * w
    dy = (p1.y - p2.y) * h
    dz = (p1.z - p2.z) * w  # Z is normalized roughly to image width
    return math.sqrt(dx*dx + dy*dy + dz*dz)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20)
]

# Landmark indices: TIP, PIP (proximal interphalangeal), MCP
FINGER_LANDMARKS = {
    'index':  {'tip': 8,  'pip': 6,  'mcp': 5},
    'middle': {'tip': 12, 'pip': 10, 'mcp': 9},
    'ring':   {'tip': 16, 'pip': 14, 'mcp': 13},
    'pinky':  {'tip': 20, 'pip': 18, 'mcp': 17},
}

def get_hand_orientation(hand, w, h):
    """
    Returns a unit vector pointing from wrist (0) to middle MCP (9).
    This is the hand's 'up' direction regardless of rotation.
    """
    wrist = hand[0]
    mid_mcp = hand[9]
    dx = (mid_mcp.x - wrist.x) * w
    dy = (mid_mcp.y - wrist.y) * h
    length = math.hypot(dx, dy)
    if length == 0:
        return (0, -1)
    return (dx / length, dy / length)

def project_along_axis(point, origin, axis, w, h):
    """
    Projects (point - origin) onto axis.
    Positive value = point is 'ahead' of origin along axis direction.
    """
    dx = (point.x - origin.x) * w
    dy = (point.y - origin.y) * h
    return dx * axis[0] + dy * axis[1]

def is_finger_extended(hand, finger_key, axis, w, h):
    lm = FINGER_LANDMARKS[finger_key]
    wrist = hand[0]
    
    # Method 1: Axis projection (works well at medium/far range)
    tip_proj = project_along_axis(hand[lm['tip']], wrist, axis, w, h)
    pip_proj = project_along_axis(hand[lm['pip']], wrist, axis, w, h)
    proj_extended = (tip_proj - pip_proj) > 5
    
    # Method 2: 3D Distance ratio (works at close range where fingers are foreshortened)
    # Extended finger: 3D MCP-to-TIP > 3D MCP-to-PIP
    d_tip_3d = calculate_distance_3d(hand[lm['mcp']], hand[lm['tip']], w, h)
    d_pip_3d = calculate_distance_3d(hand[lm['mcp']], hand[lm['pip']], w, h)
    dist_extended = d_tip_3d > (d_pip_3d * 1.15)
    
    # Either method detecting extension = finger is extended
    return proj_extended or dist_extended

def is_thumb_extended(hand, axis, w, h):
    perp_axis = (-axis[1], axis[0])  # 90° rotation
    wrist = hand[0]
    
    # Method 1: Lateral projection
    tip_proj = project_along_axis(hand[4], wrist, perp_axis, w, h)
    ip_proj  = project_along_axis(hand[3], wrist, perp_axis, w, h)
    proj_extended = abs(tip_proj - ip_proj) > 5
    
    # Method 2: 3D Distance ratio (CMC=1 to TIP=4 vs CMC=1 to IP=3)
    d_tip_3d = calculate_distance_3d(hand[1], hand[4], w, h)
    d_ip_3d  = calculate_distance_3d(hand[1], hand[3], w, h)
    dist_extended = d_tip_3d > (d_ip_3d * 1.15)
    
    return proj_extended or dist_extended

def check_tripod_grip(hand, w, h):
    """
    Tripod grip = thumb, index, middle tips are extremely close together in 3D space.
    """
    palm_size_3d = calculate_distance_3d(hand[0], hand[9], w, h)
    d_ti = calculate_distance_3d(hand[4], hand[8], w, h)
    d_im = calculate_distance_3d(hand[8], hand[12], w, h)
    d_tm = calculate_distance_3d(hand[4], hand[12], w, h)
    max_grip = max(d_ti, d_im, d_tm)
    return max_grip < (palm_size_3d * 0.45)

class HandDetector:
    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4)
        self.landmarker = HandLandmarker.create_from_options(options)
        self._consec = {}       # {username: int} consecutive suspicious frames
        self._cooldown = {}     # {username: float} last flag time
        self.CONSEC_REQUIRED = 1   # Instant trigger — any single detection fires
        self.COOLDOWN_SEC = 8.0
        
    def process_frame(self, img, username="default"):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        try:
            result = self.landmarker.detect(mp_image)
        except Exception:
            return None
        
        suspicious = False
        
        if result.hand_landmarks:
            h, w, _ = img.shape
            
            for hand in result.hand_landmarks:
                axis = get_hand_orientation(hand, w, h)
                palm_size = calculate_distance(hand[0], hand[9], w, h)

                index_ext  = is_finger_extended(hand, 'index',  axis, w, h)
                middle_ext = is_finger_extended(hand, 'middle', axis, w, h)
                ring_ext   = is_finger_extended(hand, 'ring',   axis, w, h)
                pinky_ext  = is_finger_extended(hand, 'pinky',  axis, w, h)
                thumb_ext  = is_thumb_extended(hand, axis, w, h)

                is_fist = not (index_ext or middle_ext or ring_ext or pinky_ext)

                is_tripod = check_tripod_grip(hand, w, h)
                is_writing = is_tripod

                # Count extended fingers
                ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext, thumb_ext])

                # Suspicious if hand has 1+ extended fingers AND is NOT in writing position AND NOT a fist
                if ext_count >= 1 and not is_writing and not is_fist:
                    suspicious = True
                    break

        current_consec = self._consec.get(username, 0)
        if suspicious:
            self._consec[username] = current_consec + 1
            if self._consec[username] >= self.CONSEC_REQUIRED:
                last_flag = self._cooldown.get(username, 0)
                if time.time() - last_flag > self.COOLDOWN_SEC:
                    self._cooldown[username] = time.time()
                    self._consec[username] = 0  # reset after flagging
                    return "Suspicious Hand Movement"
        else:
            self._consec[username] = max(0, current_consec - 1)

        return None


def main():
    """Standalone visual debug mode — shows skeleton, per-finger status, and detection result."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting Hand Detection Debug Feed... Press 'q' to quit.")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.2,
        min_tracking_confidence=0.2)
    landmarker = HandLandmarker.create_from_options(options)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        status = "NO HAND"
        color = (128, 128, 128)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                # Draw skeleton
                for s, e in HAND_CONNECTIONS:
                    cv2.line(img,
                        (int(hand[s].x * w), int(hand[s].y * h)),
                        (int(hand[e].x * w), int(hand[e].y * h)),
                        (0, 255, 0), 2)
                for i, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)

                axis = get_hand_orientation(hand, w, h)
                palm_size = calculate_distance(hand[0], hand[9], w, h)

                index_ext  = is_finger_extended(hand, 'index',  axis, w, h)
                middle_ext = is_finger_extended(hand, 'middle', axis, w, h)
                ring_ext   = is_finger_extended(hand, 'ring',   axis, w, h)
                pinky_ext  = is_finger_extended(hand, 'pinky',  axis, w, h)
                thumb_ext  = is_thumb_extended(hand, axis, w, h)

                is_fist = not (index_ext or middle_ext or ring_ext or pinky_ext)
                is_tripod = check_tripod_grip(hand, w, h)
                is_writing = is_tripod
                ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext, thumb_ext])

                # Show per-finger status
                finger_str = f"I:{'Y' if index_ext else 'N'} M:{'Y' if middle_ext else 'N'} R:{'Y' if ring_ext else 'N'} P:{'Y' if pinky_ext else 'N'} T:{'Y' if thumb_ext else 'N'}"
                cv2.putText(img, f"Palm:{palm_size:.0f}px  {finger_str}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"Count:{ext_count} Fist:{is_fist} Writing:{is_writing}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if is_writing:
                    status = "WRITING (OK)"
                    color = (0, 255, 255)
                elif is_fist:
                    status = "FIST (OK)"
                    color = (200, 200, 200)
                elif ext_count >= 1:
                    status = f"CHEATING ({ext_count} fingers)"
                    color = (0, 0, 255)
        
        cv2.putText(img, status, (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        cv2.imshow('Hand Detection Debug', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()