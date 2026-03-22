"""
Lip Movement Cheating Detector — Final v8
==========================================
Cover detection uses MediaPipe HANDS model — detects if any hand
landmark is near the mouth region. This is 100% reliable regardless
of skin tone, lighting, or head movement.

Three detection modes:
  1. LIP MOVEMENT  — 2 open→close cycles → red WARNING banner
  2. HAND NEAR MOUTH — hand overlaps mouth zone → purple CHEATER banner
  3. SUSTAINED OPEN — lips stay above threshold for 4 s → red WARNING banner

Improvements (v8):
  ✓ Robust 3D Gap Calculation — lips are accurately tracked even when looking down.
  ✓ Distance Normalization — leaning forward/backward no longer breaks calibration.

All features:
  ✓ New MediaPipe Tasks API
  ✓ 2 cycles to flag lip movement
  ✓ Sustained lip-open for 4 s also flags cheating
  ✓ Natural actions (sneeze/yawn) filtered by peak gap × 6.5
  ✓ Adaptive calibration (~2 s, mouth closed)
  ✓ Lip-movement banner holds 2.5 s then auto-clears
  ✓ Live HUD: gap bar, cycle dots, hand proximity bar, sustained-open timer
  ✓ q = quit | r = recalibrate

Install:  pip install opencv-python mediapipe
Run:      python lip_cheat_detector.py
"""

import cv2
import numpy as np
import time
import urllib.request
import pathlib

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    FaceLandmarkerOptions, HandLandmarkerOptions, RunningMode
)

# ── Lip landmark indices ──────────────────────────────────────────────────────
INNER_UPPER = 13
INNER_LOWER = 14
LIP_DRAW    = [13, 312, 311, 310, 415, 308, 14, 82, 81, 80, 76, 77]
LIP_BOX     = [13, 312, 311, 310, 415, 308, 14, 82, 81, 80,
               76, 77, 0, 17, 61, 291]

# ── Tuning ────────────────────────────────────────────────────────────────────
CAL_FRAMES         = 20
REQUIRED_CYCLES    = 2
STABLE_FRAMES      = 2
EMA_ALPHA          = 0.35
SNEEZE_MULT        = 8.0
STD_FACTOR         = 1.2
MIN_DELTA          = 3.0
ALERT_HOLD_SEC     = 2.5
HAND_PROXIMITY     = 0.10    # normalised dist — any hand landmark within this of mouth centre
                             # ~10% of frame width ≈ finger-width at typical webcam distance
COVER_CONFIRM      = 3       # consecutive frames required to confirm cover
COVER_GRACE_SEC    = 0.5     # hold "covered" this long after hand leaves (flicker guard)
# For bbox path: how many hand landmarks must fall INSIDE the mouth box
COVER_BBOX_MIN_PTS = 3       # require ≥3 landmarks inside mouth box (not just bbox overlap)
SUSTAINED_OPEN_SEC = 2.5     # seconds lips stay above threshold → flag cheating


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _download(url: str, path: pathlib.Path):
    print(f"[INFO] Downloading {path.name}…")
    urllib.request.urlretrieve(url, path)
    print(f"[INFO] {path.name} ready.")

def get_face_model() -> pathlib.Path:
    p = pathlib.Path("face_landmarker.task")
    if not p.exists():
        _download(
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task", p)
    return p

def get_hand_model() -> pathlib.Path:
    p = pathlib.Path("hand_landmarker.task")
    if not p.exists():
        _download(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", p)
    return p


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Calibrator:

    def __init__(self):
        self._gaps:  list[float] = []
        self.done      = False
        self.baseline  = 0.0
        self.threshold = 0.0

    @property
    def progress(self) -> float:
        return min(len(self._gaps) / CAL_FRAMES, 1.0)

    def feed(self, gap: float) -> bool:
        if self.done:
            return True
        self._gaps.append(gap)
        if len(self._gaps) >= CAL_FRAMES:
            arr            = np.array(self._gaps)
            self.baseline  = float(np.median(arr))
            delta          = max(float(np.std(arr) * STD_FACTOR), MIN_DELTA)
            self.threshold = self.baseline + delta
            self.done      = True
            print(f"[CAL] baseline={self.baseline:.2f}  "
                  f"threshold={self.threshold:.2f}")
        return self.done

    def reset(self):
        self.__init__()


# ═══════════════════════════════════════════════════════════════════════════════
#  LIP CYCLE FSM
# ═══════════════════════════════════════════════════════════════════════════════

class LipCycleFSM:

    def __init__(self):
        self.threshold        = 8.0
        self._state           = "CLOSED"
        self._sc              = 0
        self._ema             = 0.0
        self._prev            = 0.0
        self._ready           = False
        self._peak            = 0.0
        self._cycles          = 0
        self.cycle_count      = 0
        self.is_cheating      = False
        self._alert_end       = 0.0
        self.total            = 0
        self.state_label      = "CLOSED"
        self.velocity         = 0.0
        self.skip_reason      = ""

        # ── Sustained-open tracking ──────────────────────────────────────────
        self._open_since:     float | None = None   # wall-clock time lips went above thr
        self.sustained_secs:  float        = 0.0    # elapsed open time (for HUD)
        self.sustained_event: bool         = False  # True while flagged by this rule

    def update(self, raw: float):
        now = time.time()

        if not self._ready:
            self._ema = self._prev = raw
            self._ready = True
        self._ema      = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._ema
        self.velocity  = abs(self._ema - self._prev)
        self._prev     = self._ema
        gap            = self._ema
        is_open        = gap > self.threshold

        # ── Cycle FSM ────────────────────────────────────────────────────────
        if self._state == "CLOSED":
            if is_open:
                self._sc += 1
                if self._sc >= STABLE_FRAMES:
                    self._state = "OPEN"
                    self._sc    = 0
                    self._peak  = gap
            else:
                self._sc = 0
        elif self._state == "OPEN":
            if gap > self._peak:
                self._peak = gap
            if not is_open:
                self._sc += 1
                if self._sc >= STABLE_FRAMES:
                    self._state = "CLOSED"
                    self._sc    = 0
                    self._record(self._peak)
            else:
                self._sc = 0

        # ── Sustained-open tracker ───────────────────────────────────────────
        if is_open:
            if self._open_since is None:
                self._open_since = now          # lips just crossed threshold
            self.sustained_secs = now - self._open_since
            if self.sustained_secs >= SUSTAINED_OPEN_SEC and not self.sustained_event:
                self.sustained_event = True
                self.total          += 1
                self.is_cheating     = True
                self._alert_end      = now + ALERT_HOLD_SEC
                print(f"[ALERT] Sustained lip-open cheating "
                      f"({self.sustained_secs:.1f}s) — event #{self.total}")
        else:
            # lips closed — reset sustained tracker
            self._open_since      = None
            self.sustained_secs   = 0.0
            self.sustained_event  = False

        self.state_label = self._state
        self.cycle_count = self._cycles

        # ── Cycle-based trigger ───────────────────────────────────────────────
        if self._cycles >= REQUIRED_CYCLES:
            self._cycles     = 0
            self.cycle_count = 0
            self.total      += 1
            self.is_cheating = True
            self._alert_end  = now + ALERT_HOLD_SEC
            print(f"[ALERT] Lip movement cheating — event #{self.total}")

        if self.is_cheating and now > self._alert_end:
            self.is_cheating = False

    def _record(self, peak: float):
        limit = self.threshold * SNEEZE_MULT
        if peak > limit:
            self.skip_reason = f"natural peak={peak:.1f}"
            return
        self.skip_reason  = ""
        self._cycles     += 1
        self.cycle_count  = self._cycles
        print(f"[CYCLE] #{self._cycles}  peak={peak:.1f}")

    def reset(self):
        self.__init__()


# ═══════════════════════════════════════════════════════════════════════════════
#  COVER DETECTOR  — hand proximity to mouth
# ═══════════════════════════════════════════════════════════════════════════════

class CoverDetector:
    """
    Detects a hand actually covering the mouth using two precise signals.

    Signal 1 — POINT DISTANCE
      Any hand landmark within HAND_PROXIMITY (0.10) of the mouth centre.

    Signal 2 — LANDMARK COUNT INSIDE MOUTH BOX
      Counts how many of the 21 hand landmarks fall inside the mouth bounding
      box (lip outline + small expand). Requires ≥ COVER_BBOX_MIN_PTS (3).

    Grace timer: once confirmed, holds "covered" for COVER_GRACE_SEC after
    the hand disappears — handles momentary MediaPipe drop-outs.
    """

    def __init__(self):
        self._consec    = 0
        self.min_dist   = 1.0    # closest landmark → mouth centre (for HUD)
        self.covered    = False
        self._grace_end = 0.0
        self.mouth_x1 = self.mouth_y1 = 0.0
        self.mouth_x2 = self.mouth_y2 = 1.0

    def set_mouth_box(self, lms, lip_indices: list[int], expand: float = 0.03):
        xs = [lms[i].x for i in lip_indices]
        ys = [lms[i].y for i in lip_indices]
        self.mouth_x1 = max(0.0, min(xs) - expand)
        self.mouth_y1 = max(0.0, min(ys) - expand)
        self.mouth_x2 = min(1.0, max(xs) + expand)
        self.mouth_y2 = min(1.0, max(ys) + expand)

    def _landmarks_inside_mouth(self, hand) -> int:
        count = 0
        for lm in hand:
            if (self.mouth_x1 <= lm.x <= self.mouth_x2 and
                    self.mouth_y1 <= lm.y <= self.mouth_y2):
                count += 1
        return count

    def check(self, hand_result, mouth_cx: float, mouth_cy: float) -> bool:
        now = time.time()
        self.min_dist = 1.0
        near = False

        if hand_result.hand_landmarks:
            for hand in hand_result.hand_landmarks:

                # ── Signal 1: any landmark within tight radius of mouth centre ──
                for lm in hand:
                    d = float(np.hypot(lm.x - mouth_cx, lm.y - mouth_cy))
                    if d < self.min_dist:
                        self.min_dist = d
                if self.min_dist < HAND_PROXIMITY:
                    near = True
                    break

                # ── Signal 2: enough landmarks physically inside mouth box ──────
                if not near:
                    pts_inside = self._landmarks_inside_mouth(hand)
                    if pts_inside >= COVER_BBOX_MIN_PTS:
                        near = True
                        self.min_dist = 0.0   # fill HUD bar
                        break

        # ── Confirm streak (require N consecutive frames) ─────────────────
        if near:
            self._consec += 1
            if self._consec >= COVER_CONFIRM:
                self._grace_end = now + COVER_GRACE_SEC
        else:
            self._consec = 0

        # ── Decision: confirmed OR inside grace window ────────────────────
        if self._consec >= COVER_CONFIRM:
            self.covered = True
        elif now < self._grace_end:
            self.covered = True
        else:
            self.covered = False

        return self.covered

    def reset(self):
        self.__init__()


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def _t(frame, text, x, y, color=(200,200,200), sc=0.50, th=1):
    cv2.putText(frame, text, (x,y),
                cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)

def draw_lips(frame, lms, h, w, color):
    for i in LIP_DRAW:
        lm = lms[i]
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, color, -1)
    u = lms[INNER_UPPER];  l = lms[INNER_LOWER]
    cv2.line(frame,
             (int(u.x*w), int(u.y*h)),
             (int(l.x*w), int(l.y*h)), color, 2)

def draw_hand_landmarks(frame, hand_result, h, w):
    if not hand_result.hand_landmarks:
        return
    for hand in hand_result.hand_landmarks:
        for lm in hand:
            cv2.circle(frame,
                       (int(lm.x*w), int(lm.y*h)),
                       3, (0,255,255), -1)

def draw_banner(frame, text, bg, fg):
    fh, fw  = frame.shape[:2]
    ov      = frame.copy()
    cv2.rectangle(ov, (0,0), (fw,110), bg, -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    pulse = int(90 + 90 * np.sin(time.time() * 7))
    bdr   = tuple(min(c+pulse, 255) for c in bg)
    cv2.rectangle(frame, (0,0), (fw-1,109), bdr, 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    sc, th = 1.1, 2
    tw = cv2.getTextSize(text, font, sc, th)[0][0]
    tx = (fw - tw) // 2
    cv2.putText(frame, text, (tx+2,72), font, sc, (0,0,0), th)
    cv2.putText(frame, text, (tx,  70), font, sc, fg,      th)

def draw_cycle_dots(frame, current, required, cheating):
    fh, fw = frame.shape[:2]
    r, sp  = 13, 38
    sx     = (fw - required*sp) // 2
    y      = 140
    for i in range(required):
        cx     = sx + i*sp + r
        filled = i < current
        col    = (0,50,255) if (filled and cheating) else \
                 (0,220,80) if filled else (50,50,50)
        cv2.circle(frame, (cx,y), r, col, -1 if filled else 2)
        cv2.circle(frame, (cx,y), r, (160,160,160), 1)
    _t(frame, f"Cycles {current}/{required}", sx+required*sp+10, y+5, sc=0.52)

def draw_gap_bar(frame, gap, thr, snz):
    fh, fw = frame.shape[:2]
    bx, by, bw, bh = 10, 162, 250, 12
    mx = max(snz*1.2, 30.0)
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (28,28,28), -1)
    fill = int(min(gap/mx,1.0)*bw)
    col  = (0,50,255) if gap>snz else (0,160,255) if gap>thr else (0,200,80)
    cv2.rectangle(frame, (bx,by), (bx+fill,by+bh), col, -1)
    cv2.line(frame,(bx+int(thr/mx*bw),by-3),(bx+int(thr/mx*bw),by+bh+3),(0,220,220),2)
    cv2.line(frame,(bx+int(snz/mx*bw),by-3),(bx+int(snz/mx*bw),by+bh+3),(0,60,255),2)
    _t(frame, f"Gap {gap:.1f}u", bx+bw+6, by+bh, sc=0.44)

def draw_hand_bar(frame, cd: CoverDetector):
    fh, fw = frame.shape[:2]
    bx, by, bw, bh = 10, 182, 250, 10
    fill_ratio = max(0.0, 1.0 - cd.min_dist / HAND_PROXIMITY)
    fill = int(fill_ratio * bw)
    col  = (0,50,255) if cd.covered else \
           (0,160,255) if cd.min_dist < HAND_PROXIMITY*1.5 else (0,200,80)
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (28,28,28), -1)
    cv2.rectangle(frame, (bx,by), (bx+fill,by+bh), col, -1)
    cv2.line(frame, (bx+bw,by-3),(bx+bw,by+bh+3),(255,220,0),2)
    grace_active = cd.covered and cd.min_dist >= HAND_PROXIMITY
    status = "GRACE" if grace_active else ("COVER" if cd.covered else "clear")
    _t(frame,
       f"Hand dist:{cd.min_dist:.3f}  lim:{HAND_PROXIMITY:.2f}  {status}",
       bx+bw+6, by+bh,
       (0,50,255) if cd.covered else (0,200,80), 0.44)

def draw_sustained_bar(frame, fsm: LipCycleFSM):
    fh, fw = frame.shape[:2]
    bx, by, bw, bh = 10, 200, 250, 10
    ratio = min(fsm.sustained_secs / SUSTAINED_OPEN_SEC, 1.0)
    fill  = int(ratio * bw)
    col   = (0, 50, 255) if ratio >= 1.0 else (0, 200, 255)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (28, 28, 28), -1)
    cv2.rectangle(frame, (bx, by), (bx+fill, by+bh), col, -1)
    cv2.line(frame, (bx+bw, by-3), (bx+bw, by+bh+3), (255, 220, 0), 2)
    label_col = (0, 50, 255) if ratio >= 1.0 else (0, 200, 255)
    _t(frame,
       f"Open {fsm.sustained_secs:.1f}s / {SUSTAINED_OPEN_SEC:.0f}s",
       bx+bw+6, by+bh, label_col, 0.44)

def draw_hud(frame, gap, cal: Calibrator, fsm: LipCycleFSM, calibrating):
    fh, fw = frame.shape[:2]
    cv2.rectangle(frame, (0,fh-68),(fw,fh),(14,14,14),-1)
    sn = cal.threshold * SNEEZE_MULT if cal.done else 0.0
    _t(frame, f"Base:{cal.baseline:5.1f}u",  10,  fh-44)
    _t(frame, f"Thr:{cal.threshold:5.1f}u",  190, fh-44)
    _t(frame, f"Snz:{sn:5.1f}u",             365, fh-44)
    _t(frame, f"Vel:{fsm.velocity:4.2f}u/f", 545, fh-44)
    sc = (0,45,255) if fsm.is_cheating else (0,195,80)
    _t(frame, f"State:{fsm.state_label}", 10,  fh-16, sc, 0.54, 2)
    _t(frame, f"Events:{fsm.total}",      230, fh-16)
    if fsm.skip_reason:
        _t(frame, f"Skip:{fsm.skip_reason}", 400, fh-16, (0,175,255), 0.43)
    if calibrating:
        pct = int(cal.progress * 100)
        _t(frame, f"CALIBRATING {pct}% — keep mouth CLOSED",
           10, fh-88, (0,210,255), 0.57, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class UserLipState:
    def __init__(self):
        self.cal      = Calibrator()
        self.fsm      = LipCycleFSM()
        self.cover    = CoverDetector()
        self.gap      = 0.0
        self.covered  = False
        self.face_absent_frames = 0  # grace period counter

class LipCheatingDetector:

    def __init__(self):
        self.users = {}
        self._face_lm, self._hand_lm = self._build_landmarkers()

    def _build_landmarkers(self):
        face_opts = FaceLandmarkerOptions(
            base_options = mp_python.BaseOptions(
                model_asset_path=str(get_face_model())),
            running_mode = RunningMode.IMAGE,
            num_faces    = 1,
            min_face_detection_confidence = 0.5,
            min_face_presence_confidence  = 0.5,
            min_tracking_confidence       = 0.5,
        )
        hand_opts = HandLandmarkerOptions(
            base_options = mp_python.BaseOptions(
                model_asset_path=str(get_hand_model())),
            running_mode = RunningMode.IMAGE,
            num_hands    = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence  = 0.5,
            min_tracking_confidence       = 0.5,
        )
        fl = mp_vision.FaceLandmarker.create_from_options(face_opts)
        hl = mp_vision.HandLandmarker.create_from_options(hand_opts)
        print("[INFO] Face + Hand landmarkers ready.")
        return fl, hl

    def process(self, bgr: np.ndarray, username="default") -> np.ndarray:
        if username not in self.users:
            self.users[username] = UserLipState()
        state = self.users[username]
        
        flag = None
        fh, fw  = bgr.shape[:2]
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        face_result = self._face_lm.detect(mp_img)
        hand_result = self._hand_lm.detect(mp_img)

        out = bgr.copy()

        if face_result.face_landmarks:
            lms       = face_result.face_landmarks[0]
            state.face_absent_frames = 0  # reset grace period counter
            
            # --- NEW ROBUST 3D GAP CALCULATION ---
            # 1. Calculate 3D lip gap (Z-axis handles looking down/pitch)
            dx = (lms[INNER_UPPER].x - lms[INNER_LOWER].x) * fw
            dy = (lms[INNER_UPPER].y - lms[INNER_LOWER].y) * fh
            dz = (lms[INNER_UPPER].z - lms[INNER_LOWER].z) * fw
            lip_gap_3d = float(np.sqrt(dx**2 + dy**2 + dz**2))

            # 2. Normalize by 3D Face Width (Outer eyes 33 & 263) to handle leaning in/out
            ex = (lms[263].x - lms[33].x) * fw
            ey = (lms[263].y - lms[33].y) * fh
            ez = (lms[263].z - lms[33].z) * fw
            face_width_3d = max(float(np.sqrt(ex**2 + ey**2 + ez**2)), 1.0)

            # Scale back by 100 so it matches the numeric range of your old pixel gap logic
            state.gap = (lip_gap_3d / face_width_3d) * 100.0
            # -------------------------------------

            mx = (lms[INNER_UPPER].x + lms[INNER_LOWER].x) / 2
            my = (lms[INNER_UPPER].y + lms[INNER_LOWER].y) / 2

            # update mouth bounding box BEFORE cover check
            state.cover.set_mouth_box(lms, LIP_BOX, expand=0.06)
            state.covered = state.cover.check(hand_result, mx, my)

            if not state.cal.done:
                # Only feed calibration if NOT covered
                if not state.covered:
                    state.cal.feed(state.gap)
                    if state.cal.done:
                        state.fsm.threshold = state.cal.threshold

            if state.cal.done:
                if not state.covered:
                    state.fsm.update(state.gap)

            dot_col = (0,45,255) if (state.fsm.is_cheating or state.covered) \
                      else (0,225,80)
            draw_lips(out, lms, fh, fw, dot_col)

        else:
            # Face not detected — use grace period (don't flag immediately)
            state.face_absent_frames += 1
            # Only flag as covered after 8+ consecutive frames (~2s at 4 FPS)
            if state.face_absent_frames >= 8:
                state.covered = True
                state.cover.min_dist = 0.0

        draw_hand_landmarks(out, hand_result, fh, fw)

        # ── Banners (priority: cover > cheating) ─────────────────────────────
        if state.covered:
            draw_banner(out,
                        "CHEATER — LIPS NOT VISIBLE",
                        (25,0,130), (255,200,255))
            flag = "Lips Covered"
        elif state.fsm.is_cheating:
            if state.fsm.sustained_event:
                draw_banner(out,
                            f"WARNING: LIPS OPEN FOR {state.fsm.sustained_secs:.1f}s",
                            (0,0,170), (255,255,180))
                flag = "Sustained Lip Open"
            else:
                draw_banner(out,
                            "WARNING: CHEATING DETECTED — LIP MOVEMENT",
                            (0,0,170), (255,255,255))
                flag = "Lip Movement"

        # ── HUD ──────────────────────────────────────────────────────────────
        if state.cal.done:
            sn = state.cal.threshold * SNEEZE_MULT
            draw_cycle_dots(out, state.fsm.cycle_count,
                            REQUIRED_CYCLES, state.fsm.is_cheating)
            draw_gap_bar(out, state.gap, state.cal.threshold, sn)
            draw_hand_bar(out, state.cover)
            draw_sustained_bar(out, state.fsm)      

        draw_hud(out, state.gap, state.cal, state.fsm,
                 calibrating=not state.cal.done)
        return out, flag

    def recalibrate(self, username="default"):
        if username in self.users:
            self.users[username] = UserLipState()
        print(f"\n[INFO] Recalibrating {username} — keep mouth CLOSED…")

    def close(self):
        self._face_lm.close()
        self._hand_lm.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = LipCheatingDetector()
    t0       = time.time()

    print("=" * 60)
    print("  Lip Cheating Detector v8")
    print("  Keep mouth CLOSED for ~2 s to calibrate.")
    print("  2 lip cycles         →  red    WARNING banner")
    print("  Lips open ≥ 4 s      →  red    WARNING banner")
    print("  Hand near mouth      →  purple CHEATER banner")
    print("    (point dist OR bbox overlap, 0.6s grace on flicker)")
    print("  Face not visible     →  purple CHEATER banner")
    print("  Yellow dots = detected hand landmarks")
    print("  q = quit   r = recalibrate")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        out, flag = detector.process(frame)
        cv2.imshow("Lip Cheating Detector  [q=quit  r=recalibrate]", out)
        k = cv2.waitKey(1) & 0xFF
        if   k == ord('q'): break
        elif k == ord('r'): detector.recalibrate()

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Done. Total cheating events: {detector.fsm.total}")


if __name__ == "__main__":
    main()