import os
import json
import time
import base64
import uuid
import threading
import traceback
import tempfile
import struct
from datetime import datetime
from collections import deque
import queue
import gc
import sys
import signal
import subprocess
import wave

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import imageio
import torch

# --- Import Detection Scripts ---
from phone import PhoneDetector
from face_multi import FaceMultiDetector
from face_move import FaceMoveDetector
from lip_cheat_detector import LipCheatingDetector
from eye_cheat_detector import EyeCheatDetector
from hand import HandDetector

# ==========================================
# App Configuration
# ==========================================
# Project root = one level up from this file (vision/vision/app.py → vision/)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)

app = Flask(
    __name__,
    static_folder=os.path.join(_PROJECT_ROOT, 'static'),
    static_url_path='/static',
    template_folder=os.path.join(_APP_DIR, 'templates'),
)
app.secret_key = 'super_secret_proctor_key'
CORS(app)

print(f"[INFO] Project root: {_PROJECT_ROOT}")
print(f"[INFO] Static folder: {app.static_folder}")

# SocketIO: use threading async_mode for maximum compatibility & stability
# threading mode avoids eventlet/gevent monkey-patching issues that cause random crashes
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB max per message
    logger=False,
    engineio_logger=False,
)

# ==========================================
# Globals and Directories
# ==========================================
FLAGS_FILE = os.path.join(_PROJECT_ROOT, 'flags.json')
STUDENTS_FILE = os.path.join(_PROJECT_ROOT, 'students.json')
FLAGS_DIR = os.path.join(_PROJECT_ROOT, 'static', 'flags')
LATEST_DIR = os.path.join(_PROJECT_ROOT, 'static', 'latest')

os.makedirs(FLAGS_DIR, exist_ok=True)
os.makedirs(LATEST_DIR, exist_ok=True)

# Always start with fresh data on server boot
with open(FLAGS_FILE, 'w') as f:
    json.dump([], f)

with open(STUDENTS_FILE, 'w') as f:
    json.dump([], f)

# ==========================================
# Thread-Safe File I/O
# ==========================================
_file_lock = threading.Lock()

def safe_read_json(filepath, default=None):
    """Thread-safe JSON file read. Never crashes."""
    if default is None:
        default = []
    try:
        with _file_lock:
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception:
        return default

def safe_write_json(filepath, data):
    """Thread-safe JSON file write. Never crashes."""
    try:
        with _file_lock:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=None)
    except Exception as e:
        print(f"[ERROR] Failed to write {filepath}: {e}")

# ==========================================
# Trust Score System
# ==========================================
trust_scores = {}           # {username: int}
terminated_students = set()
_state_lock = threading.Lock()

SCORE_WEIGHTS = {
    'multi': 10,
    'lip': 3,
    'head': 5,
    'eye': 5,
    'hand': 5,
    'phone': 15,
}

def get_score_delta(flag_summary):
    """Parse the flag summary string and return the trust score increment."""
    flag_lower = flag_summary.lower()
    delta = 0
    for key, weight in SCORE_WEIGHTS.items():
        if key in flag_lower:
            delta += weight
    return max(delta, 5)

# ==========================================
# AI Model Initialization (with crash protection)
# ==========================================
print("[INFO] Loading AI Models...")
_models_loaded = True

try:
    phone_dt = PhoneDetector()
    print("[INFO]   ✓ PhoneDetector loaded")
except Exception as e:
    print(f"[WARN]   ✗ PhoneDetector failed to load: {e}")
    phone_dt = None

try:
    face_multi_dt = FaceMultiDetector()
    print("[INFO]   ✓ FaceMultiDetector loaded")
except Exception as e:
    print(f"[WARN]   ✗ FaceMultiDetector failed to load: {e}")
    face_multi_dt = None

try:
    facemove_dt = FaceMoveDetector()
    print("[INFO]   ✓ FaceMoveDetector loaded")
except Exception as e:
    print(f"[WARN]   ✗ FaceMoveDetector failed to load: {e}")
    facemove_dt = None

try:
    lip_dt = LipCheatingDetector()
    print("[INFO]   ✓ LipCheatingDetector loaded")
except Exception as e:
    print(f"[WARN]   ✗ LipCheatingDetector failed to load: {e}")
    lip_dt = None

try:
    eye_dt = EyeCheatDetector()
    print("[INFO]   ✓ EyeCheatDetector loaded")
except Exception as e:
    print(f"[WARN]   ✗ EyeCheatDetector failed to load: {e}")
    eye_dt = None

try:
    hand_dt = HandDetector()
    print("[INFO]   ✓ HandDetector loaded")
except Exception as e:
    print(f"[WARN]   ✗ HandDetector failed to load: {e}")
    hand_dt = None

_active = [n for n, d in [('phone', phone_dt), ('face_multi', face_multi_dt),
           ('face_move', facemove_dt), ('lip', lip_dt), ('eye', eye_dt), ('hand', hand_dt)] if d]
print(f"[INFO] AI Models Ready: {len(_active)}/6 active → {', '.join(_active)}")

# ==========================================
# Frame Processing Pipeline
# ==========================================
# Large queue + frame skipping for zero-lag real-time processing
ai_queue = queue.Queue(maxsize=10)

# Per-user state: frame buffers, cooldowns, last processed time
frame_buffers = {}      # {username: deque of {timestamp, cv2_frame}}
audio_buffers = {}      # {username: deque of {timestamp, pcm_bytes}}
flag_cooldowns = {}     # {username: {flag_type: float timestamp}} — per-flag-type cooldowns
_last_ai_frame = {}     # {username: float} — throttle AI per user

# Constants
FRAME_BUFFER_MAX = 36           # ~6 seconds at 6 FPS
AI_THROTTLE_INTERVAL = 0.15     # Process AI at most every 150ms per user (~6.6 AI FPS)
STALE_BUFFER_TIMEOUT = 30.0     # Clean up buffers for users inactive > 30s
MEMORY_CLEANUP_INTERVAL = 15.0  # Run garbage collection every 15s

# ==========================================
# Helper: Save Flag History
# ==========================================
def save_flag(username, flag_type, proof_filename):
    """Save flag to JSON and emit real-time event. Thread-safe."""
    flag_entry = {
        'username': username,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'flag_type': flag_type,
        'proof_link': f'/static/flags/{proof_filename}'
    }

    flags = safe_read_json(FLAGS_FILE, [])
    flags.append(flag_entry)
    safe_write_json(FLAGS_FILE, flags)

    print(f"\n[ALERT] {username} flagged: {flag_type} → {proof_filename}")

    try:
        socketio.emit('new_flag', flag_entry)
    except Exception:
        pass

# ==========================================
# Helper: Write proof video and emit events
# ==========================================
def write_proof_video_async(proof_path, buffer_snapshot, audio_snapshot, username, flag_summary, proof_filename, trust_delta, fps=6.0):
    """Emit dashboard updates IMMEDIATELY, then write proof video in a background thread."""

    # ── 1. Update trust score + emit events IMMEDIATELY (before video write) ──
    with _state_lock:
        if username not in trust_scores:
            trust_scores[username] = 0
        trust_scores[username] = min(trust_scores[username] + trust_delta, 100)
        current_score = trust_scores[username]

    ts_data = {
        'username': username,
        'score': current_score,
        'flag_type': flag_summary,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        socketio.emit('trust_update', ts_data)
    except Exception:
        pass

    # Auto-terminate if score >= 100
    if current_score >= 100:
        with _state_lock:
            terminated_students.add(username)
        try:
            socketio.emit('exam_terminated', {
                'username': username,
                'reason': 'Trust score exceeded threshold'
            })
        except Exception:
            pass
        print(f"[TERMINATED] {username} (trust: {current_score})")

    # ── 2. Write proof video in background thread ──
    def _write():
        try:
            if not buffer_snapshot:
                return
            
            # Use temporary files for muxing
            temp_video = proof_path.replace('.mp4', '_temp.mp4')
            temp_audio = proof_path.replace('.mp4', '_temp.wav')
            
            writer = imageio.get_writer(
                temp_video, fps=fps, codec='libx264',
                macro_block_size=None,
            )
            for buf in buffer_snapshot:
                frame_rgb = cv2.cvtColor(buf['cv2_frame'], cv2.COLOR_BGR2RGB)
                writer.append_data(frame_rgb)
            writer.close()

            # Write audio to temporary WAV
            if audio_snapshot:
                # audio is 16kHz, mono, 16-bit
                with wave.open(temp_audio, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    for a_buf in audio_snapshot:
                        wf.writeframes(a_buf['pcm_bytes'])
                
                # Mux using ffmpeg
                try:
                    result = subprocess.run(['ffmpeg', '-y', '-i', temp_video, '-i', temp_audio, '-c:v', 'copy', '-c:a', 'aac', proof_path], capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"[ERROR] FFmpeg failed: {result.stderr}")
                        import shutil
                        shutil.move(temp_video, proof_path)
                except Exception as e:
                    print(f"[ERROR] FFmpeg exception: {e}")
                    import shutil
                    shutil.move(temp_video, proof_path)
                
                # Cleanup
                if os.path.exists(temp_video): os.remove(temp_video)
                if os.path.exists(temp_audio): os.remove(temp_audio)
            else:
                # If no audio, just rename video
                import shutil
                shutil.move(temp_video, proof_path)

        except Exception as e:
            print(f"[ERROR] Video write failed: {e}")
            
        finally:
            # Video is fully saved on disk — NOW save flag entry to JSON + emit to audit page
            save_flag(username, flag_summary, proof_filename)
    
    t = threading.Thread(target=_write, daemon=True)
    t.start()

# ==========================================
# Routes: Authentication & Pages
# ==========================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if username == 'ayush123' and password == '1234':
            session['username'] = username

            students = safe_read_json(STUDENTS_FILE, [])
            # Remove old entry for this user if re-logging in
            students = [s for s in students if s.get('username') != username]
            students.append({
                'username': username,
                'login_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            safe_write_json(STUDENTS_FILE, students)

            return redirect(url_for('index'))
        else:
            return "Invalid Credentials", 401
    return render_template('login.html')

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

@app.route('/logout')
def logout():
    username = session.get('username')
    if username:
        try:
            students = safe_read_json(STUDENTS_FILE, [])
            for s in students:
                if s['username'] == username and not s.get('logout_time'):
                    s['logout_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
            safe_write_json(STUDENTS_FILE, students)
        except Exception as e:
            print(f"[Logout] Error: {e}")
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    students = safe_read_json(STUDENTS_FILE, [])
    return render_template('dashboard.html', students=students)

@app.route('/audit')
def audit_page():
    flags = safe_read_json(FLAGS_FILE, [])
    return render_template('audit.html', flags=flags)

@app.route('/dashboard_data')
def dashboard_data():
    flags = safe_read_json(FLAGS_FILE, [])
    return jsonify(flags)

@app.route('/api/trust_scores')
def api_trust_scores():
    with _state_lock:
        return jsonify(dict(trust_scores))

@app.route('/api/students')
def api_students():
    students = safe_read_json(STUDENTS_FILE, [])
    return jsonify(students)

@app.route('/health')
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'active_buffers': len(frame_buffers),
        'ai_queue_size': ai_queue.qsize(),
        'trust_scores': len(trust_scores),
        'terminated': len(terminated_students),
    })

# ==========================================
# WebSocket: Real-Time Stream Handling
# ==========================================
@socketio.on('connect')
def handle_connect():
    print(f"[WS] Connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WS] Disconnected: {request.sid}")

@socketio.on('frame')
def handle_frame(data):
    """
    Receives base64 JPEG frames from student at ~6 FPS.
    Pipeline:
      1. Decode frame
      2. Broadcast to teacher IMMEDIATELY (zero-lag live feed)
      3. Throttle + queue for AI processing (non-blocking)
    """
    try:
        username = data.get('username')
        b64_img = data.get('image')

        if not username or not b64_img:
            return

        # Initialize per-user state
        if username not in frame_buffers:
            frame_buffers[username] = deque(maxlen=FRAME_BUFFER_MAX)
            flag_cooldowns[username] = {}
            _last_ai_frame[username] = 0

        # 1. Decode Base64 → OpenCV frame
        if "," in b64_img:
            encoded = b64_img.split(",", 1)[1]
        else:
            encoded = b64_img

        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # 2. Store in rolling buffer for proof video capture
        frame_buffers[username].append({
            'timestamp': time.time(),
            'cv2_frame': frame
        })

        # 3. Broadcast to teacher dashboard IMMEDIATELY (zero lag)
        socketio.emit('teacher_frame', {'username': username, 'image': b64_img})

        # 4. Throttled AI queueing — only queue if enough time has passed
        now = time.time()
        if now - _last_ai_frame.get(username, 0) >= AI_THROTTLE_INTERVAL:
            _last_ai_frame[username] = now
            try:
                ai_queue.put_nowait((username, frame))
            except queue.Full:
                # Queue full = AI is busy, skip this frame (no lag on stream)
                pass

    except Exception as e:
        # NEVER let a frame handler crash the WebSocket
        print(f"[WS] Frame error for {data.get('username', '?')}: {e}")

# ==========================================
# Smart Audio Detection (YAMNet Classification)
# ==========================================
from audio import analyze_audio_array

# ── Duration-based filtering (prevents false positives) ───────────────
AUDIO_COOLDOWN_SECONDS = 8.0    # min seconds between audio flags per user

# Per-user audio state
_audio_state = {}  # {username: {last_flag}}
_audio_lock = threading.Lock()


@socketio.on('audio')
def handle_audio(data):
    """
    Receives raw 16-bit PCM audio at 16kHz mono from the student frontend.
    Passes directly to YAMNet for classification.
    """
    try:
        username = data.get('username')
        audio_b64 = data.get('audio')
        if not username or not audio_b64:
            return

        pcm_bytes = base64.b64decode(audio_b64)
        if len(pcm_bytes) < 1024:  # too small to analyze
            return

        # Store raw PCM audio for proof video (approx 500ms per packet, maxlen 12 = 6 sec)
        if username not in audio_buffers:
            audio_buffers[username] = deque(maxlen=12)
        audio_buffers[username].append({
            'timestamp': time.time(),
            'pcm_bytes': pcm_bytes
        })

        # Convert raw PCM bytes to float32 tensor between -1 and 1
        int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        float_arr = int16_arr.astype(np.float32) / 32768.0

        # Run analysis using YAMNet model from audio.py
        flag_reason = analyze_audio_array(float_arr)

        if not flag_reason:
            return

        now = time.time()
        
        with _audio_lock:
            if username not in _audio_state:
                _audio_state[username] = {'last_flag': 0}

            state = _audio_state[username]

            if now - state['last_flag'] > AUDIO_COOLDOWN_SECONDS:
                state['last_flag'] = now

                print(f"[AUDIO] {flag_reason} for {username}")

                # Save flag with video proof from frame buffer
                buffer_snapshot = list(frame_buffers.get(username, []))
                audio_snapshot = list(audio_buffers.get(username, []))
                if buffer_snapshot:
                    proof_filename = f"{username}_{int(now)}_audio_{uuid.uuid4().hex[:6]}.mp4"
                    proof_path = os.path.join(FLAGS_DIR, proof_filename)
                    write_proof_video_async(proof_path, buffer_snapshot, audio_snapshot, username, flag_reason, proof_filename, 15)

    except Exception as e:
        print(f"[AUDIO] Error for {data.get('username', '?')}: {e}")

# ==========================================
# AI Worker Thread (background processing)
# ==========================================
def ai_worker():
    """
    Dedicated thread that processes queued frames through AI models.
    - Drains queue aggressively (skips old frames, only processes latest per user)
    - Never crashes — every model call is wrapped in try/except
    - Video proof writing is offloaded to yet another thread
    """
    print("[AI Worker] Started. Waiting for frames...")

    while True:
        try:
            # Drain queue: get all pending frames, keep only latest per user
            batch = {}
            try:
                # Block for up to 0.5s waiting for first frame
                username, frame = ai_queue.get(timeout=0.5)
                batch[username] = frame
            except queue.Empty:
                continue

            # Drain remaining without blocking
            while not ai_queue.empty():
                try:
                    username, frame = ai_queue.get_nowait()
                    batch[username] = frame  # overwrite = keep only latest
                except queue.Empty:
                    break

            # Process each user's latest frame
            for username, frame in batch.items():
                try:
                    _process_single_frame(username, frame)
                except Exception as e:
                    print(f"[AI Worker] Error for {username}: {e}")

        except Exception as e:
            print(f"[AI Worker] Loop error: {e}")
            time.sleep(0.5)


def _process_single_frame(username, frame):
    """Process one frame through all AI models. Fully crash-protected."""

    # Skip terminated students
    with _state_lock:
        if username in terminated_students:
            return

    flagged_reasons = set()

    # Run each model independently — one failure doesn't stop others
    if face_multi_dt:
        try:
            r = face_multi_dt.process_frame(frame)
            if r:
                flagged_reasons.add(r)
        except Exception as e:
            print(f"[AI] face_multi error: {e}")

    if phone_dt:
        try:
            r = phone_dt.process_frame(frame, username)
            if r:
                flagged_reasons.add(r)
        except Exception as e:
            print(f"[AI] phone error: {e}")

    if lip_dt:
        try:
            _, r = lip_dt.process(frame, username)
            if r:
                flagged_reasons.add(r)
        except Exception as e:
            print(f"[AI] lip error: {e}")

    if eye_dt:
        try:
            r = eye_dt.process_frame(frame, username)
            if r:
                flagged_reasons.add(r)
        except Exception as e:
            print(f"[AI] eye error: {e}")

    if facemove_dt:
        try:
            r = facemove_dt.process_frame(frame, username)
            if r:
                flagged_reasons.add(r)
        except Exception as e:
            print(f"[AI] face_move error: {e}")

    if hand_dt:
        try:
            r = hand_dt.process_frame(frame, username)
            if r:
                flagged_reasons.add(r)
        except Exception as e:
            print(f"[AI] hand error: {e}")

    # Filter conflicting flags
    if "Lips Covered" in flagged_reasons and "Eyes Off Screen" in flagged_reasons:
        flagged_reasons.remove("Eyes Off Screen")

    # If flagged, apply per-flag-type cooldowns (each flag type is independent)
    if not flagged_reasons:
        return

    current_time = time.time()

    # Per-flag-type cooldown durations (seconds)
    FLAG_TYPE_COOLDOWNS = {
        'Lips Covered': 5.0,
        'Lip Movement Detected': 5.0,
        'Multiple Faces Detected': 5.0,
        'Suspicious Head Movement': 5.0,
        'Suspicious Hand Movement': 5.0,
        'Eyes Off Screen': 3.0,
        'Phone Detected': 10.0,
    }
    DEFAULT_COOLDOWN = 3.0

    # Initialize per-user per-flag-type cooldown dict
    if username not in flag_cooldowns:
        flag_cooldowns[username] = {}

    # Filter flags that have passed their cooldown
    ready_flags = set()
    for flag in flagged_reasons:
        cooldown_dur = FLAG_TYPE_COOLDOWNS.get(flag, DEFAULT_COOLDOWN)
        last_time = flag_cooldowns[username].get(flag, 0)
        if current_time - last_time > cooldown_dur:
            ready_flags.add(flag)
            flag_cooldowns[username][flag] = current_time

    if not ready_flags:
        return

    print(f"[FLAG] {username}: {ready_flags}")

    flag_summary = " | ".join(ready_flags)

    # Snapshot buffer for proof video
    buffer_snapshot = list(frame_buffers.get(username, []))
    audio_snapshot = list(audio_buffers.get(username, []))
    if buffer_snapshot:
        proof_filename = f"{username}_{int(current_time)}_{uuid.uuid4().hex[:6]}.mp4"
        proof_path = os.path.join(FLAGS_DIR, proof_filename)

        # Calculate score increment
        delta = get_score_delta(flag_summary)

        # Write video in SEPARATE thread, emitting events when fully written
        write_proof_video_async(proof_path, buffer_snapshot, audio_snapshot, username, flag_summary, proof_filename, delta)


# ==========================================
# Memory Cleanup Thread
# ==========================================
def memory_cleanup_worker():
    """Periodically clean up stale frame buffers and force garbage collection."""
    while True:
        try:
            time.sleep(MEMORY_CLEANUP_INTERVAL)
            now = time.time()
            stale_users = []

            for user, buf in list(frame_buffers.items()):
                if buf and (now - buf[-1]['timestamp']) > STALE_BUFFER_TIMEOUT:
                    stale_users.append(user)

            for user in stale_users:
                frame_buffers.pop(user, None)
                audio_buffers.pop(user, None)
                _last_ai_frame.pop(user, None)
                print(f"[Cleanup] Removed stale buffer for {user}")

            # Force garbage collection to free frame memory
            gc.collect()

        except Exception as e:
            print(f"[Cleanup] Error: {e}")

# ==========================================
# Entry Point
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("  VISION® AI PROCTORING SERVER")
    print(f"  Starting at http://0.0.0.0:5000")
    print("=" * 60)

    # Start AI worker thread
    ai_thread = threading.Thread(target=ai_worker, daemon=True, name="AI-Worker")
    ai_thread.start()

    # Start memory cleanup thread
    cleanup_thread = threading.Thread(target=memory_cleanup_worker, daemon=True, name="Mem-Cleanup")
    cleanup_thread.start()

    # Run with threading mode (most stable, no monkey-patching required)
    socketio.run(
        app,
        debug=False,         # debug=False prevents double-loading and instability
        host='0.0.0.0',
        port=5000,
        use_reloader=False,  # reloader causes duplicate threads
        allow_unsafe_werkzeug=True,
    )
