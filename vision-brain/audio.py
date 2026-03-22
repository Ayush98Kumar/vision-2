import os
import csv
import urllib.request
import numpy as np
import librosa
import tensorflow as tf
import tensorflow._api.v2.compat.v2.__internal__ as tf_internal
tf_internal.register_load_context_function = tf_internal.register_call_context_function
import tensorflow_hub as hub

# --- 1. CONFIGURATION FOR FALSE-FLAG PREVENTION ---
CONFIDENCE_THRESHOLD = 0.60  # Must be at least 60% sure it's speech

# The ONLY sounds that will trigger a cheating flag. Everything else is ignored.
SUSPICIOUS_SOUNDS = [
    'Speech', 
    'Male speech, man speaking', 
    'Female speech, woman speaking', 
    'Child speech, kid speaking', 
    'Conversation', 
    'Whispering'
]

# --- 2. LOAD AI MODEL (Runs once at startup) ---
print("Downloading/Loading YAMNet AI... (This takes a moment)")
model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_yamnet_class_map():
    class_map_path = 'yamnet_class_map.csv'
    if not os.path.exists(class_map_path):
        url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        urllib.request.urlretrieve(url, class_map_path)
    
    classes = []
    with open(class_map_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            classes.append(row[2])
    return classes

class_names = load_yamnet_class_map()
print("✅ YAMNet AI Ready!")

# --- 3. ANALYSIS LOGIC ---
def analyze_audio_array(wav_data):
    """Analyzes raw audio array and strictly filters out non-speech sounds."""
    try:
        # Run AI prediction
        scores, _, _ = model(wav_data)
        scores_np = scores.numpy()
        
        # Check every 0.48-second chunk
        for chunk_scores in scores_np:
            top_class_index = np.argmax(chunk_scores)
            top_class_name = class_names[top_class_index]
            top_score = chunk_scores[top_class_index]
            
            # THE FILTER: Is it speech AND is it highly confident?
            if top_class_name in SUSPICIOUS_SOUNDS and top_score >= CONFIDENCE_THRESHOLD:
                return f"Voice Detected: {top_class_name}"
                
        return None
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def analyze_audio_clip(filepath):
    """Analyzes the audio and strictly filters out non-speech sounds from a file."""
    try:
        # Load audio at 16kHz mono (Required by YAMNet)
        wav_data, _ = librosa.load(filepath, sr=16000, mono=True)
        return analyze_audio_array(wav_data)
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None