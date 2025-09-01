"""
Configuration module for Lumii Voice Assistant
Contains all configuration constants and settings
"""

import os
import re
import multiprocessing as mp

# =============================================================================
# Audio Configuration - Optimized for low latency
# =============================================================================

SAMPLE_RATE = 16000  # Standard for ASR
CHANNELS = 1
CHUNK_SIZE = 512  # Smaller chunks for lower latency
DTYPE = 'float32'  # Use float32 instead of float64 for ARM efficiency

# =============================================================================
# VAD Configuration - More responsive settings
# =============================================================================

VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS = 30  # 30ms frames (480 samples at 16kHz)
VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
SPEECH_THRESHOLD = 5  # Frames to confirm speech start
SILENCE_THRESHOLD = 15  # Frames of silence to stop
MAX_RECORD_SECONDS = 30

# =============================================================================
# Whisper Configuration - Optimized for Pi 5
# =============================================================================

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
WHISPER_COMPUTE_TYPE = "int8"  # Optimal for ARM
WHISPER_BEAM_SIZE = 1  # Greedy decoding for speed
WHISPER_THREADS = max(1, mp.cpu_count() // 2)  # Use half cores for whisper

# =============================================================================
# ElevenLabs Configuration
# =============================================================================

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID = "19STyYD15bswVz51nqLf"
ELEVEN_MODEL_ID = "eleven_monolingual_v1"

# =============================================================================
# Ollama Configuration
# =============================================================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")  # Slightly larger model for Pi 5
OLLAMA_TIMEOUT = 30

# =============================================================================
# Memory Management
# =============================================================================

MAX_HISTORY = 8
AUDIO_BUFFER_SIZE = SAMPLE_RATE * MAX_RECORD_SECONDS
GC_INTERVAL = 10  # Garbage collection interval

# =============================================================================
# Safety Configuration
# =============================================================================

UNSAFE_TERMS = [
    r"bomb", r"weapon", r"kill", r"murder", r"drugs", r"explosive",
    r"terrorist", r"violence", r"hack", r"steal", r"arson"
]

CRISIS_TERMS = [
    r"suicide", r"kill myself", r"end it all", r"die", r"hurt myself",
    r"worthless", r"i want to die", r"it's over for me", r"don't want to go on"
]

UNSAFE_RE = re.compile(r"\b(" + "|".join(UNSAFE_TERMS) + r")\b", re.IGNORECASE)
CRISIS_RE = re.compile(r"\b(" + "|".join(CRISIS_TERMS) + r")\b", re.IGNORECASE)
