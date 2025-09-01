"""
Audio processing module for Lumii Voice Assistant
Contains optimized audio processing functions and Voice Activity Detection
"""

import numpy as np
import webrtcvad
from config import (
    VAD_AGGRESSIVENESS, VAD_FRAME_SIZE, SAMPLE_RATE,
    SPEECH_THRESHOLD, SILENCE_THRESHOLD
)

# Optional Numba for JIT compilation on ARM64
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: numba not available - install for better performance")
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit

# =============================================================================
# Optimized Audio Processing with Numba JIT
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def compute_energy(audio_chunk: np.ndarray) -> float:
        """Optimized energy computation using Numba JIT for ARM64"""
        return np.mean(np.square(audio_chunk))

    @njit(fastmath=True, cache=True)
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Fast audio normalization with Numba"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
else:
    def compute_energy(audio_chunk: np.ndarray) -> float:
        return np.mean(np.square(audio_chunk))

    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data

# =============================================================================
# Optimized Voice Activity Detection
# =============================================================================

class OptimizedVAD:
    """Optimized Voice Activity Detection for Raspberry Pi 5"""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_size = VAD_FRAME_SIZE
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Process audio frame and return VAD decision"""
        # Convert to int16 for WebRTC VAD
        audio_int16 = (audio_frame * 32767).astype(np.int16)

        # Ensure frame is correct size
        if len(audio_int16) != self.frame_size:
            if len(audio_int16) < self.frame_size:
                # Pad with zeros
                audio_int16 = np.pad(audio_int16, (0, self.frame_size - len(audio_int16)))
            else:
                # Truncate
                audio_int16 = audio_int16[:self.frame_size]

        # Run VAD
        try:
            is_speech = self.vad.is_speech(audio_int16.tobytes(), SAMPLE_RATE)
        except:
            return False

        # State machine for robust detection
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            if self.speech_frames >= SPEECH_THRESHOLD:
                self.is_speaking = True
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            if self.silence_frames >= SILENCE_THRESHOLD and self.is_speaking:
                self.is_speaking = False
                return "STOP"  # Signal to stop recording

        return self.is_speaking
