"""
Audio processing module for Lumii Voice Assistant
Contains optimized audio processing functions and Voice Activity Detection
"""

import numpy as np
import webrtcvad
from scipy import signal
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

def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Resample audio data to target sample rate (default 16kHz for VAD).
    
    Parameters
    ----------
    audio_data : np.ndarray
        Input audio data
    orig_sr : int
        Original sample rate of the audio (e.g., 44100)
    target_sr : int
        Target sample rate (defaults to SAMPLE_RATE which is 16000 for VAD)
    
    Returns
    -------
    np.ndarray
        Resampled audio data
    """
    if orig_sr == target_sr:
        return audio_data
    
    # Calculate number of samples for target duration
    orig_duration = len(audio_data) / orig_sr
    target_samples = int(orig_duration * target_sr)
    
    # Resample using scipy.signal.resample
    resampled = signal.resample(audio_data, target_samples)
    
    return resampled

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
        self.orig_sr = None  # Will be set on first call if resampling is needed

    def process_frame(self, audio_frame: np.ndarray, orig_sr: int = None) -> bool:
        """
        Process audio frame and return VAD decision
        
        Parameters
        ----------
        audio_frame : np.ndarray
            Input audio frame
        orig_sr : int, optional
            Original sample rate of the audio. Only needed on first call if
            different from SAMPLE_RATE (16000)
        """
        # Set original sample rate on first call if provided
        if orig_sr is not None:
            self.orig_sr = orig_sr

        # Resample if needed
        if self.orig_sr and self.orig_sr != SAMPLE_RATE:
            audio_frame = resample_audio(audio_frame, self.orig_sr)

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
        except Exception as e:
            print(f"VAD error: {e}")
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