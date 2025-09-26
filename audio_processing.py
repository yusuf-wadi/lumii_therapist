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
    
    try:
        # Calculate target number of samples
        target_samples = int(len(audio_data) * (target_sr / orig_sr))
        
        # Ensure the output will be exactly the frame size needed for VAD
        target_samples = VAD_FRAME_SIZE
        
        # Resample using scipy.signal.resample
        resampled = signal.resample(audio_data, target_samples)
        
        # Ensure we got exactly what we wanted
        if len(resampled) != VAD_FRAME_SIZE:
            print(f"Warning: Resampling produced {len(resampled)} samples, expected {VAD_FRAME_SIZE}")
            # Fix the size if needed
            if len(resampled) < VAD_FRAME_SIZE:
                resampled = np.pad(resampled, (0, VAD_FRAME_SIZE - len(resampled)))
            else:
                resampled = resampled[:VAD_FRAME_SIZE]
        
        return resampled
        
    except Exception as e:
        print(f"Error in resample_audio: {str(e)}")
        print(f"Input shape: {audio_data.shape}, orig_sr: {orig_sr}, target_sr: {target_sr}")
        return np.zeros(VAD_FRAME_SIZE)  # Return silent frame on error

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
        
        # Validate frame size (must be 10, 20, or 30ms at 16kHz)
        frame_ms = (self.frame_size / SAMPLE_RATE) * 1000
        if frame_ms not in [10, 20, 30]:
            raise ValueError(f"Frame size must be 10, 20, or 30ms at 16kHz. Current: {frame_ms}ms")

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
        try:
            # Set original sample rate on first call if provided
            if orig_sr is not None:
                self.orig_sr = orig_sr

            # Debug info
            if len(audio_frame) == 0:
                print("Warning: Empty audio frame received")
                return False

            # Resample if needed
            if self.orig_sr and self.orig_sr != SAMPLE_RATE:
                audio_frame = resample_audio(audio_frame, self.orig_sr)
                
                # Debug info
                print(f"Resampled from {self.orig_sr}Hz to {SAMPLE_RATE}Hz. "
                      f"Frame size before: {len(audio_frame)}, "
                      f"expected: {self.frame_size}")

            # Convert to int16 for WebRTC VAD
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            
            # Ensure data is contiguous
            if not audio_int16.flags['C_CONTIGUOUS']:
                audio_int16 = np.ascontiguousarray(audio_int16)

            # Ensure frame is correct size
            if len(audio_int16) != self.frame_size:
                if len(audio_int16) < self.frame_size:
                    # Pad with zeros
                    audio_int16 = np.pad(audio_int16, (0, self.frame_size - len(audio_int16)))
                else:
                    # Truncate
                    audio_int16 = audio_int16[:self.frame_size]
                
                # Debug info
                print(f"Adjusted frame size from {len(audio_frame)} to {self.frame_size}")

            # Validate frame before VAD
            if len(audio_int16) != self.frame_size:
                print(f"Error: Invalid frame size {len(audio_int16)}, expected {self.frame_size}")
                return False
            
            if not audio_int16.dtype == np.int16:
                print(f"Error: Invalid dtype {audio_int16.dtype}, expected int16")
                return False

            # Run VAD
            try:
                is_speech = self.vad.is_speech(audio_int16.tobytes(), SAMPLE_RATE)
            except Exception as e:
                print(f"VAD error: {e}")
                print(f"Frame info: size={len(audio_int16)}, "
                      f"dtype={audio_int16.dtype}, "
                      f"min={audio_int16.min()}, "
                      f"max={audio_int16.max()}, "
                      f"contiguous={audio_int16.flags['C_CONTIGUOUS']}")
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

        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return False