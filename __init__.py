"""
Lumii Voice Assistant - A modular voice assistant for Raspberry Pi 5
"""

__version__ = "1.0.0"
__author__ = "Yusuf Wadi (Lumii Assistant Team)"

from .assistant import LumiiAssistant
from .models import OptimizedModels
from .audio_recorder import StreamingAudioRecorder
from .audio_processing import OptimizedVAD, normalize_audio, compute_energy

__all__ = [
    'LumiiAssistant',
    'OptimizedModels', 
    'StreamingAudioRecorder',
    'OptimizedVAD',
    'normalize_audio',
    'compute_energy'
]
