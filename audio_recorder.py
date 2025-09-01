"""
Audio recording module for Lumii Voice Assistant
Contains the StreamingAudioRecorder class for callback-based recording
"""

import queue
import numpy as np
import sounddevice as sd
from typing import Optional

from config import (
    SAMPLE_RATE, CHANNELS, DTYPE, CHUNK_SIZE,
    MAX_RECORD_SECONDS, AUDIO_BUFFER_SIZE
)
from audio_processing import OptimizedVAD, normalize_audio

# =============================================================================
# Optimized Audio Recording with Callbacks
# =============================================================================

class StreamingAudioRecorder:
    """Callback-based audio recorder optimized for low latency"""

    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_recording = False
        self.vad = OptimizedVAD()
        self.audio_buffer = []
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        """Low-latency audio callback"""
        if status:
            print(f"Audio callback status: {status}")

        if self.is_recording:
            # Process audio chunk
            audio_chunk = indata[:, 0].copy()  # Get mono channel

            # VAD processing
            vad_result = self.vad.process_frame(audio_chunk)

            if vad_result == "STOP":
                self.is_recording = False
                self.audio_queue.put("STOP")
            elif vad_result or self.vad.is_speaking:
                self.audio_buffer.append(audio_chunk)

                # Prevent buffer overflow
                if len(self.audio_buffer) > AUDIO_BUFFER_SIZE // CHUNK_SIZE:
                    self.is_recording = False
                    self.audio_queue.put("STOP")

    def start_recording(self) -> Optional[np.ndarray]:
        """Start recording and return audio when speech ends"""
        self.audio_buffer.clear()
        self.is_recording = True
        self.vad.speech_frames = 0
        self.vad.silence_frames = 0
        self.vad.is_speaking = False

        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        print("🎤 Listening...")

        # Start audio stream if not already started
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_SIZE,
                callback=self.audio_callback,
                latency='low'
            )
            self.stream.start()

        # Wait for recording to complete
        try:
            result = self.audio_queue.get(timeout=MAX_RECORD_SECONDS)
            if result == "STOP" and self.audio_buffer:
                audio_data = np.concatenate(self.audio_buffer)
                return normalize_audio(audio_data)
        except queue.Empty:
            print("⏰ Recording timeout")

        return None

    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
