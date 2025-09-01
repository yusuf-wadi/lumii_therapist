"""
Models module for Lumii Voice Assistant
Contains the OptimizedModels class for model management and processing
"""

import re
import subprocess
import numpy as np
from collections import deque
from typing import Tuple

from config import (
    WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE, WHISPER_THREADS,
    ELEVENLABS_API_KEY, VOICE_ID, ELEVEN_MODEL_ID,
    OLLAMA_MODEL, OLLAMA_TIMEOUT,
    MAX_HISTORY, GC_INTERVAL,
    UNSAFE_RE, CRISIS_RE
)
from utils import debug_print

# Optimized imports with error handling
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Warning: faster-whisper not available")
    WhisperModel = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Warning: vaderSentiment not available")
    SentimentIntensityAnalyzer = None

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play
except ImportError:
    print("Warning: elevenlabs not available")
    ElevenLabs = None

# =============================================================================
# Optimized Models and Processing
# =============================================================================

class OptimizedModels:
    """Container for optimized model loading and management"""

    def __init__(self):
        self.whisper_model = None
        self.vader = None
        self.elevenlabs_client = None 
        self.conversation_history = deque(maxlen=MAX_HISTORY)
        self.gc_counter = 0

    def load_models(self):
        """Load models with optimizations"""
        print("🔧 Loading optimized models...")

        # Load Whisper with optimal settings for Pi 5
        if WhisperModel:
            self.whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                compute_type=WHISPER_COMPUTE_TYPE,
                device="cpu",
                cpu_threads=WHISPER_THREADS,
                num_workers=1,
                download_root=None  # Use default cache
            )
            print(f"✅ Whisper {WHISPER_MODEL_SIZE} loaded")

        # Load VADER sentiment analyzer
        if SentimentIntensityAnalyzer:
            self.vader = SentimentIntensityAnalyzer()
            print("✅ VADER sentiment analyzer loaded")

        # Initialize ElevenLabs client
        if ElevenLabs and ELEVENLABS_API_KEY:
            self.elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            print("✅ ElevenLabs client initialized")

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Optimized audio transcription"""
        if not self.whisper_model:
            return ""

        try:
            # Use in-memory processing instead of file I/O
            segments, _ = self.whisper_model.transcribe(
                audio_data,
                beam_size=WHISPER_BEAM_SIZE,
                language="en",
                condition_on_previous_text=False,
                vad_filter=False,  # We already did VAD
                word_timestamps=False
            )

            text = "".join(segment.text for segment in segments).strip()
            return text

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

    def analyze_emotion(self, text: str) -> str:
        """Lightweight emotion analysis"""
        if not self.vader or not text:
            return "neutral"

        try:
            scores = self.vader.polarity_scores(text)
            compound = scores["compound"]
            debug_print(f"VADER scores: {scores}")
            if compound >= 0.5:
                return "joy"
            elif compound <= -0.6:
                return "anger" if scores["neg"] > 0.5 else "sadness"
            elif compound < -0.2:
                return "sadness"
            elif compound > 0.2:
                return "joy"
            elif re.search(r"\b(scared|afraid|anxious|worried|nervous)\b", text, re.IGNORECASE):
                return "fear"
            else:
                return "neutral"

        except Exception as e:
            print(f"❌ Emotion analysis error: {e}")
            return "neutral"

    def safety_filter(self, text: str) -> Tuple[bool, str]:
        """Check message safety and return appropriate response"""
        if CRISIS_RE.search(text):
            return False, "It sounds like you're going through something serious. Please call or text 988 for support."

        if UNSAFE_RE.search(text):
            return False, "I can't help with that topic. Is there something else I can assist you with?"

        return True, ""

    def generate_response(self, user_text: str, emotion: str) -> str:
        """Generate response using Ollama with optimizations"""
        is_safe, safety_response = self.safety_filter(user_text)
        if not is_safe:
            return safety_response

        # Build conversation context
        history = "\n".join(self.conversation_history)

        prompt = (
            f"{history}\n\n"
            f'User says: "{user_text}" (emotion: {emotion})\n'
            "You are Lumii, a therapeutic companion. Respond with kindness and support. "
            "Keep responses short, warm, and encouraging. Never diagnose or give medical advice."
        )

        try:
            result = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL],
                input=prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=OLLAMA_TIMEOUT,
                encoding="utf-8"
            )

            response = (result.stdout or "").strip()
            if not response:
                response = "I'm here to listen and support you."

            # Update conversation history
            self.conversation_history.append(f"User: {user_text}")
            self.conversation_history.append(f"Lumii: {response}")

            # Periodic garbage collection
            self.gc_counter += 1
            if self.gc_counter >= GC_INTERVAL:
                import gc
                gc.collect()
                self.gc_counter = 0

            return response

        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return "I'm here for you, even when words are hard to find."

    def speak(self, text: str):
        """Optimized text-to-speech"""
        if not text or not self.elevenlabs_client:
            print(f"Lumii: {text}")
            return

        try:
            # Clean text for TTS
            clean_text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
            clean_text = re.sub(r'[\*\[\]]+', '', clean_text)  # Remove markdown
            clean_text = clean_text.strip()

            if not clean_text:
                return

            print(f"🔊 Lumii: {clean_text}")

            # Generate and play audio
            audio_stream = self.elevenlabs_client.text_to_speech.convert(
                voice_id=VOICE_ID,
                model_id=ELEVEN_MODEL_ID,
                text=clean_text,
                optimize_streaming_latency=1
            )

            play(audio_stream)

        except Exception as e:
            print(f"❌ TTS error: {e}")
            print(f"Lumii: {text}")
