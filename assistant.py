"""
Assistant module for Lumii Voice Assistant
Contains the main LumiiAssistant class that orchestrates all components
"""

import signal
import sys
import sounddevice as sd
from config import SAMPLE_RATE
from models import OptimizedModels
from audio_recorder import StreamingAudioRecorder

# =============================================================================
# Main Application
# =============================================================================

class LumiiAssistant:
    """Main Lumii Assistant class with optimizations"""

    def __init__(self):
        self.models = OptimizedModels()
        self.recorder = StreamingAudioRecorder()
        self.running = False

        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n🛑 Shutting down Lumii...")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def initialize(self):
        """Initialize the assistant"""
        print("🚀 Initializing Lumii Voice Assistant (Pi 5 Optimized)")

        # Check audio devices
        try:
            devices = sd.query_devices()
            print(f"📱 Available audio devices: {len(devices)}")
        except Exception as e:
            print(f"❌ Audio device error: {e}")
            return False

        # Load models
        self.models.load_models()

        print("✅ Lumii is ready!")
        return True

    def process_conversation_turn(self):
        """Process one conversation turn"""
        # Record audio
        audio_data = self.recorder.start_recording()

        if audio_data is None or len(audio_data) < SAMPLE_RATE * 0.5:  # Min 0.5 seconds
            return True

        # Transcribe
        print("🔤 Transcribing...")
        user_text = self.models.transcribe_audio(audio_data)

        if not user_text:
            print("❓ Could not understand audio")
            return True

        print(f"👤 You said: {user_text}")

        # Check for exit commands
        if user_text.lower().strip() in ["exit.", "quit.", "goodbye.", "bye.", "stop."]:
            self.models.speak("Goodbye! Take care of yourself.")
            return False

        # Analyze emotion
        emotion = self.models.analyze_emotion(user_text)

        # Generate response
        print("🧠 Thinking...")
        response = self.models.generate_response(user_text, emotion)

        # Speak response
        self.models.speak(response)

        return True

    def run(self):
        """Main application loop"""
        if not self.initialize():
            print("❌ Failed to initialize Lumii")
            return

        self.running = True

        # Welcome message
        welcome_msg = "Hello! I'm Lumii, your voice companion. How are you feeling today? Start with your name."
        print(f"🤖 {welcome_msg}")
        self.models.speak(welcome_msg)

        # Main conversation loop
        try:
            while self.running:
                if not self.process_conversation_turn():
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        self.recorder.cleanup()
        print("👋 Goodbye!")
