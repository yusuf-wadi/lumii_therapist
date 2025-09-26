# Lumii Voice Assistant - Modular Architecture

A modular voice assistant optimized for Raspberry Pi 5 with ARM64 NEON optimizations, callback-based audio, memory efficiency, and multiprocessing.

## Architecture Overview

The codebase has been restructured from a single monolithic file into separate, interacting modules for better maintainability, testing, and extensibility.

### Module Structure

```
lumii_therapist/
├── __init__.py              # Package initialization
├── main.py                  # Entry point
├── config.py                # Configuration constants
├── utils.py                 # Utility functions
├── audio_processing.py      # Audio processing & VAD
├── audio_recorder.py        # Streaming audio recorder
├── models.py                # Model management (Whisper, VADER, ElevenLabs)
├── assistant.py             # Main assistant orchestration
├── requirements.txt         # Dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

### Module Responsibilities

#### `config.py`
- Audio configuration (sample rate, channels, chunk size)
- VAD configuration (aggressiveness, thresholds)
- Model configuration (Whisper, ElevenLabs, Ollama)
- Safety filters (unsafe terms, crisis terms)
- Memory management settings

#### `utils.py`
- Utility functions like `debug_print()`
- Shared helper functions across modules

#### `audio_processing.py`
- Optimized audio processing functions with Numba JIT
- `compute_energy()` - Energy computation
- `normalize_audio()` - Audio normalization
- `OptimizedVAD` class - Voice Activity Detection

#### `audio_recorder.py`
- `StreamingAudioRecorder` class
- Callback-based audio recording
- Real-time VAD integration
- Buffer management

#### `models.py`
- `OptimizedModels` class
- Model loading and management (Whisper, VADER, ElevenLabs)
- Audio transcription
- Emotion analysis
- Response generation with Ollama
- Text-to-speech
- Safety filtering

#### `assistant.py`
- `LumiiAssistant` class
- Main application orchestration
- Conversation loop management
- Signal handling for graceful shutdown
- Component initialization

#### `main.py`
- Entry point with system optimizations
- Process priority and CPU affinity settings
- Assistant instantiation and execution

## Installation

### Prerequisites

1. **Raspberry Pi 5** with ARM64 OS
2. **Python 3.8+**
3. **Ollama** installed and running
4. **Audio hardware** (microphone and speakers)

### Setup

1. Clone or download the modular code:
```bash
mkdir lumii_assistant
cd lumii_assistant
# Copy all the Python files to this directory
```

2. Create Env with [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install#command-prompt):
```bash
conda env create -f environment.yml --force-reinstal
```
or less reliably
```bash
pip install -r requirements.txt
```

3. Install system dependencies:
```bash
# For audio processing
sudo apt-get install portaudio19-dev python3-pyaudio

# For ARM64 optimizations (optional but recommended)
sudo apt-get install libblas-dev liblapack-dev
```

4. Install and configure Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma2:2b
```

## Usage

### Running the Assistant

```bash
python main.py
```

Or if installed as a package:
```bash
lumii
```

### Programmatic Usage

```python
from assistant import LumiiAssistant

# Create assistant instance
assistant = LumiiAssistant()

# Run the assistant
assistant.run()
```

### Using Individual Components

```python
from models import OptimizedModels
from audio_recorder import StreamingAudioRecorder
from audio_processing import OptimizedVAD

# Use models independently
models = OptimizedModels()
models.load_models()

# Use audio recorder independently
recorder = StreamingAudioRecorder()
audio_data = recorder.start_recording()

# Use VAD independently
vad = OptimizedVAD()
is_speech = vad.process_frame(audio_frame)
```

## Configuration

Edit `config.py` to customize:

- Audio settings (sample rate, chunk size)
- Model settings (Whisper model size, Ollama model)
- API keys (ElevenLabs)
- Safety filters
- Performance parameters

## Optimizations

### ARM64 Specific
- Numba JIT compilation for audio processing
- Optimal CPU affinity settings
- Memory-efficient data types (`float32`)

### Audio Processing
- Callback-based streaming audio
- WebRTC VAD for speech detection
- Minimal latency buffering

### Model Management
- Lazy loading of heavy models
- Periodic garbage collection
- Memory usage monitoring

## Architecture Benefits

### Modularity
- **Separation of Concerns**: Each module has a single responsibility
- **Testability**: Individual components can be tested in isolation
- **Maintainability**: Changes to one module don't affect others
- **Reusability**: Modules can be used independently or in other projects

### Extensibility
- **Plugin Architecture**: New models or audio processors can be easily added
- **Configuration Management**: Centralized configuration makes customization simple
- **Interface Consistency**: Clear APIs between modules

### Performance
- **Optimized Imports**: Conditional imports reduce startup time
- **Resource Management**: Proper cleanup and resource handling
- **Memory Efficiency**: Garbage collection and buffer management

## Development

### Adding New Models
1. Add configuration to `config.py`
2. Implement model interface in `models.py`
3. Update `load_models()` method

### Adding New Audio Processors
1. Implement processing functions in `audio_processing.py`
2. Update VAD or recorder classes as needed
3. Add Numba optimizations if applicable

### Adding New Features
1. Determine the appropriate module
2. Implement feature with proper error handling
3. Update configuration if needed
4. Add tests for the new functionality

## Troubleshooting

### Audio Issues
- Check `sd.query_devices()` for available audio devices
- Verify microphone permissions
- Adjust audio buffer sizes in `config.py`

### Model Loading Issues
- Ensure sufficient RAM (at least 4GB for Pi 5)
- Check internet connection for model downloads
- Verify API keys in `config.py`

### Performance Issues
- Install Numba for JIT optimizations
- Adjust CPU affinity settings
- Monitor memory usage

## License

MIT License - see LICENSE file for details.
