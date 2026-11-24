# Harmonix — AI Vocal Harmonizer

**An experimental real-time vocal harmonization system built with Python.**

Harmonix explores how digital signal processing and music theory can be combined to create layered vocal harmonies from live audio input.

The project is designed as a modular desktop application, with separate components for pitch detection, harmony generation, pitch shifting, audio input/output, session management, and visualization.

It is an ongoing engineering project focused on real-time audio processing, modular architecture, and performance optimization.

> **Status:** Experimental. Core processing modules and automated tests are implemented. The desktop application shell is available, while real-time playback stability and latency are still being optimized.

## Core Components

- **Pitch Detection:** Analyses vocal audio to estimate fundamental frequency.
- **Harmony Generation:** Generates additional harmony parts using musical relationships.
- **Pitch Shifting:** Applies pitch transformations to create layered vocal output.
- **Key and Scale Resolution:** Supports musical key and scale configuration for harmony processing.
- **Live Audio Processing:** Coordinates microphone input and real-time audio processing.
- **Audio Playback:** Handles processed audio output.
- **Audio Upload and Analysis:** Provides modules for loading and analysing audio files.
- **Session Management:** Separates application state, commands, and state transitions.
- **Audio Export:** Includes a dedicated export pipeline.
- **Latency Measurement:** Tracks processing and audio timing information.

## Architecture

Harmonix uses a modular `src/`-based Python architecture. Audio processing, application state, file operations, and the graphical interface are separated into individual components.

| Component | Responsibility |
|---|---|
| `input/` | Audio input and sample-rate handling |
| `pitch_detector.py` | Fundamental frequency estimation |
| `harmony_generator.py` | Harmony generation |
| `pitch_shifter.py` | Pitch transformation |
| `key_scale_resolver.py` | Musical key and scale resolution |
| `live_audio.py` | Live audio coordination |
| `playback_engine.py` | Audio playback |
| `latency.py` | Latency measurement and reporting |
| `session_state/` | Application state and command handling |
| `upload/` | Audio loading and analysis |
| `export/` | Processed audio export |
| `persistence/` | Session persistence |
| `ui/` | Pygame desktop application shell |

This separation makes individual components easier to test, maintain, and extend.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core application |
| NumPy | Numerical and array operations |
| Librosa | Audio analysis and pitch-related processing |
| PyWorld | Vocal and speech signal processing |
| Music21 | Music theory and musical representations |
| SoundDevice | Audio input/output |
| Pygame | Desktop interface |
| Samplerate | Audio resampling |

## Project Structure

```text
harmonix-ai-vocal-harmonizer/
│
├── src/
│   └── harmonix/
│       ├── config/
│       ├── domain/
│       ├── export/
│       ├── input/
│       ├── persistence/
│       ├── session_state/
│       ├── ui/
│       ├── upload/
│       │
│       ├── harmony_generator.py
│       ├── key_scale_resolver.py
│       ├── latency.py
│       ├── live_audio.py
│       ├── pitch_detector.py
│       ├── pitch_shifter.py
│       ├── playback_engine.py
│       └── visualizer_bridge.py
│
├── tests/
├── requirements.txt
├── bottleneck_test.py
├── m8_hardware_validation.py
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.12
- Windows (primary development environment)
- Microphone and audio output device for live audio functionality

### 1. Clone the repository

```bash
git clone https://github.com/PrarthanaRamesh/harmonix-ai-vocal-harmonizer.git

cd harmonix-ai-vocal-harmonizer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some audio-processing dependencies may have platform-specific installation requirements.

### 4. Configure the Python source path

The project currently uses a `src/` layout.

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

### 5. Launch the application

```bash
python -m harmonix
```

This launches the current Pygame application shell.

## Testing

Harmonix includes automated tests covering:

- Pitch detection
- Harmony generation
- Key and scale resolution
- Pitch shifting
- Live audio processing
- Playback
- Audio input handling
- Export and persistence
- Session state and UI components
- Latency measurement
- Hardware validation

Run the test suite using:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
```

### Latest recorded test results

| Metric | Result |
|---|---:|
| Tests passed | 357 |
| Subtests passed | 73 |
| Warnings | 4 |

These results are from the development environment. They do not yet establish compatibility across all operating systems or hardware configurations.

## Performance and Hardware Validation

Two dedicated scripts support performance investigation and hardware testing.

**`bottleneck_test.py`**

Compares processing time with and without input recording to investigate recording overhead.

**`m8_hardware_validation.py`**

Runs controlled validation using actual microphone input and audio output devices.

The hardware validation runner:
- Uses the real audio input path rather than injected synthetic samples.
- Supports configurable input and output devices.
- Records processing and queue health information.
- Separates physical capture-to-DAC latency from callback-to-DAC pipeline latency.
- Reports physical latency only when verified capture timestamps are available.

## Known Limitations

Harmonix is an experimental project under active development.

- Real-time playback can exhibit choppiness under certain conditions.
- Processing latency and playback stability require further optimization.
- Hardware behavior may vary across audio devices and systems.
- Cross-platform compatibility has not yet been established.
- The current desktop entry point launches a minimal application shell rather than a fully polished production interface.

## Future Improvements

- Improve real-time playback stability.
- Reduce audio-processing latency.
- Optimize pitch-shifting performance.
- Improve audio visualization.
- Expand hardware compatibility.
- Introduce more flexible harmony configurations.
- Refine the desktop user experience.

## About

Harmonix is an independent engineering project exploring real-time audio processing, music theory, and vocal harmonization.

**Built with Python, signal processing, and a love for music.**