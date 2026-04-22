# InfraUltraScope UI  
**Infrasound & Ultrasound Visualizer + Audible Decoder**

InfraUltraScope is a real-time scientific visualization and signal-decoding application built with PyQt5.  
It captures audio signals across **infrasound (<20 Hz)** and **ultrasound (>20 kHz)** ranges, visualizes them in a live spectrogram, and **transposes inaudible frequencies into the audible range**.

Designed for researchers, hobbyists, and engineers working with acoustic signals beyond human hearing.

---

# Features

## Real-Time Audio Capture
- Live microphone input
- Adjustable sampling rate (up to 768 kHz supported)
- Device selection support
- Continuous buffered recording

## Spectrogram Visualization
- Full-spectrum real-time spectrogram
- Adjustable frequency display range
- Multiple scientific color maps:
  - inferno
  - viridis
  - plasma
  - magma
  - cividis
  - hot

## Frequency Band Analysis
- Select custom frequency bands
- Drag-select regions directly on spectrogram
- Separate display and analysis ranges

## Audible Frequency Transposition
Convert inaudible frequencies into audible sound using pitch shifting.

Useful for:
- Bat ultrasound detection
- Dog whistle monitoring
- Infrasound observation
- Mechanical diagnostics
- Environmental sensing

## Live Playback
- Real-time playback of transposed audio
- Adjustable volume control

## Recording Support
Save:

- Raw captured audio (.wav)
- Transposed audible output (.wav)

## Power Spectrum Analysis
- Real-time PSD (Power Spectral Density)
- Automatic peak detection
- RMS monitoring

## Offline Analysis Mode
Analyze existing WAV files without live input.

---

# Scientific Applications

InfraUltraScope can be used in:

- Bioacoustics (bat monitoring, insect tracking)
- Structural vibration monitoring
- Machinery diagnostics
- Atmospheric research
- Environmental sensing
- Animal behavior studies
- Acoustic research
- Paranormal / exploratory research (experimental)

---

# Installation

## Requirements

Python 3.9+

Install dependencies:

```bash
pip install numpy scipy sounddevice soundfile pyqt5 pyqtgraph matplotlib
```

---

# Hardware Requirements

## For Ultrasound (>20 kHz)

You need:

- Ultrasonic microphone
- Bat detector microphone
- ≥192 kHz sampling interface

Examples:

- Ultrasonic MEMS microphones
- Specialized bat detection mics
- High-sample-rate USB audio interface

---

## For Infrasound (<20 Hz)

You need:

- MEMS pressure sensor
- Infrasound microphone

Standard microphones usually **cannot detect true infrasound**.

---

# Quick Start

Run the application:

```bash
python infraultrascope.py
```

Click:

```
▶ Start capture
```

Then:

1. Select frequency band
2. Adjust transpose ratio
3. Enable live playback (optional)

---

# Command Line Usage

## List Audio Devices

```bash
python infraultrascope.py --devices
```

Example output:

```
[ 0] [IN] Built-in Microphone
[ 1] [OUT] Speakers
[ 2] [IN,OUT] USB Audio Interface
```

---

## Run with Custom Settings

```bash
python infraultrascope.py \
    --samplerate 192000 \
    --band-low 20000 \
    --band-high 80000 \
    --transpose-ratio 10 \
    --play-live
```

---

## Offline File Analysis

Analyze a WAV file:

```bash
python infraultrascope.py --file sample.wav
```

Optional:

Save transposed output:

```bash
python infraultrascope.py \
    --file ultrasound.wav \
    --band-low 20000 \
    --band-high 80000 \
    --transpose-ratio 10 \
    --save-transposed audible.wav
```

---

# GUI Overview

## Main Panels

### Spectrogram
Displays live frequency spectrum over time.

Supports:

- Drag-to-select bands
- Zoomed display regions
- Frequency masking

---

### Waveform Viewer

Shows:

- Live signal waveform
- Real-time amplitude behavior

---

### Power Spectrum (PSD)

Displays:

- Frequency intensity
- Peak frequency detection
- Energy distribution

---

### Control Panel

Includes:

- Display frequency sliders
- Analysis band sliders
- Transpose ratio control
- Preset buttons
- Playback controls
- Metrics readout

---

# Built-in Presets

## Bat Ultrasound

```
20 kHz – 80 kHz
Transpose ÷10
```

Use for:

- Bat detection
- Ultrasonic wildlife monitoring

---

## Dog Whistle

```
18 kHz – 25 kHz
Transpose ÷8
```

Use for:

- Dog whistle detection
- Ultrasonic signal study

---

## Infrasound

```
1 Hz – 19 Hz
Transpose ×50
```

Use for:

- Seismic activity
- Structural vibration
- Atmospheric pressure waves

---

# Core Signal Processing Pipeline

The system performs:

1. Audio Capture
2. Bandpass Filtering
3. Frequency Transposition
4. Normalization
5. Visualization
6. Playback / Recording

---

## Frequency Transposition

Transposition is performed via:

```
Resampling-based pitch shifting
```

Which shifts inaudible frequencies into audible range.

---

# File Outputs

## Raw Capture

Saved as:

```
raw_capture.wav
```

Contains:

- Original signal
- Full frequency content

---

## Transposed Audio

Saved as:

```
transposed_audible.wav
```

Contains:

- Filtered frequency band
- Audible-converted output

---

# Performance Notes

Recommended:

| Use Case | Sample Rate |
|----------|-------------|
| Infrasound | 48 kHz |
| Ultrasound | 192 kHz+ |
| High-frequency research | 384–768 kHz |

---

# Troubleshooting

## No Input Device Found

Run:

```bash
python infraultrascope.py --devices
```

Then specify:

```bash
--input-device INDEX
```

---

## No Audible Output

Check:

- Output device selection
- Volume slider
- Transpose ratio

---

## Spectrogram Not Updating

Possible causes:

- Incorrect sample rate
- Device unsupported
- Buffer overflow

Try lowering:

```bash
--block-size
```

---

# Project Architecture

```
infraultrascope.py
│
├── GUI (PyQt5)
├── Visualization (PyQtGraph)
├── Signal Processing (SciPy)
├── Audio IO (SoundDevice)
├── File Handling (SoundFile)
└── CLI Interface (argparse)
```

---

# Dependencies

Core:

- numpy
- scipy
- sounddevice
- soundfile
- pyqt5
- pyqtgraph
- matplotlib

---

# Use Cases

InfraUltraScope supports:

- Bat monitoring research
- Dog whistle decoding
- Mechanical vibration detection
- Industrial diagnostics
- Acoustic experiments
- Educational labs
- Field audio exploration

---

# Limitations

- Standard microphones cannot detect true ultrasound
- High sample rates require compatible hardware
- Real-time processing depends on CPU performance

---

# Future Improvements (Suggested)

Potential upgrades:

- GPU spectrogram acceleration
- Multi-channel input
- Data logging system
- Event detection AI
- Frequency tagging
- Remote streaming support

---

# License

Recommended:

```
MIT License
```



---

# Author

InfraUltraScope Project  
Scientific Audio Visualization Toolkit

---

# Disclaimer

This software is intended for:

- Research
- Educational
- Experimental use

Hardware limitations determine actual frequency detection capability.

Not intended for medical or safety-critical applications.

---

# Screenshots (Optional)

You may add:

```
docs/screenshot1.png
docs/screenshot2.png
```

Inside your repository.

---

# Contributing

Contributions welcome.

Suggested areas:

- Signal processing improvements
- UI enhancements
- Performance optimization
- Hardware integration

---

# Summary

InfraUltraScope provides:

- Real-time spectral visualization
- Inaudible signal decoding
- Scientific-grade acoustic analysis
- Flexible live and offline workflows

A powerful tool for exploring the **hidden acoustic world** beyond human hearing.