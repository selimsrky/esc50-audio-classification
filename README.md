# ESC-50 Audio Classification with PyTorch & Librosa

An end-to-end deep learning pipeline for environmental sound classification on the ESC-50 dataset. Audio waveforms are transformed into Log Mel-Spectrograms using Librosa and classified using a custom 2D Convolutional Neural Network (CNN) built with PyTorch.

## Architecture & Pipeline
1. **Digital Signal Processing (DSP):** Audio files are normalized to 22.05 kHz and converted to 64-band Mel-Spectrograms via Short-Time Fourier Transform (STFT).
2. **Deep Learning Model:** 3-block 2D CNN with Batch Normalization, Max Pooling, and Adaptive Global Pooling.
3. **Cross-Validation:** Evaluated on the standard 5-fold cross-validation scheme (Fold 5 test split).

## Tech Stack
- **Python 3.10**
- **PyTorch** (Deep learning model training and inference)
- **Librosa** (Digital signal processing and spectrogram generation)
- **Docker** (Isolated container execution)

## Quickstart

### Native Setup
```bash
pip install -r requirements.txt
python train.py
