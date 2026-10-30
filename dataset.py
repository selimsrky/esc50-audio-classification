import os
import zipfile
import requests
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

DATA_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
DATA_DIR = "data"

def download_and_extract_esc50():
    """Downloads and extracts ESC-50 dataset if not already present."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    zip_path = os.path.join(DATA_DIR, "ESC-50-master.zip")
    extracted_dir = os.path.join(DATA_DIR, "ESC-50-master")

    if not os.path.exists(extracted_dir):
        print("Downloading ESC-50 dataset (~600MB)...")
        response = requests.get(DATA_URL, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("Extracting dataset archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path)
        print("Dataset ready.")

class ESC50Dataset(Dataset):
    def __init__(self, meta_df, base_audio_dir, sample_rate=22050, n_mels=64, duration=5):
        self.meta_df = meta_df.reset_index(drop=True)
        self.base_audio_dir = base_audio_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = sample_rate * duration

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        file_path = os.path.join(self.base_audio_dir, row['filename'])
        label = row['target']

        # 1. Load raw audio
        y, sr = librosa.load(file_path, sr=self.sample_rate)

        # 2. Pad or truncate to exact target length (5 seconds)
        if len(y) < self.target_length:
            y = np.pad(y, (0, self.target_length - len(y)))
        else:
            y = y[:self.target_length]

        # 3. Compute Log Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=512
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 4. Shape to (Channels, Mel-Bands, Time-Steps) -> (1, 64, T)
        spec_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spec_tensor, label_tensor

if __name__ == "__main__":
    download_and_extract_esc50()
