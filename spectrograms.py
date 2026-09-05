import hashlib
import os

import librosa
import numpy as np
from tqdm import tqdm

from paths import CACHE_DIR
from config import SR, SEGMENT_DURATION, MIN_LAST_DURATION, N_FFT
from dataset import _find_audio_path

# lower resolution than the classical pipeline's HOP_LENGTH=512 - CPU training speed matters more
# here than fine time/frequency detail, and this is ~4x faster per sample to train on
N_MELS = 64
HOP_LENGTH = 1024


def extract_melspec(segment, sr):
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def make_spectrogram_dataset(
    metadata_split,
    audio_dir,
    sr=SR,
    segment_duration=SEGMENT_DURATION,
    min_last_duration=MIN_LAST_DURATION,
    use_cache=True
):
    ids = sorted(metadata_split["id"].astype(str).tolist())
    with open(__file__, "rb") as f:
        code_hash = hashlib.sha1(f.read()).hexdigest()[:8]
    key = f"melspec|{sr}|{segment_duration}|{min_last_duration}|{code_hash}|{','.join(ids)}"
    cache_path = CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"

    if use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"], cached["y"], cached["groups"]

    X = []
    y = []
    groups = []

    segment_length = sr * segment_duration
    min_last_length = sr * min_last_duration

    for _, row in tqdm(
        metadata_split.iterrows(),
        total=len(metadata_split),
        desc="Processing spectrograms"
    ):
        path = _find_audio_path(audio_dir, row["id"])
        audio, _ = librosa.load(path, sr=sr)

        for start in range(0, len(audio), segment_length):
            segment = audio[start:min(start + segment_length, len(audio))]

            if len(segment) < min_last_length:
                continue

            X.append(extract_melspec(segment, sr))
            y.append(row["composer"])
            groups.append(row["work_id"])

    X, y, groups = np.array(X), np.array(y), np.array(groups)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, groups=groups)

    return X, y, groups
