import hashlib
import os

import librosa
import numpy as np
from tqdm import tqdm

from src.utils.paths import CACHE_DIR
from src.config import SEGMENT_DURATION, MIN_LAST_DURATION
from src.features.dataset import _find_audio_path

PANNS_SR = 32000  

_model = None


def _get_model():
    global _model
    if _model is None:
        from panns_inference import AudioTagging
        _model = AudioTagging(checkpoint_path=None, device="cpu")  
    return _model


def extract_panns_embedding(segment):
    model = _get_model()
    _, embedding = model.inference(segment[None, :].astype(np.float32))
    return embedding[0].astype(np.float32)


def make_panns_dataset(
    metadata_split,
    audio_dir,
    sr=PANNS_SR,
    segment_duration=SEGMENT_DURATION,
    min_last_duration=MIN_LAST_DURATION,
    use_cache=True
):
    ids = sorted(metadata_split["id"].astype(str).tolist())
    with open(__file__, "rb") as f:
        code_hash = hashlib.sha1(f.read()).hexdigest()[:8]
    key = f"panns|{sr}|{segment_duration}|{min_last_duration}|{code_hash}|{','.join(ids)}"
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
        desc="Processing PANNs embeddings"
    ):
        path = _find_audio_path(audio_dir, row["id"])
        audio, _ = librosa.load(path, sr=sr)

        for start in range(0, len(audio), segment_length):
            segment = audio[start:min(start + segment_length, len(audio))]

            if len(segment) < min_last_length:
                continue

            X.append(extract_panns_embedding(segment))
            y.append(row["composer"])
            groups.append(row["work_id"])

    X, y, groups = np.array(X), np.array(y), np.array(groups)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, groups=groups)

    return X, y, groups
