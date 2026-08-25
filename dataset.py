import hashlib
import os

import librosa
import numpy as np
from tqdm import tqdm

from config import SR, SEGMENT_DURATION, MIN_LAST_DURATION
from paths import CACHE_DIR
import feature_engineering
from feature_engineering import extract_features

AUDIO_SPLIT_DIRS = ("train_data", "test_data")


def _feature_engineering_version():
    """Hash of feature_engineering.py so a cache is auto-invalidated when it changes."""
    with open(feature_engineering.__file__, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]


def _find_audio_path(audio_dir, composition_id):
    """MusicNet splits recordings across train_data/ and test_data/ subfolders."""
    for split_dir in AUDIO_SPLIT_DIRS:
        path = os.path.join(audio_dir, split_dir, f"{composition_id}.wav")

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No audio file for composition_id={composition_id} under {audio_dir} "
        f"(looked in {AUDIO_SPLIT_DIRS})"
    )


def _cache_path(metadata_split, sr, segment_duration, min_last_duration):
    composition_ids = sorted(metadata_split["id"].astype(str).tolist())

    key = "|".join([
        str(sr),
        str(segment_duration),
        str(min_last_duration),
        _feature_engineering_version(),
        ",".join(composition_ids)
    ])

    digest = hashlib.sha1(key.encode()).hexdigest()[:16]

    return CACHE_DIR / f"dataset_{digest}.npz"


def make_dataset(
    metadata_split,
    audio_dir,
    sr=SR,
    segment_duration=SEGMENT_DURATION,
    min_last_duration=MIN_LAST_DURATION,
    use_cache=True,
    cache_path=None
):
    """Same as before, but caches (X, y, groups) to disk so repeated runs with the same
    metadata split, segment settings and feature_engineering.py content skip recomputation."""

    if cache_path is None:
        cache_path = _cache_path(
            metadata_split,
            sr,
            segment_duration,
            min_last_duration
        )

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
        desc="Processing compositions"
    ):
        composition_id = row["id"]
        composer = row["composer"]
        work_id = row["work_id"]

        path = _find_audio_path(audio_dir, composition_id)

        audio, _ = librosa.load(path, sr=sr)

        for start in range(0, len(audio), segment_length):
            segment = audio[start: min(start + segment_length, len(audio))]

            if len(segment) < min_last_length:
                continue

            features = extract_features(
                segment,
                sr
            )

            X.append(features)
            y.append(composer)
            groups.append(work_id)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, X=X, y=y, groups=groups)

    return X, y, groups
