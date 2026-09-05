import hashlib
import os

import pandas as pd

from config import SR, SEGMENT_DURATION, MIN_LAST_DURATION, MIN_WORKS_PER_COMPOSER
from paths import CACHE_DIR, METADATA_PATH

AUDIO_SPLIT_DIRS = ("train_data", "test_data")


def filter_metadata(
    metadata,
    min_works_per_composer=MIN_WORKS_PER_COMPOSER,
    exclude_composers=None,
):
    """Return a filtered copy with a work_id for grouping movements of one work.

    Count distinct works (composer + composition), not recordings or movements.
    Keep composers with at least min_works_per_composer works, then optionally
    exclude named composers. Preserve the input's row order and index without
    modifying it. exclude_composers accepts a name or an iterable of names.
    """
    required_columns = {"id", "composer", "composition"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {', '.join(sorted(missing_columns))}")
    if min_works_per_composer < 1:
        raise ValueError("min_works_per_composer must be at least 1")

    metadata = metadata.copy()
    metadata["work_id"] = (
        metadata["composer"].astype(str)
        + " | "
        + metadata["composition"].astype(str)
    )

    work_counts = metadata.groupby("composer")["work_id"].nunique()
    valid_composers = work_counts[work_counts >= min_works_per_composer].index
    keep = metadata["composer"].isin(valid_composers)

    if exclude_composers is not None:
        if isinstance(exclude_composers, str):
            exclude_composers = [exclude_composers]
        keep &= ~metadata["composer"].isin(exclude_composers)

    return metadata.loc[keep].copy()


def load_metadata(
    metadata_path=METADATA_PATH,
    *,
    min_works_per_composer=MIN_WORKS_PER_COMPOSER,
    exclude_composers=None,
):
    """Read MusicNet metadata and apply filter_metadata, ready for dataset builders.

    Uses paths.METADATA_PATH and config.MIN_WORKS_PER_COMPOSER by default.
    For an already loaded DataFrame, call filter_metadata directly.
    """
    return filter_metadata(
        pd.read_csv(metadata_path),
        min_works_per_composer=min_works_per_composer,
        exclude_composers=exclude_composers,
    )


def make_work_level_dataset(X, y, groups):
    """Aggregate segment features into one mean/std feature vector per work.

    Return (X_work, y_work, work_ids), ordered by work ID.
    """
    import numpy as np

    X_work = []
    y_work = []
    work_ids = []

    for work_id in np.unique(groups):
        mask = groups == work_id
        X_segments = X[mask]
        y_segments = y[mask]

        mean_features = X_segments.mean(axis=0)
        std_features = X_segments.std(axis=0)
        work_features = np.concatenate([mean_features, std_features])

        X_work.append(work_features)
        y_work.append(y_segments[0])
        work_ids.append(work_id)

    return np.array(X_work), np.array(y_work), np.array(work_ids)


def _feature_engineering_version():
    """Hash of feature_engineering.py so a cache is auto-invalidated when it changes."""
    import feature_engineering

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

    # Keep audio dependencies optional when only loading or filtering metadata.
    import librosa
    import numpy as np
    from tqdm import tqdm

    from feature_engineering import extract_features

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
