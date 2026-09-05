import hashlib
import os
import logging
import time

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import SR, SEGMENT_DURATION, MIN_LAST_DURATION, MIN_WORKS_PER_COMPOSER
from src.utils.paths import CACHE_DIR, FEATURE_CACHE_DIR, METADATA_PATH
from src.utils.cache_hash import group_hash
from src.features import feature_engineering

logger = logging.getLogger(__file__)


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
    """Read MusicNet metadata and apply filter_metadata, ready for dataset builders."""
    return filter_metadata(
        pd.read_csv(metadata_path),
        min_works_per_composer=min_works_per_composer,
        exclude_composers=exclude_composers,
    )


def make_work_level_dataset(X, y, groups):
    """Aggregate segment features into one mean/std feature vector per work.

    Return (X_work, y_work, work_ids), ordered by work ID.
    """
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


def _find_audio_path(audio_dir, composition_id):
    """
    MusicNet recordings are split across train_data/ and test_data/ folders.
    Since data quantity is low - we use cross validation so we don't care for the splits,
    we collect the audio files from both sources and delay the split.
    """
    for split_dir in AUDIO_SPLIT_DIRS:
        path = os.path.join(audio_dir, split_dir, f"{composition_id}.wav")

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No audio file for composition_id={composition_id} under {audio_dir} "
        f"(looked in {AUDIO_SPLIT_DIRS})"
    )


def _load_audio_cached(path, composition_id, sr, use_cache=True):
    """Caches the decoded waveform itself, per (composition_id, sr) - 
    re-running feature extraction doesn't waste time on re-decoding of audio."""
    cache_path = CACHE_DIR / f"audio_{composition_id}_{sr}.npy"

    if use_cache and os.path.exists(cache_path):
        logger.debug(f"Cache hit for {composition_id}. Loading it...")
        return np.load(cache_path)

    logger.debug(f"Cache miss for {composition_id}. Loading it from .vaw...")
    audio, _ = librosa.load(path, sr=sr)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_path, audio)

    logger.debug(f"Composition {composition_id} loaded and saved to cache.")
    return audio


def make_dataset(
    metadata_split,
    audio_dir,
    sr=SR,
    segment_duration=SEGMENT_DURATION,
    min_last_duration=MIN_LAST_DURATION,
    use_cache=True
):
    """
    Caches each feature group separately.
    Adding, removing or editing one group in feature_engineering.FEATURE_REGISTRY 
    only recomputes that group
    """
    ids = sorted(metadata_split["id"].astype(str).tolist())
    dataset_id = hashlib.sha1(
        f"{sr}|{segment_duration}|{min_last_duration}|{','.join(ids)}".encode()
    ).hexdigest()[:12]

    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

    group_paths = {}
    for name in feature_engineering.FEATURE_REGISTRY:
        h = group_hash(*feature_engineering.GROUP_DEPENDENCIES[name])
        group_paths[name] = FEATURE_CACHE_DIR / f"feature_{name}_v_{h}_{dataset_id}.npz"

    meta_path = FEATURE_CACHE_DIR / f"meta_{dataset_id}.npz"

    cached_groups = {}
    missing_groups = []

    for name, path in group_paths.items():
        if use_cache and os.path.exists(path):
            logger.info(f"Cache hit for feature group {name}. Loading cached values...")
            cached_groups[name] = np.load(path)["X"]
        else:
            logger.info(f"Cache miss from feature group {name}. Features will be regenerated...")
            missing_groups.append(name)

    if not missing_groups and use_cache and os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        X = np.concatenate(
            [cached_groups[name] for name in feature_engineering.FEATURE_REGISTRY], axis=1
        )
        logger.info(f"No cache misses. Loaded all data")
        return X, meta["y"], meta["groups"]

    computed = {name: [] for name in missing_groups}
    computing_time = {name: 0 for name in missing_groups}
    context_time = 0
 
    y = []
    groups = []

    segment_length = sr * segment_duration
    min_last_length = sr * min_last_duration

    for _, row in tqdm(
        metadata_split.iterrows(),
        total=len(metadata_split),
        desc="Processing (grouped)"
    ):
        path = _find_audio_path(audio_dir, row["id"])
        audio = _load_audio_cached(path, row["id"], sr, use_cache=use_cache)

        for start in range(0, len(audio), segment_length):
            segment = audio[start: min(start + segment_length, len(audio))]

            if len(segment) < min_last_length:
                continue

            if missing_groups:
                start_time_ = time.time()
                ctx = feature_engineering._build_context(segment, sr, needed_groups=missing_groups)
                end_time_ = time.time()
                context_time += end_time_ - start_time_

                for name in missing_groups:
                    start_time_ = time.time()
                    values_fn, _ = feature_engineering.FEATURE_REGISTRY[name]
                    computed[name].append(values_fn(ctx))
                    end_time_ = time.time()
                    computing_time[name] += end_time_ - start_time_

            y.append(row["composer"])
            groups.append(row["work_id"])

    logger.info(f"Total time for context build: {context_time:.2f} seconds")

    y = np.array(y)
    groups = np.array(groups)

    for name in missing_groups:
        arr = np.array(computed[name], dtype=np.float32)
        cached_groups[name] = arr

        if use_cache:
            logger.info(f"Saved cache for feature group {name} | Total: {computing_time[name]:.2f} seconds")
            np.savez(group_paths[name], X=arr)

    if use_cache:
        np.savez(meta_path, y=y, groups=groups)

    X = np.concatenate(
        [cached_groups[name] for name in feature_engineering.FEATURE_REGISTRY], axis=1
    )

    return X, y, groups
