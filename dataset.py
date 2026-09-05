import hashlib
import os
import logging
import time

import librosa
import numpy as np
from tqdm import tqdm

from config import SR, SEGMENT_DURATION, MIN_LAST_DURATION
from paths import CACHE_DIR, FEATURE_CACHE_DIR
from cache_hash import group_hash
import feature_engineering

logger = logging.getLogger(__file__)


AUDIO_SPLIT_DIRS = ("train_data", "test_data")


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


def _load_audio_cached(path, composition_id, sr, use_cache=True):
    """Caches the decoded waveform itself, per (composition_id, sr) - so re-running feature
    extraction (e.g. to fill in one missing feature group) doesn't need to re-decode audio."""
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
    """Caches each feature GROUP separately under FEATURE_CACHE_DIR (one .npz per group,
    keyed by a hash of only that group's dependency functions). Adding, removing or editing
    one group in feature_engineering.FEATURE_REGISTRY only recomputes that group - audio is
    decoded (from the audio cache) only if at least one group is missing."""

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
