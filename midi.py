import hashlib
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import CACHE_DIR
from config import SR, SEGMENT_DURATION, MIN_LAST_DURATION

LABEL_SPLIT_DIRS = ("train_labels", "test_labels")

# fixed GM instrument program ids present in MusicNet
INSTRUMENTS = [1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74]


def _find_labels_path(labels_dir, composition_id):
    for split_dir in LABEL_SPLIT_DIRS:
        path = os.path.join(labels_dir, split_dir, f"{composition_id}.csv")
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No label file for composition_id={composition_id} under {labels_dir}")


def midi_feature_names():
    names = [
        "note_density", "pitch_mean", "pitch_std", "pitch_range",
        "interval_mean", "interval_std", "interval_abs_mean",
        "prop_ascending", "prop_descending", "prop_repeated",
        "prop_interval_unison", "prop_interval_step", "prop_interval_third", "prop_interval_leap",
        "ioi_mean", "ioi_std",
        "duration_beats_mean", "duration_beats_std",
        "duration_samples_mean", "duration_samples_std",
        "n_instruments", "mean_chord_size", "max_chord_size", "prop_monophonic",
        "chord_spread_mean", "chord_spread_std",
        "on_beat_ratio",
        "prop_register_low", "prop_register_mid", "prop_register_high",
    ]
    names += [f"pitch_class_{i}" for i in range(12)]
    names += [f"instrument_{i}" for i in INSTRUMENTS]
    return names


def extract_midi_features(notes, window_seconds=SEGMENT_DURATION):
    notes = notes.sort_values("start_time")
    pitches = notes["note"].to_numpy()

    # melodic intervals need time order, not pitch order
    intervals = np.diff(pitches)
    abs_intervals = np.abs(intervals)

    ioi = np.diff(notes["start_time"].to_numpy()) / SR

    duration_beats = notes["end_beat"].to_numpy()  # end_beat is actually note duration in beats, not a position
    duration_samples = (notes["end_time"] - notes["start_time"]).to_numpy()

    pitch_class_hist = np.bincount(pitches % 12, minlength=12) / len(pitches)

    instrument_counts = notes["instrument"].value_counts()
    instrument_hist = np.array([instrument_counts.get(i, 0) for i in INSTRUMENTS]) / len(notes)

    chords = notes.groupby("start_time")["note"]
    chord_sizes = chords.size().to_numpy()
    chord_spread = (chords.max() - chords.min()).to_numpy()

    on_beat_ratio = np.mean(np.abs(notes["start_beat"] % 1.0) < 0.05)
    register = np.digitize(pitches, bins=[55, 72])  # 0=low, 1=mid, 2=high

    features = [
        len(notes) / window_seconds,
        np.mean(pitches), np.std(pitches),
        pitches.max() - pitches.min(),
        np.mean(intervals) if len(intervals) else 0.0,
        np.std(intervals) if len(intervals) else 0.0,
        np.mean(abs_intervals) if len(intervals) else 0.0,
        np.mean(intervals > 0) if len(intervals) else 0.0,
        np.mean(intervals < 0) if len(intervals) else 0.0,
        np.mean(intervals == 0) if len(intervals) else 0.0,
        np.mean(abs_intervals == 0) if len(intervals) else 0.0,
        np.mean((abs_intervals >= 1) & (abs_intervals <= 2)) if len(intervals) else 0.0,
        np.mean((abs_intervals >= 3) & (abs_intervals <= 4)) if len(intervals) else 0.0,
        np.mean(abs_intervals >= 5) if len(intervals) else 0.0,
        np.mean(ioi) if len(ioi) else 0.0,
        np.std(ioi) if len(ioi) else 0.0,
        np.mean(duration_beats), np.std(duration_beats),
        np.mean(duration_samples), np.std(duration_samples),
        notes["instrument"].nunique(),
        np.mean(chord_sizes), np.max(chord_sizes), np.mean(chord_sizes == 1),
        np.mean(chord_spread), np.std(chord_spread),
        on_beat_ratio,
        np.mean(register == 0), np.mean(register == 1), np.mean(register == 2),
    ]
    features.extend(pitch_class_hist)
    features.extend(instrument_hist)

    return np.array(features, dtype=np.float32)


def make_midi_dataset(
    metadata_split,
    labels_dir,
    sr=SR,
    segment_duration=SEGMENT_DURATION,
    min_last_duration=MIN_LAST_DURATION,
    use_cache=True
):
    ids = sorted(metadata_split["id"].astype(str).tolist())
    with open(__file__, "rb") as f:
        code_hash = hashlib.sha1(f.read()).hexdigest()[:8]
    key = f"midi|{sr}|{segment_duration}|{min_last_duration}|{code_hash}|{','.join(ids)}"
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
        desc="Processing MIDI labels"
    ):
        notes = pd.read_csv(_find_labels_path(labels_dir, row["id"]))

        total_length = int(row["seconds"] * sr)

        for start in range(0, total_length, segment_length):
            end = min(start + segment_length, total_length)

            if end - start < min_last_length:
                continue

            segment_notes = notes[
                (notes["start_time"] >= start) & (notes["start_time"] < end)
            ]

            if len(segment_notes) < 2:
                continue

            X.append(extract_midi_features(segment_notes, window_seconds=(end - start) / sr))
            y.append(row["composer"])
            groups.append(row["work_id"])

    X, y, groups = np.array(X), np.array(y), np.array(groups)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, X=X, y=y, groups=groups)

    return X, y, groups
