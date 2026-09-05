import librosa
import numpy as np

from src.config import N_FFT, HOP_LENGTH, N_MFCC

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def extract_power(segment):
    n_fft = N_FFT
    hop_length = HOP_LENGTH

    stft = librosa.stft(
        segment,
        n_fft=n_fft,
        hop_length=hop_length
    )

    magnitude = np.abs(stft)
    power = magnitude ** 2
    return magnitude, power


def extract_mfcc_matrix(power, sr):
    mel = librosa.feature.melspectrogram(
        S=power,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mfcc = librosa.feature.mfcc(
        S=mel_db,
        sr=sr,
        n_mfcc=N_MFCC
    )

    return mfcc


def extract_mfcc(power, sr):

    mfcc = extract_mfcc_matrix(power, sr)
    return np.mean(mfcc, axis=1), np.std(mfcc, axis=1)


def extract_mfcc_delta(mfcc):

    delta = librosa.feature.delta(mfcc, order=1)
    return np.mean(delta, axis=1), np.std(delta, axis=1)


def extract_chroma(power, sr):

    chroma = librosa.feature.chroma_stft(
        S=power,
        sr=sr
    )
    return chroma, np.mean(chroma, axis=1), np.std(chroma, axis=1)


def extract_tonnetz(chroma, sr):

    tonnetz = librosa.feature.tonnetz(
        chroma=chroma,
        sr=sr
    )
    return np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)


def extract_spectral_centroid(magnitude, sr):

    centroid = librosa.feature.spectral_centroid(
        S=magnitude,
        sr=sr
    )
    return np.mean(centroid), np.std(centroid)

def extract_spectral_bandwidth(magnitude, sr):

    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude,
        sr=sr
    )
    return np.mean(bandwidth), np.std(bandwidth)

def extract_spectral_rollof(magnitude, sr):

    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sr
    )
    return np.mean(rolloff), np.std(rolloff)


def extract_spectral_contrast(magnitude, sr):

    contrast = librosa.feature.spectral_contrast(
        S=magnitude,
        sr=sr
    )
    return np.mean(contrast, axis=1), np.std(contrast, axis=1)


def extract_spectral_flatness(magnitude):
    """How noise-like (flat) vs. tone-like the spectrum is."""
    flatness = librosa.feature.spectral_flatness(
        S=magnitude
    )
    return np.mean(flatness), np.std(flatness)


def extract_zero_crossing_rate(segment):

    zcr = librosa.feature.zero_crossing_rate(
        segment,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )
    return np.mean(zcr), np.std(zcr)


def extract_tempo_onset(segment, sr):

    onset_env = librosa.onset.onset_strength(
        y=segment,
        sr=sr,
        hop_length=HOP_LENGTH
    )
    tempo = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH
    )[0]
    return tempo, np.mean(onset_env), np.std(onset_env)


def extract_harmonic_percussive_ratio(segment):

    harmonic, percussive = librosa.effects.hpss(segment)

    harmonic_rms = np.sqrt(np.mean(harmonic ** 2))
    percussive_rms = np.sqrt(np.mean(percussive ** 2))

    harmonic_ratio = harmonic_rms / (harmonic_rms + percussive_rms + 1e-10)

    return harmonic_ratio, percussive_rms


def extract_rms(magnitude):

    rms = librosa.feature.rms(
        S=magnitude
    )
    return np.mean(rms), np.std(rms)


N_MFCC_DELTA = 8


# which context keys each group needs
CONTEXT_REQUIREMENTS = {
    "mfcc": {"mfcc_matrix"},
    "mfcc_delta": {"mfcc_matrix"},
    "chroma": {"chroma"},
    "tonnetz": {"chroma"},
    "centroid": {"magnitude"},
    "bandwidth": {"magnitude"},
    "rolloff": {"magnitude"},
    "contrast": {"magnitude"},
    "flatness": {"magnitude"},
    "zcr": set(),
    "rms": {"magnitude"},
    "tempo_onset": set(),
    "harmonic_percussive": set(),
}


def _build_context(segment, sr, needed_groups=None):
    """Shared intermediates (STFT/MFCC matrix/chroma), computed lazily: only what
    needed_groups actually requires"""
    if needed_groups is None:
        needed_groups = FEATURE_REGISTRY.keys()

    needed_keys = set()
    for group in needed_groups:
        needed_keys |= CONTEXT_REQUIREMENTS.get(group, set())

    ctx = {"segment": segment, "sr": sr}

    if needed_keys & {"magnitude", "mfcc_matrix", "chroma"}:
        magnitude, power = extract_power(segment)
        ctx["magnitude"] = magnitude
        ctx["power"] = power

    if "mfcc_matrix" in needed_keys:
        ctx["mfcc_matrix"] = extract_mfcc_matrix(ctx["power"], sr)

    if "chroma" in needed_keys:
        chroma, _, _ = extract_chroma(ctx["power"], sr)
        ctx["chroma"] = chroma

    return ctx


def _mfcc_values(ctx):
    return np.concatenate([np.mean(ctx["mfcc_matrix"], axis=1), np.std(ctx["mfcc_matrix"], axis=1)])


def _mfcc_names():
    return [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)]


def _mfcc_delta_values(ctx):
    mean, std = extract_mfcc_delta(ctx["mfcc_matrix"])
    return np.concatenate([mean[:N_MFCC_DELTA], std[:N_MFCC_DELTA]])


def _mfcc_delta_names():
    return [f"mfcc{i}_delta_mean" for i in range(N_MFCC_DELTA)] + [f"mfcc{i}_delta_std" for i in range(N_MFCC_DELTA)]


def _chroma_values(ctx):
    return np.concatenate([np.mean(ctx["chroma"], axis=1), np.std(ctx["chroma"], axis=1)])


def _chroma_names():
    return [f"chroma_{p}_mean" for p in PITCH_CLASSES] + [f"chroma_{p}_std" for p in PITCH_CLASSES]


def _tonnetz_values(ctx):
    mean, std = extract_tonnetz(ctx["chroma"], ctx["sr"])
    return np.concatenate([mean, std])


def _tonnetz_names():
    return [f"tonnetz{i}_mean" for i in range(6)] + [f"tonnetz{i}_std" for i in range(6)]


def _centroid_values(ctx):
    return np.array(extract_spectral_centroid(ctx["magnitude"], ctx["sr"]))


def _bandwidth_values(ctx):
    return np.array(extract_spectral_bandwidth(ctx["magnitude"], ctx["sr"]))


def _rolloff_values(ctx):
    return np.array(extract_spectral_rollof(ctx["magnitude"], ctx["sr"]))


def _contrast_values(ctx):
    mean, std = extract_spectral_contrast(ctx["magnitude"], ctx["sr"])
    return np.concatenate([mean, std])


def _contrast_names():
    return [f"contrast_band{i}_mean" for i in range(7)] + [f"contrast_band{i}_std" for i in range(7)]


def _flatness_values(ctx):
    return np.array(extract_spectral_flatness(ctx["magnitude"]))


def _zcr_values(ctx):
    return np.array(extract_zero_crossing_rate(ctx["segment"]))


def _rms_values(ctx):
    return np.array(extract_rms(ctx["magnitude"]))


def _tempo_onset_values(ctx):
    return np.array(extract_tempo_onset(ctx["segment"], ctx["sr"]))


def _harmonic_percussive_values(ctx):
    return np.array(extract_harmonic_percussive_ratio(ctx["segment"]))


# one row per feature group: (values_fn, names_fn)
FEATURE_REGISTRY = {
    "mfcc": (_mfcc_values, _mfcc_names),
    "mfcc_delta": (_mfcc_delta_values, _mfcc_delta_names),
    # feature importance review: chroma/tonnetz/flatness had ~0 individual features in the
    # top-25 and the lowest per-dimension importance of all groups - disabled.
    # "chroma": (_chroma_values, _chroma_names),
    # "tonnetz": (_tonnetz_values, _tonnetz_names),
    "centroid": (_centroid_values, lambda: ["centroid_mean", "centroid_std"]),
    "bandwidth": (_bandwidth_values, lambda: ["bandwidth_mean", "bandwidth_std"]),
    "rolloff": (_rolloff_values, lambda: ["rolloff_mean", "rolloff_std"]),
    "contrast": (_contrast_values, _contrast_names),
    # "flatness": (_flatness_values, lambda: ["flatness_mean", "flatness_std"]),
    "zcr": (_zcr_values, lambda: ["zcr_mean", "zcr_std"]),
    "rms": (_rms_values, lambda: ["rms_mean", "rms_std"]),
    "tempo_onset": (_tempo_onset_values, lambda: ["tempo", "onset_mean", "onset_std"]),
    "harmonic_percussive": (_harmonic_percussive_values, lambda: ["harmonic_ratio", "percussive_rms"]),
}


# every function each group's value transitively depends on - a change to ANY of them must
# invalidate that group's cache (e.g. tonnetz depends on extract_chroma, not just extract_tonnetz)
GROUP_DEPENDENCIES = {
    "mfcc": (extract_power, extract_mfcc_matrix, _mfcc_values, N_MFCC),
    "mfcc_delta": (extract_power, extract_mfcc_matrix, extract_mfcc_delta, _mfcc_delta_values, N_MFCC_DELTA),
    "chroma": (extract_power, extract_chroma, _chroma_values),
    "tonnetz": (extract_power, extract_chroma, extract_tonnetz, _tonnetz_values),
    "centroid": (extract_power, extract_spectral_centroid, _centroid_values),
    "bandwidth": (extract_power, extract_spectral_bandwidth, _bandwidth_values),
    "rolloff": (extract_power, extract_spectral_rollof, _rolloff_values),
    "contrast": (extract_power, extract_spectral_contrast, _contrast_values),
    "flatness": (extract_power, extract_spectral_flatness, _flatness_values),
    "zcr": (extract_zero_crossing_rate, _zcr_values),
    "rms": (extract_power, extract_rms, _rms_values),
    "tempo_onset": (extract_tempo_onset, _tempo_onset_values),
    "harmonic_percussive": (extract_harmonic_percussive_ratio, _harmonic_percussive_values),
}


def extract_feature_group(group_name, segment, sr, ctx=None):
    """Compute a single registered group's values for one segment."""
    values_fn, _ = FEATURE_REGISTRY[group_name]

    if ctx is None:
        ctx = _build_context(segment, sr, needed_groups=[group_name])

    return values_fn(ctx)


def extract_features(segment, sr):
    """Builds the full feature vector for one audio segment, via FEATURE_REGISTRY."""
    ctx = _build_context(segment, sr)

    return np.concatenate(
        [values_fn(ctx) for values_fn, _ in FEATURE_REGISTRY.values()]
    ).astype(np.float32)


def feature_names():
    """Names matching extract_features()'s order - derived from the same FEATURE_REGISTRY,
    so the two can't drift out of sync."""
    names = []

    for _, names_fn in FEATURE_REGISTRY.values():
        names.extend(names_fn())

    return names