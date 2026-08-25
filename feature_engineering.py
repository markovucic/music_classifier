import librosa
import numpy as np

from config import N_FFT, HOP_LENGTH, N_MFCC

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
    """Raw MFCC matrix (n_mfcc x frames), shared by the mean/std and delta features."""
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
    """Extracts MFCC and returns tuple (mfcc mean, mfcc std deviation)"""
    mfcc = extract_mfcc_matrix(power, sr)

    return np.mean(mfcc, axis=1), np.std(mfcc, axis=1)


def extract_mfcc_delta(mfcc):
    """First-order MFCC delta (rate of timbre change) mean/std."""
    delta = librosa.feature.delta(mfcc, order=1)

    return np.mean(delta, axis=1), np.std(delta, axis=1)


def extract_chroma(power, sr):

    chroma = librosa.feature.chroma_stft(
        S=power,
        sr=sr
    )

    return chroma, np.mean(chroma, axis=1), np.std(chroma, axis=1)


def extract_tonnetz(chroma, sr):
    """Harmonic-relation (tonnetz) mean/std, derived from an already-computed chroma."""
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
    """Peak-vs-valley contrast per frequency band (mean+std per band)."""
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
    """How often the raw waveform changes sign — roughness/percussiveness."""
    zcr = librosa.feature.zero_crossing_rate(
        segment,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    return np.mean(zcr), np.std(zcr)


def extract_tempo_onset(segment, sr):
    """Estimated tempo plus mean/std of the onset-strength envelope (rhythmic density)."""
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
    """Share of energy that is harmonic (pitched/legato) vs. percussive (attack-like)."""
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


def extract_features(segment, sr):
    """Builds the full feature vector for one audio segment. Order must match feature_names()."""
    features = []

    magnitude, power = extract_power(segment)

    mfcc = extract_mfcc_matrix(power, sr)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    mfcc_delta_mean, mfcc_delta_std = extract_mfcc_delta(mfcc)
    features.extend(mfcc_delta_mean)
    features.extend(mfcc_delta_std)

    chroma, chroma_mean, chroma_std = extract_chroma(power, sr)
    features.extend(chroma_mean)
    features.extend(chroma_std)

    tonnetz_mean, tonnetz_std = extract_tonnetz(chroma, sr)
    features.extend(tonnetz_mean)
    features.extend(tonnetz_std)

    centroid_mean, centroid_std = extract_spectral_centroid(magnitude, sr)
    features.append(centroid_mean)
    features.append(centroid_std)

    bandwidth_mean, bandwidth_std = extract_spectral_bandwidth(magnitude, sr)
    features.append(bandwidth_mean)
    features.append(bandwidth_std)

    rolloff_mean, rolloff_std = extract_spectral_rollof(magnitude, sr)
    features.append(rolloff_mean)
    features.append(rolloff_std)

    contrast_mean, contrast_std = extract_spectral_contrast(magnitude, sr)
    features.extend(contrast_mean)
    features.extend(contrast_std)

    flatness_mean, flatness_std = extract_spectral_flatness(magnitude)
    features.append(flatness_mean)
    features.append(flatness_std)

    zcr_mean, zcr_std = extract_zero_crossing_rate(segment)
    features.append(zcr_mean)
    features.append(zcr_std)

    rms_mean, rms_std = extract_rms(magnitude)
    features.append(rms_mean)
    features.append(rms_std)

    tempo, onset_mean, onset_std = extract_tempo_onset(segment, sr)
    features.append(tempo)
    features.append(onset_mean)
    features.append(onset_std)

    harmonic_ratio, percussive_rms = extract_harmonic_percussive_ratio(segment)
    features.append(harmonic_ratio)
    features.append(percussive_rms)

    return np.array(features, dtype=np.float32)


def feature_names():
    """Names matching the order extract_features() builds its vector in — used to label
    feature-importance plots etc."""
    names = []

    names += [f"mfcc{i}_mean" for i in range(N_MFCC)]
    names += [f"mfcc{i}_std" for i in range(N_MFCC)]

    names += [f"mfcc{i}_delta_mean" for i in range(N_MFCC)]
    names += [f"mfcc{i}_delta_std" for i in range(N_MFCC)]

    names += [f"chroma_{p}_mean" for p in PITCH_CLASSES]
    names += [f"chroma_{p}_std" for p in PITCH_CLASSES]

    names += [f"tonnetz{i}_mean" for i in range(6)]
    names += [f"tonnetz{i}_std" for i in range(6)]

    names += ["centroid_mean", "centroid_std"]
    names += ["bandwidth_mean", "bandwidth_std"]
    names += ["rolloff_mean", "rolloff_std"]

    names += [f"contrast_band{i}_mean" for i in range(7)]
    names += [f"contrast_band{i}_std" for i in range(7)]

    names += ["flatness_mean", "flatness_std"]
    names += ["zcr_mean", "zcr_std"]
    names += ["rms_mean", "rms_std"]

    names += ["tempo", "onset_mean", "onset_std"]
    names += ["harmonic_ratio", "percussive_rms"]

    return names
