from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_PATH / "data"

MUSICNET_ROOT = DATA_DIR / "musicnet" / "musicnet"

AUDIO_DIR = str(MUSICNET_ROOT)
METADATA_PATH = str(DATA_DIR / "musicnet_metadata.csv")
CACHE_DIR = DATA_DIR / "cache"
FEATURE_CACHE_DIR = DATA_DIR / "feature_cache"

RAW_MIDI_DIR = str(DATA_DIR / "musicnet_midis" / "musicnet_midis")

