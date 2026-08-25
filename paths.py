from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent
DATA_DIR = ROOT_PATH / "data"

MUSICNET_ROOT = DATA_DIR / "musicnet" / "musicnet"

AUDIO_DIR = str(MUSICNET_ROOT)
METADATA_PATH = str(DATA_DIR / "musicnet_metadata.csv")
CACHE_DIR = DATA_DIR / "cache"

