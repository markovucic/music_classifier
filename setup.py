import kagglehub
from paths import DATA_DIR

def download_music_net() -> str:
    path = kagglehub.dataset_download("imsparsh/musicnet-dataset", output_dir=DATA_DIR)
    return path 

if __name__ == "__main__":
    path = download_music_net()
    print(f"MusicNet dataset downloaded from Kaggle onto path: {path}")