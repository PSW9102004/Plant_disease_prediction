import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import kagglehub
import shutil

# Base directory
BASE_DIR = Path("/content/drive/MyDrive/25DLS343_Capstone Project/Capstone Project")
RAW_DIR = BASE_DIR / "data/raw"
PLANTDOC_DIR = RAW_DIR / "PlantDoc"
PLANTVILLAGE_DIR = RAW_DIR / "PlantVillage"


def extract_kagglehub_folder(kagglehub_path: str, extract_to: Path):
    """Copy extracted files from kagglehub dataset directory to a flat folder."""
    src_dir = Path(kagglehub_path)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = Path(root) / file
                rel_path = src_file.relative_to(src_dir)
                dest_file = extract_to / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_file, dest_file)


def download_plantdoc():
    print("Downloading PlantDoc from KaggleHub (abdulhasibuddin/plant-doc-dataset)...")
    dataset_dir = kagglehub.dataset_download("abdulhasibuddin/plant-doc-dataset")
    print("Copying PlantDoc to", PLANTDOC_DIR)
    extract_kagglehub_folder(dataset_dir, PLANTDOC_DIR)
    print("✅ PlantDoc dataset downloaded and copied.")


def download_plantvillage():
    print("Downloading PlantVillage from KaggleHub (emmarex/plantdisease)...")
    dataset_dir = kagglehub.dataset_download("emmarex/plantdisease")
    print("Copying PlantVillage to", PLANTVILLAGE_DIR)
    extract_kagglehub_folder(dataset_dir, PLANTVILLAGE_DIR)
    print("✅ PlantVillage dataset downloaded and copied.")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    try:
        download_plantdoc()
    except Exception as e:
        print(f"Error downloading PlantDoc: {e}")

    try:
        download_plantvillage()
    except Exception as e:
        print(f"Error downloading PlantVillage: {e}")


if __name__ == '__main__':
    main()
