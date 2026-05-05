"""Download and extract the FMA dataset (optional).

This script downloads the full FMA small subset (~8GB) and metadata.
NOTE: The canonical 2,000-track metadata subset (fma_2000_metadata/) is already
included in the repo, so this script is only needed if you want the full 8,000-track
metadata. All pipelines use fma_2000_metadata/ by default (defined in src/config.py).
"""
import argparse
import hashlib
import os
import sys
import zipfile
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import requests
from tqdm import tqdm

from src.config import RAW_DIR, DATA_DIR

BASE_URL = "https://os.unil.cloud.switch.ch/fma/"
FILES = ["fma_metadata.zip", "fma_small.zip"]

# SHA-256 checksums (from FMA repo)
CHECKSUMS = {
    "fma_metadata.zip": None,  # Verify manually if needed
    "fma_small.zip": None,
}


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path, dest_dir):
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"Extracted to {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract FMA dataset.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction (data already extracted).")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download (zips already present).")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for file_name in FILES:
        dest_path = RAW_DIR / file_name

        if not args.skip_download:
            if dest_path.exists():
                print(f"{file_name} already exists, skipping download.")
            else:
                url = BASE_URL + file_name
                print(f"Downloading {file_name}...")
                try:
                    download_file(url, dest_path)
                except Exception as e:
                    print(f"Error downloading {file_name}: {e}")
                    continue

        if not args.skip_extract:
            extract_name = file_name.replace(".zip", "")
            extract_dest = DATA_DIR / extract_name
            if extract_dest.exists():
                print(f"{extract_name}/ already exists, skipping extraction.")
            else:
                extract_zip(dest_path, DATA_DIR)


if __name__ == "__main__":
    main()
