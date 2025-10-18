import os
import requests
import zipfile

URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
OUT_PATH = "./data/DIV2K_train_HR.zip"
EXTRACT_PATH = "./data/DIV2K"

os.makedirs("./data", exist_ok=True)
print("📥 Downloading DIV2K dataset...")
with requests.get(URL, stream=True) as r:
    r.raise_for_status()
    with open(OUT_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("✅ Download complete.")

print("📂 Extracting files...")
with zipfile.ZipFile(OUT_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)
print("✅ Extraction done.")
