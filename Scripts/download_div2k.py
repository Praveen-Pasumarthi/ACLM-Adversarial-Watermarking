import os
import requests
import zipfile
from tqdm import tqdm

URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
OUT_PATH = "./data/DIV2K_train_HR.zip"
EXTRACT_PATH = "./data/DIV2K"

# 1. Setup Directories
os.makedirs("./data", exist_ok=True)

# 2. Download with Progress Bar
print("📥 Downloading DIV2K dataset...")

try:
    with requests.get(URL, stream=True) as r:
        r.raise_for_status()
        
        # Get total file size from headers (in bytes)
        total_size = int(r.headers.get('content-length', 0))
        
        with open(OUT_PATH, "wb") as f:
            # Initialize tqdm progress bar
            with tqdm(
                desc=OUT_PATH,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size) # Update the progress bar
                    
    print("\n✅ Download complete.")

except requests.exceptions.RequestException as e:
    print(f"\n❌ Download failed: {e}")
    exit()

# 3. Extraction
print("📂 Extracting files...")
try:
    with zipfile.ZipFile(OUT_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("✅ Extraction done.")
    
    # 4. Optional: Clean up the large zip file after successful extraction
    # os.remove(OUT_PATH) 
    # print(f"🗑️ Deleted zip file: {OUT_PATH}")

except zipfile.BadZipFile:
    print("\n❌ Extraction failed: The zip file is corrupted. Please try the download again.")
    exit()