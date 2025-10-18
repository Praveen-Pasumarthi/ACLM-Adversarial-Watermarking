import os
import subprocess
from pathlib import Path # <-- Add this import

# --- Dynamic Path Identification ---
# This path is where pip installs local executables for the Windows Store Python
scripts_dir = Path.home() / 'AppData' / 'Local' / 'Packages' / 'PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0' / 'LocalCache' / 'local-packages' / 'Python312' / 'Scripts'
KAGGLE_EXEC = scripts_dir / 'kaggle.exe'

if not KAGGLE_EXEC.exists():
    # Fallback to the system PATH if the explicit path doesn't exist
    KAGGLE_EXEC = 'kaggle' 
    print("⚠️ Warning: Could not find explicit kaggle.exe path. Falling back to system PATH...")
# -----------------------------------


# Set Kaggle config directory
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')

# Make sure the data folder exists
os.makedirs('./data', exist_ok=True)

# Download and unzip dataset from Kaggle
print("📥 Downloading Kaggle dataset (2017-2017)...")

try:
    # Use the explicitly found path to kaggle.exe
    subprocess.run([
        str(KAGGLE_EXEC), 'datasets', 'download', 
        '-d', 'sabahesaraki/2017-2017', 
        '-p', './data', 
        '--unzip'
    ], check=True)
    print("✅ Download complete and extracted to ./data")
except subprocess.CalledProcessError as e:
    print(f"❌ Error during Kaggle download: {e}")
    print("Please ensure your .kaggle/kaggle.json file is correct and your API key is valid.")

# --- End of modified script ---