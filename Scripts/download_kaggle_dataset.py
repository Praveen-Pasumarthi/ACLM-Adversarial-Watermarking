import os

# Set Kaggle config directory
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')

# Make sure the data folder exists
os.makedirs('./data', exist_ok=True)

# Download and unzip dataset from Kaggle
print("📥 Downloading Kaggle dataset (2017-2017)...")
os.system('kaggle datasets download -d sabahesaraki/2017-2017 -p ./data --unzip')
print("✅ Download complete and extracted to ./data")
