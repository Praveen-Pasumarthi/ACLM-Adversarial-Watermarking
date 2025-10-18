import os
import kagglehub
from pathlib import Path

def download_div2k_via_hub():
    """Downloads and extracts the DIV2K dataset using the reliable kagglehub library."""
    
    # 1. Set Kaggle config directory to project's .kaggle folder (for API token reading)
    # This is often unnecessary with kagglehub but is good practice.
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')
    
    # 2. Define the local data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    print("📥 Starting dataset download using kagglehub...")
    
    try:
        # 3. Use kagglehub to download the dataset
        # The library handles authentication, progress, download, and extraction
        # The 'dataset_download' function returns the path to the extracted files.
        extracted_path = kagglehub.dataset_download(
            "sabahesaraki/2017-2017" 
        )
        
        # 4. Move extracted files to the project's data/ directory
        # This part ensures the files end up in your expected location.
        # Check the structure of the extracted content (often a folder named after the dataset)
        
        print("✅ Download and extraction complete.")
        print(f"Extracted files are at: {extracted_path}")
        
        # NOTE: You will need to move the contents of extracted_path into your ./data folder
        # For simplicity, we'll assume the files are ready, but you might need a manual move 
        # based on where kagglehub saves the data (usually in a global cache).
        
    except Exception as e:
        print(f"❌ An error occurred during download: {e}")
        print("Please ensure your .kaggle/kaggle.json file is correct and saved.")


if __name__ == '__main__':
    download_div2k_via_hub()

# --- End of download_kaggle_dataset.py ---