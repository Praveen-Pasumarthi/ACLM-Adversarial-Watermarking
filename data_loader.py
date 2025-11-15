import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Configuration (DF2K Dataset Paths) ---
# NOTE: The DataLoader will load images from BOTH directories listed below.
DIV2K_IMAGE_DIR = './data/DIV2K/DIV2K_train_HR/' 
FLICKR2K_IMAGE_DIR = './data/Flickr2K/Flickr2K_HR/' # Assuming this is where the Flickr2K HR images were manually extracted
IMAGE_SIZE = 256  
BATCH_SIZE = 4
NUM_WORKERS = 0 # Keep at 0 for system stability on Windows

# --- 1. PyTorch Dataset (DF2K) ---

class ACLMImageDataset(Dataset):
    """Custom Dataset for loading the combined DF2K images (DIV2K + Flickr2K)."""
    def __init__(self, transform=None):
        
        # 1. Load files from DIV2K (if directory exists)
        div2k_files = []
        if os.path.exists(DIV2K_IMAGE_DIR):
            # Recursively find all PNG files
            div2k_files = [f for f in glob(os.path.join(DIV2K_IMAGE_DIR, '**', '*.png'), recursive=True)]
        
        # 2. Load files from Flickr2K (if directory exists)
        flickr2k_files = []
        if os.path.exists(FLICKR2K_IMAGE_DIR):
            flickr2k_files = [f for f in glob(os.path.join(FLICKR2K_IMAGE_DIR, '**', '*.png'), recursive=True)]
        
        # 3. Combine both lists (DF2K Dataset)
        self.image_files = div2k_files + flickr2k_files

        if not self.image_files:
            raise FileNotFoundError("No PNG images found in either DIV2K or Flickr2K directories. Total Images: 0.")
        
        print(f"✅ DataLoader initialized. Total images loaded (DF2K): {len(self.image_files)}")
        
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# --- 2. Data Transformations and Loader Factory ---

def get_data_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Defines the image preprocessing pipeline and returns the DataLoader.
    The image_dir parameter is removed as the paths are hardcoded for DF2K consistency.
    """
    
    data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), 
        transforms.CenterCrop(IMAGE_SIZE), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # The dataset initialization no longer needs the image_dir parameter
    dataset = ACLMImageDataset(transform=data_transforms)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return data_loader, dataset

if __name__ == '__main__':
    try:
        # Test function call to verify file loading
        loader, ds = get_data_loader()
        print(f"Total batches: {len(loader)}")
        # Check the size of the first batch
        for batch_idx, images in enumerate(loader):
            print(f"Sample Batch Shape: {images.shape}")
            break
    except FileNotFoundError as e:
        print(f"Error during test run: {e}")