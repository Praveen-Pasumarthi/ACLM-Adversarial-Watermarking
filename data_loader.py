import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Configuration ---
# Match this to your image folder structure after successful extraction!
DIV2K_IMAGE_DIR = './data/DIV2K/DIV2K_train_HR/'
IMAGE_SIZE = 256  # Common size for LDM training, which is divisible by 8 (for VAE)
BATCH_SIZE = 4
NUM_WORKERS = 4

# --- 1. PyTorch Dataset ---

class ACLMImageDataset(Dataset):
    """Custom Dataset for loading high-resolution DIV2K images."""
    def __init__(self, image_dir, transform=None):
        # Recursively find all PNG files, which is common for DIV2K
        self.image_files = [f for f in glob(os.path.join(image_dir, '**', '*.png'), recursive=True)]
        
        if not self.image_files:
            raise FileNotFoundError(f"No PNG images found in the directory: {image_dir}")
        
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Open and ensure image is in RGB format
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# --- 2. Data Transformations and Loader Factory (Preprocessing Pipeline) ---

def get_data_loader(image_dir=DIV2K_IMAGE_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Defines the image preprocessing pipeline and returns the DataLoader.
    The images are preprocessed to be compatible with a pre-trained VAE/LDM.
    """
    
    # 🌟 The Core Preprocessing Pipeline for LDM Input 🌟
    data_transforms = transforms.Compose([
        # 1. Resize/Crop: Resize the smaller edge to IMAGE_SIZE and then CenterCrop
        # This ensures square images of the correct size.
        transforms.Resize(IMAGE_SIZE), 
        transforms.CenterCrop(IMAGE_SIZE), 
        
        # 2. Convert to Tensor: PIL image (H, W, C) -> Tensor (C, H, W), range [0, 1]
        transforms.ToTensor(), 
        
        # 3. Normalization: Scale tensor from [0, 1] to the required [-1, 1] range.
        # This is standard practice for VAE input in Latent Diffusion Models.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    dataset = ACLMImageDataset(image_dir, transform=data_transforms)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return data_loader, dataset

# ----------------------------------------------------------------------
# 🌟 NEXT STEP VALIDATION (Run this only after data is moved and extracted!)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    try:
        loader, ds = get_data_loader()
        print(f"Total images loaded: {len(ds)}")
        
        # Test fetching one batch to confirm tensor shape and range
        for batch_idx, images in enumerate(loader):
            # Expected shape: [BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE]
            print(f"Batch {batch_idx+1} shape: {images.shape}")
            
            # Expected range: Min should be close to -1.0, Max should be close to 1.0
            print(f"Tensor range: Min={images.min().item():.2f}, Max={images.max().item():.2f}")
            break
            
    except FileNotFoundError as e:
        print(f"Data not ready. Please confirm the folder path: {DIV2K_IMAGE_DIR} and extraction are complete.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Check your NUM_WORKERS setting (try 0).")