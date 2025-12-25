import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ----------------------------------------------------------------------
# DATASET PATHS
# ----------------------------------------------------------------------

TRAIN_DIRS = [
    "./data/train/DIV2K/",
    "./data/train/Flickr2K/",
]

TEST_DIRS = [
    "./data/test/large_dataset/",
]

# Supported image formats (file-type neutral)
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 0

# ----------------------------------------------------------------------
# DATASET CLASS
# ----------------------------------------------------------------------

class ACLMImageDataset(Dataset):
    def __init__(self, roots, transform=None):

        self.image_files = []

        for root in roots:
            if os.path.exists(root):
                for ext in IMAGE_EXTENSIONS:
                    self.image_files.extend(
                        glob(os.path.join(root, "**", ext), recursive=True)
                    )

        if not self.image_files:
            raise FileNotFoundError(
                "No images found in the specified dataset directories."
            )

        print(f" DataLoader initialized. Total images loaded: {len(self.image_files)}")

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# ----------------------------------------------------------------------
# DATALOADER FACTORY
# ----------------------------------------------------------------------

def get_data_loader(mode="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

    if mode == "train":
        roots = TRAIN_DIRS
    elif mode == "test":
        roots = TEST_DIRS
    else:
        raise ValueError("mode must be either 'train' or 'test'")

    data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    dataset = ACLMImageDataset(
        roots=roots,
        transform=data_transforms,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=False,
    )

    return data_loader, dataset

# ----------------------------------------------------------------------
# DEBUG TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        loader, ds = get_data_loader(mode="train")
        print(f"Training batches: {len(loader)}")

        loader, ds = get_data_loader(mode="test")
        print(f"Testing batches: {len(loader)}")

        for images in loader:
            print(f"Sample batch shape: {images.shape}")
            break

    except FileNotFoundError as e:
        print(f"Error during test run: {e}")