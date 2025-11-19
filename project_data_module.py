
import os
import torch
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader
from PIL import UnidentifiedImageError
# Note: PIL/Image is implicitly used by default_loader

# Configure logging to see which files are skipped
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# --- 1. Robust Dataset Wrapper ---
class RobustImageFolder(Dataset):
    """A wrapper for ImageFolder that skips corrupted images."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.loader = default_loader
        self.transform = dataset.transform

    def __getitem__(self, index):
        path, target = self.dataset.samples[index]

        try:
            # Load the image using the default loader
            sample = self.loader(path)

            if self.transform is not None:
                sample = self.transform(sample)

            return sample, target

        except (UnidentifiedImageError, OSError) as e:
            # If the image is corrupted or cannot be read, log and get a random new index
            logging.warning(f"Skipping corrupted file: {path}")

            # Get a random new index to fetch a valid sample instead
            new_index = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_index) # Recursively call to get a valid sample

    def __len__(self):
        return len(self.dataset)


# --- 2. Transformations ---
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- 3. HybridDataModule ---
class HybridDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        # Use the passed data_dir argument
        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.transform = IMAGE_TRANSFORM
        
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'valid')
        self.test_dir = os.path.join(self.data_dir, 'test')


    def setup(self, stage=None):
        try:
            print("Loading training dataset (Robust)...")
            # Load base ImageFolder, then wrap it with the RobustImageFolder
            base_train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.transform)
            self.train_dataset = RobustImageFolder(base_train_dataset)

            print("Loading validation dataset (Robust)...")
            base_val_dataset = datasets.ImageFolder(root=self.val_dir, transform=self.transform)
            self.val_dataset = RobustImageFolder(base_val_dataset)
            
            base_test_dataset = datasets.ImageFolder(root=self.test_dir, transform=self.transform)
            self.test_dataset = RobustImageFolder(base_test_dataset)
            
            self.num_classes = len(base_train_dataset.classes)
            print(f"Dataset loaded successfully. Found {self.num_classes} classes.")

        except Exception as e:
            print(f"⚠️ Warning: Failed to load REAL data (Exception during setup): {e}. Creating DUMMY data.")
            self.num_classes = 3
            # DUMMY DATA CREATION (This will only run if an exception occurs)
            self.train_dataset = [(torch.randn(3, 224, 224), torch.randint(0, self.num_classes, (1,)).item()) for _ in range(100)]
            self.val_dataset = [(torch.randn(3, 224, 224), torch.randint(0, self.num_classes, (1,)).item()) for _ in range(20)]
            self.test_dataset = [(torch.randn(3, 224, 224), torch.randint(0, self.num_classes, (1,)).item()) for _ in range(20)]


    def train_dataloader(self):
        # FIX: Setting num_workers=0 to prevent multiprocessing crash
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        # FIX: Setting num_workers=0
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        # FIX: Setting num_workers=0
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)

