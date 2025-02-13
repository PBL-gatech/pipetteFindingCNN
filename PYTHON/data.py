"""
data.py
---------
This module defines the PipetteDataset (a subclass of torch.utils.data.Dataset)
and a DataModule class that loads and splits the dataset.
"""

# data.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from torchvision import transforms

class PipetteDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):
        """
        image_dir: folder containing images
        annotations: list of tuples (filename, defocus_value)
        transform: torchvision transforms
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename, defocus_val = self.annotations[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Convert scalar to float32
        defocus_tensor = torch.tensor(defocus_val, dtype=torch.float32)
        return image, defocus_tensor


class PipetteDataModule:
    def __init__(self, image_dir, annotation_file, val_split=0.1):
        """
        image_dir: folder with images
        annotation_file: CSV with columns: filename, defocus_microns
        val_split: fraction for validation
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.val_split = val_split
        self.annotations = None
        self.train_dataset = None
        self.val_dataset = None

    @staticmethod
    def get_transforms():
        # Same transforms as before, except we only predict 1 value now.
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def load_annotations(self):
        df = pd.read_csv(self.annotation_file)
        self.annotations = []
        for _, row in df.iterrows():
            filename = row["filename"]
            defocus_val = row["defocus_microns"]  # single float
            self.annotations.append((filename, defocus_val))

    def setup(self):
        self.load_annotations()
        transform = self.get_transforms()
        full_dataset = PipetteDataset(self.image_dir, self.annotations, transform=transform)

        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        return self.train_dataset, self.val_dataset
