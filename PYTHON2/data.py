"""
data.py
---------
This module defines the PipetteDataset (a subclass of torch.utils.data.Dataset)
and a DataModule class that loads and splits the dataset.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from torchvision import transforms


class PipetteDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):
        """
        image_dir: Directory containing images.
        annotations: List of tuples (filename, [x, y, z]).
        transform: Torchvision transforms to apply.
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename, coords = self.annotations[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        return image, coords_tensor


class PipetteDataModule:
    def __init__(self, image_dir, annotation_file, val_split=0.1):
        """
        image_dir: Directory containing images.
        annotation_file: Path to a CSV file with columns: filename, x, y, z.
        val_split: Fraction of the data to use for validation.
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.val_split = val_split
        self.annotations = None
        self.train_dataset = None
        self.val_dataset = None

    @staticmethod
    def get_transforms():
        """
        Returns a torchvision transforms pipeline for the training images.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Standard ImageNet normalization.
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def load_annotations(self):
        """
        Reads the annotation CSV and creates a list of (filename, [x, y, z]) tuples.
        """
        df = pd.read_csv(self.annotation_file)
        self.annotations = []
        for _, row in df.iterrows():
            filename = row['filename']
            coords = [row['x'], row['y'], row['z']]
            self.annotations.append((filename, coords))

    def setup(self):
        """
        Loads the annotations, creates the dataset, and splits it into training and validation sets.
        Returns:
            train_dataset, val_dataset
        """
        self.load_annotations()
        transform = self.get_transforms()
        full_dataset = PipetteDataset(self.image_dir, self.annotations, transform=transform)
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        return self.train_dataset, self.val_dataset
