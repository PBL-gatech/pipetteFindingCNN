"""
data.py
---------
This module defines the PipetteDataset (a subclass of torch.utils.data.Dataset)
and a PipetteDataModule class that loads and splits the dataset.
The dataset is split into 70% training, 20% validation, and 10% testing using a fixed seed.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
from torchvision import transforms
import random
import pickle

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
        # Convert scalar to a float32 tensor
        defocus_tensor = torch.tensor(defocus_val, dtype=torch.float32)
        return image, defocus_tensor

class PipetteDataModule:
    def __init__(self, image_dir, annotation_file, train_split=0.7, val_split=0.2, test_split=0.1, seed=42):
        """
        image_dir: folder with images
        annotation_file: CSV with columns: filename, defocus_microns
        train_split: fraction of data for training
        val_split: fraction of data for validation
        test_split: fraction of data for testing
        seed: random seed for reproducibility
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.annotations = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @staticmethod
    def get_transforms():
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
            defocus_val = row["defocus_microns"]
            self.annotations.append((filename, defocus_val))

    def setup(self):
        self.load_annotations()
        transform = self.get_transforms()
        full_dataset = PipetteDataset(self.image_dir, self.annotations, transform=transform)
        total_len = len(full_dataset)
        indices = list(range(total_len))
        random.seed(self.seed)
        random.shuffle(indices)
        train_end = int(total_len * self.train_split)
        val_end = train_end + int(total_len * self.val_split)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)
        # Save the split indices for reproducibility
        with open("data_splits.pkl", "wb") as f:
            pickle.dump({"train": train_indices, "val": val_indices, "test": test_indices}, f)
        return self.train_dataset, self.val_dataset, self.test_dataset
