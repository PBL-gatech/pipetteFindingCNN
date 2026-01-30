#!/usr/bin/env python
"""
data.py
---------
Defines a PipetteDataset and a PipetteDataModule for loading and splitting data.
We create separate datasets for train, val, and test, each with its own Albumentations transform.
Splits are 70% train, 20% val, and 10% test by default.
Data augmentation transforms are now sourced from utils.py if not provided.
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import default transforms from utils.py
from utils import get_train_transform, get_val_transform

class PipetteDataset(Dataset):
    def __init__(self, image_dir, annotations_df, target_cols, transform=None):
        """
        image_dir: folder containing images
        annotations_df: DataFrame slice with filename and target columns
        target_cols: ordered list of target column names to pull
        transform: an Albumentations Compose object
        """
        self.image_dir = image_dir
        self.annotations_df = annotations_df.reset_index(drop=True)
        self.target_cols = target_cols
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        filename = row["filename"]
        image_path = os.path.join(self.image_dir, filename)
        # Load image as PIL and convert to RGB
        img = Image.open(image_path).convert('RGB')
        # Convert to numpy array for Albumentations
        img_np = np.array(img)
        if self.transform:
            # Albumentations expects a dict with "image"
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            # Fallback: manual conversion to tensor if no transform is provided
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        target_vals = row[self.target_cols].astype(np.float32).to_numpy()
        target_tensor = torch.tensor(target_vals, dtype=torch.float32)
        return img_tensor, target_tensor

class PipetteDataModule:
    def __init__(self, image_dir, annotation_file,
                 train_split=0.7, val_split=0.2, test_split=0.1,
                 seed=42,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 default_img_size=224,
                 target_cols=None):
        """
        image_dir: folder with images
        annotation_file: CSV with columns: filename plus targets (defocus_microns and optional pipette_x_px/pipette_y_px)
        train_split: fraction of data for training
        val_split: fraction of data for validation
        test_split: fraction of data for testing
        seed: random seed
        train_transform: Albumentations transform for training (if None, uses utils.get_train_transform)
        val_transform: Albumentations transform for validation (if None, uses utils.get_val_transform)
        test_transform: Albumentations transform for testing (if None, uses utils.get_val_transform)
        default_img_size: default image size to pass to transform generators if transforms are not provided
        target_cols: ordered list of targets to predict; if None, auto-detects available columns (x/y/z)
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.target_cols = target_cols

        # If transforms are not provided, load default transforms from utils.py
        self.train_transform = train_transform if train_transform is not None else get_train_transform(img_size=default_img_size)
        self.val_transform = val_transform if val_transform is not None else get_val_transform(img_size=default_img_size)
        self.test_transform = test_transform if test_transform is not None else get_val_transform(img_size=default_img_size)

        self.annotations_df = None
        self.target_dim = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_annotations(self):
        df = pd.read_csv(self.annotation_file)
        if self.target_cols is None:
            # Auto-detect targets in priority order: x,y,z; fall back to z only.
            auto_cols = [col for col in ["pipette_x_px", "pipette_y_px", "defocus_microns"] if col in df.columns]
            if not auto_cols:
                auto_cols = ["defocus_microns"]
            self.target_cols = auto_cols
        missing = [c for c in self.target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns in annotations: {missing}")
        self.annotations_df = df[["filename"] + self.target_cols].copy()
        self.target_dim = len(self.target_cols)

    def setup(self):
        self.load_annotations()
        total_len = len(self.annotations_df)
        indices = list(range(total_len))
        random.seed(self.seed)
        random.shuffle(indices)

        train_end = int(total_len * self.train_split)
        val_end = train_end + int(total_len * self.val_split)

        train_indices = indices[:train_end]
        val_indices   = indices[train_end:val_end]
        test_indices  = indices[val_end:]

        # Sub-DataFrames for each split
        train_df = self.annotations_df.iloc[train_indices].reset_index(drop=True)
        val_df   = self.annotations_df.iloc[val_indices].reset_index(drop=True)
        test_df  = self.annotations_df.iloc[test_indices].reset_index(drop=True)

        # Create separate datasets using the provided (or default) transforms
        self.train_dataset = PipetteDataset(
            self.image_dir,
            train_df,
            self.target_cols,
            transform=self.train_transform
        )
        self.val_dataset = PipetteDataset(
            self.image_dir,
            val_df,
            self.target_cols,
            transform=self.val_transform
        )
        self.test_dataset = PipetteDataset(
            self.image_dir,
            test_df,
            self.target_cols,
            transform=self.test_transform
        )

        # Save the split indices for reproducibility
        with open("data_splits.pkl", "wb") as f:
            pickle.dump({"train": train_indices, "val": val_indices, "test": test_indices}, f)

        return self.train_dataset, self.val_dataset, self.test_dataset
