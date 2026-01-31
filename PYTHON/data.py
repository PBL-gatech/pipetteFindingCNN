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

# Import default transforms from utils.py
from utils import get_train_transform, get_val_transform

class PipetteDataset(Dataset):
    """
    Dataset that keeps targets in real units (pixels / microns) and applies
    Albumentations transforms with keypoint safety so x/y labels are always
    transformed consistently with the image.
    """

    def __init__(self, image_dir, annotations_df, transform=None, img_size=224):
        self.image_dir = image_dir
        self.annotations_df = annotations_df.reset_index(drop=True)
        self.transform = transform
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        filename = row["filename"]
        image_path = os.path.join(self.image_dir, filename)

        img = Image.open(image_path).convert("RGB")
        w0, h0 = img.size  # raw full-frame size (no hardcoding)
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        img_np = np.array(img)

        target_vals = row[["pipette_x_px", "pipette_y_px", "defocus_microns"]].astype(np.float32).to_numpy()

        # Scale raw full-frame x/y into resized img_size frame
        sx = self.img_size / float(w0)
        sy = self.img_size / float(h0)
        target_vals[0] = target_vals[0] * sx
        target_vals[1] = target_vals[1] * sy

        if self.transform:
            keypoints = [(float(target_vals[0]), float(target_vals[1]))]
            augmented = self.transform(image=img_np, keypoints=keypoints)
            aug_x, aug_y = augmented["keypoints"][0]
            target_vals[0] = float(aug_x)
            target_vals[1] = float(aug_y)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        target_tensor = torch.tensor(target_vals, dtype=torch.float32)
        return img_tensor, target_tensor

class PipetteDataModule:
    REQUIRED_COLS = ["filename", "pipette_x_px", "pipette_y_px", "defocus_microns"]

    def __init__(self, image_dir, annotation_file,
                 train_split=0.7, val_split=0.2, test_split=0.1,
                 seed=42,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 img_size=224,
                 flip_p=0.5,
                 rotate90_p=0.5,
                 z_mean=None,
                 z_std=None):
        """
        Enforces presence of filename + pipette_x_px + pipette_y_px + defocus_microns.
        Keeps targets in real units; normalization of z is handled inside the model.
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.img_size = img_size
        self.flip_p = flip_p
        self.rotate90_p = rotate90_p
        self.provided_z_mean = z_mean
        self.provided_z_std = z_std

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.annotations_df = None
        self.target_dim = 3
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.z_mean = None
        self.z_std = None

    def load_annotations(self):
        df = pd.read_csv(self.annotation_file)
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Annotation CSV must contain columns {self.REQUIRED_COLS}; missing {missing}")
        self.annotations_df = df[self.REQUIRED_COLS].copy()

    def _compute_z_stats(self, train_df):
        z_vals = train_df["defocus_microns"].astype(np.float32).to_numpy()
        mean = float(np.mean(z_vals))
        std = float(np.std(z_vals))
        if std < 1e-6:
            std = 1e-6
        self.z_mean = mean
        self.z_std = std

    def _build_transforms(self):
        if self.train_transform is None:
            self.train_transform = get_train_transform(
                img_size=self.img_size,
                use_keypoints=True,
                flip_p=self.flip_p,
                rotate90_p=self.rotate90_p,
            )
        elif getattr(self.train_transform, "keypoint_params", None) is None:
            raise ValueError("Training transform must include keypoint_params for x/y keypoints.")

        if self.val_transform is None:
            self.val_transform = get_val_transform(img_size=self.img_size, use_keypoints=True)
        elif getattr(self.val_transform, "keypoint_params", None) is None:
            raise ValueError("Validation transform must include keypoint_params for x/y keypoints.")

        if self.test_transform is None:
            self.test_transform = get_val_transform(img_size=self.img_size, use_keypoints=True)
        elif getattr(self.test_transform, "keypoint_params", None) is None:
            raise ValueError("Test transform must include keypoint_params for x/y keypoints.")

    def setup(self):
        self.load_annotations()
        coords = list(zip(
            self.annotations_df["pipette_x_px"].astype(float).to_list(),
            self.annotations_df["pipette_y_px"].astype(float).to_list(),
        ))
        coord_to_indices = {}
        for i, key in enumerate(coords):
            coord_to_indices.setdefault(key, []).append(i)

        unique_keys = list(coord_to_indices.keys())
        random.seed(self.seed)
        random.shuffle(unique_keys)

        total_len = len(self.annotations_df)
        train_target = int(total_len * self.train_split)
        val_target   = int(total_len * self.val_split)

        train_indices, val_indices, test_indices = [], [], []
        count_train = count_val = 0

        for key in unique_keys:
            inds = coord_to_indices[key]
            if count_train < train_target:
                train_indices.extend(inds); count_train += len(inds)
            elif count_val < val_target:
                val_indices.extend(inds); count_val += len(inds)
            else:
                test_indices.extend(inds)

        # Sub-DataFrames for each split
        train_df = self.annotations_df.iloc[train_indices].reset_index(drop=True)
        val_df   = self.annotations_df.iloc[val_indices].reset_index(drop=True)
        test_df  = self.annotations_df.iloc[test_indices].reset_index(drop=True)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Build transforms now that we know whether x/y targets are present
        self._build_transforms()

        # Compute z stats from training split for model normalization
        if len(train_df) == 0:
            if self.provided_z_mean is None or self.provided_z_std is None:
                raise ValueError("Training split is empty and no z_mean/z_std provided.")
            self.z_mean = float(self.provided_z_mean)
            self.z_std = max(float(self.provided_z_std), 1e-6)
        else:
            self._compute_z_stats(train_df)

        # Create separate datasets using the provided (or default) transforms
        self.train_dataset = PipetteDataset(
            self.image_dir,
            train_df,
            transform=self.train_transform,
            img_size=self.img_size,
        )
        self.val_dataset = PipetteDataset(
            self.image_dir,
            val_df,
            transform=self.val_transform,
            img_size=self.img_size,
        )
        self.test_dataset = PipetteDataset(
            self.image_dir,
            test_df,
            transform=self.test_transform,
            img_size=self.img_size,
        )

        # Save the split indices for reproducibility
        with open("data_splits.pkl", "wb") as f:
            pickle.dump({"train": train_indices, "val": val_indices, "test": test_indices}, f)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_z_stats(self):
        if self.z_mean is None or self.z_std is None:
            raise RuntimeError("Z statistics not computed yet; call setup() first.")
        return self.z_mean, self.z_std
