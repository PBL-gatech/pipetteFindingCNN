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
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import default transforms from utils.py
from utils import (
    contrast_stretch_mu_2sigma_uint8,
    get_train_transform,
    get_val_transform,
)


CHANNEL_STATS_CACHE_VERSION = 1

class PipetteDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None,
                 z_mean: float | None = None, z_std: float | None = None,
                 normalize_targets: bool = False,
                 cache_images: bool = False,
                 logger=None):
        """
        image_dir: folder containing images
        annotations: list of tuples (filename, defocus_value)
        transform: an Albumentations Compose object
        z_mean/z_std: statistics used to normalize targets (from training split)
        normalize_targets: if True, return (z - mean) / std; otherwise raw z
        cache_images: if True, preload images into RAM as uint8 numpy arrays
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform
        self.z_mean = z_mean
        self.z_std = z_std if (z_std is None or z_std > 0) else 1e-8
        self.normalize_targets = normalize_targets
        self.cache_images = cache_images
        self.logger = logger
        self._img_cache = None

        if self.cache_images:
            self._warm_cache()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename, defocus_val = self.annotations[idx]
        if self.cache_images and self._img_cache is not None:
            img_np = self._img_cache[filename]
        else:
            image_path = os.path.join(self.image_dir, filename)
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
        if self.transform:
            # Albumentations expects a dict with "image"
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            # Fallback: manual conversion to tensor if no transform is provided
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        if self.normalize_targets and self.z_mean is not None and self.z_std is not None:
            defocus_val = (defocus_val - self.z_mean) / self.z_std
        defocus_tensor = torch.tensor(defocus_val, dtype=torch.float32)
        return img_tensor, defocus_tensor

    def _warm_cache(self):
        start = time.time()
        self._img_cache = {}
        total = len(self.annotations)
        for i, (filename, _) in enumerate(self.annotations, start=1):
            path = os.path.join(self.image_dir, filename)
            img = Image.open(path).convert('RGB')
            self._img_cache[filename] = np.array(img)
            if self.logger and (i % 200 == 0 or i == total):
                self.logger(f"Image RAM cache: {i}/{total}")
        if self.logger:
            self.logger(f"Image RAM cache ready in {time.time() - start:.1f}s; cached {total} images")

class PipetteDataModule:
    def __init__(self, image_dir, annotation_file,
                 train_split=0.7, val_split=0.2, test_split=0.1,
                 seed=42,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 default_img_size=224,
                 split_save_path: str | None = None,
                 channel_stats_cache_path: str | None = None,
                 channel_stats_max_images: int | None = 2000,
                 enable_contrast_stretch: bool = False,
                 enable_aug_flip_rotate: bool = False,
                 cache_images: bool = False,
                 logger=None):
        """
        image_dir: folder with images
        annotation_file: CSV with columns: filename, defocus_microns
        train_split: fraction of data for training
        val_split: fraction of data for validation
        test_split: fraction of data for testing
        seed: random seed
        train_transform: Albumentations transform for training (if None, uses utils.get_train_transform)
        val_transform: Albumentations transform for validation (if None, uses utils.get_val_transform)
        test_transform: Albumentations transform for testing (if None, uses utils.get_val_transform)
        default_img_size: default image size to pass to transform generators if transforms are not provided
        split_save_path: optional path to persist the split indices (e.g., under the run folder)
        channel_stats_cache_path: optional path to cache per-channel mean/std computation
        channel_stats_max_images: optional cap on number of images to use when computing channel stats (None = use all)
        enable_contrast_stretch: if True, apply paper-style contrast stretching before normalization
        enable_aug_flip_rotate: if True, use expanded flip/turn train augmentation policy
        cache_images: if True, preload images into RAM (duplicates per worker, so use num_workers=0)
        logger: optional callable for progress messages (e.g., print or Qt log)
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.default_img_size = default_img_size
        self.split_save_path = split_save_path
        self.channel_stats_cache_path = channel_stats_cache_path
        self.channel_stats_max_images = channel_stats_max_images
        self.enable_contrast_stretch = bool(enable_contrast_stretch)
        self.enable_aug_flip_rotate = bool(enable_aug_flip_rotate)
        self.cache_images = cache_images
        self.logger = logger

        self.annotations = []
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.channel_mean = None
        self.channel_std = None

    def _log(self, msg: str):
        if self.logger:
            self.logger(msg)

    def load_annotations(self):
        self._log("Loading annotations CSV...")
        df = pd.read_csv(self.annotation_file)
        # Expect columns: "filename", "defocus_microns"
        self.annotations = [(row["filename"], row["defocus_microns"]) for _, row in df.iterrows()]
        self._log(f"Loaded {len(self.annotations)} annotations.")

    def setup(self):
        self.load_annotations()
        total_len = len(self.annotations)
        indices = list(range(total_len))
        random.seed(self.seed)
        random.shuffle(indices)
        self._log("Shuffled dataset indices.")

        train_end = int(total_len * self.train_split)
        val_end = train_end + int(total_len * self.val_split)

        train_indices = indices[:train_end]
        val_indices   = indices[train_end:val_end]
        test_indices  = indices[val_end:]
        self._log(f"Split counts -> train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}")

        # Compute target normalization stats on the training subset only
        train_targets = [self.annotations[i][1] for i in train_indices]
        z_mean = float(np.mean(train_targets))
        z_std = float(np.std(train_targets) + 1e-8)

        # Sub-lists of (filename, defocus_value)
        train_annot = [self.annotations[i] for i in train_indices]
        val_annot   = [self.annotations[i] for i in val_indices]
        test_annot  = [self.annotations[i] for i in test_indices]

        # Compute per-channel mean/std from the training subset (images scaled to [0,1])
        channel_mean, channel_std = self.compute_channel_stats(train_annot)
        self.channel_mean, self.channel_std = channel_mean, channel_std

        # Instantiate transforms lazily so they can use dataset stats
        if self.train_transform is None:
            self.train_transform = get_train_transform(
                img_size=self.default_img_size,
                mean=channel_mean,
                std=channel_std,
                enable_contrast_stretch=self.enable_contrast_stretch,
                enable_aug_flip_rotate=self.enable_aug_flip_rotate,
            )
        if self.val_transform is None:
            self.val_transform = get_val_transform(
                img_size=self.default_img_size,
                mean=channel_mean,
                std=channel_std,
                enable_contrast_stretch=self.enable_contrast_stretch,
            )
        if self.test_transform is None:
            self.test_transform = get_val_transform(
                img_size=self.default_img_size,
                mean=channel_mean,
                std=channel_std,
                enable_contrast_stretch=self.enable_contrast_stretch,
            )

        # Create separate datasets using the provided (or default) transforms
        self.train_dataset = PipetteDataset(
            self.image_dir,
            train_annot,
            transform=self.train_transform,
            z_mean=z_mean,
            z_std=z_std,
            normalize_targets=True,
            cache_images=self.cache_images,
            logger=self.logger,
        )
        self.val_dataset = PipetteDataset(
            self.image_dir,
            val_annot,
            transform=self.val_transform,
            z_mean=z_mean,
            z_std=z_std,
            normalize_targets=True,
            cache_images=self.cache_images,
            logger=self.logger,
        )
        self.test_dataset = PipetteDataset(
            self.image_dir,
            test_annot,
            transform=self.test_transform,
            z_mean=z_mean,
            z_std=z_std,
            normalize_targets=True,
            cache_images=self.cache_images,
            logger=self.logger,
        )

        # Save the split indices for reproducibility
        split_path = self.split_save_path or "data_splits.pkl"
        with open(split_path, "wb") as f:
            pickle.dump({"train": train_indices, "val": val_indices, "test": test_indices}, f)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def compute_channel_stats(self, annotations_subset):
        """
        Compute per-channel mean/std over the given annotations subset.
        Images are read as RGB, converted to float32 in [0,1].
        """
        cache_key = {
            "version": CHANNEL_STATS_CACHE_VERSION,
            "enable_contrast_stretch": self.enable_contrast_stretch,
        }

        # Try cache first
        if self.channel_stats_cache_path and os.path.isfile(self.channel_stats_cache_path):
            try:
                with open(self.channel_stats_cache_path, "rb") as f:
                    cached = pickle.load(f)
                if (
                    "mean" in cached
                    and "std" in cached
                    and cached.get("preprocess_key") == cache_key
                ):
                    return cached["mean"], cached["std"]
            except Exception:
                pass

        sum_c = np.zeros(3, dtype=np.float64)
        sumsq_c = np.zeros(3, dtype=np.float64)
        pixel_count = 0

        subset = annotations_subset
        if self.channel_stats_max_images is not None and len(subset) > self.channel_stats_max_images:
            subset = random.sample(subset, self.channel_stats_max_images)
        total = len(subset)
        start = time.time()
        self._log(f"Computing channel stats on {total} images{'' if total==len(annotations_subset) else f' (capped at {self.channel_stats_max_images})'}...")

        for idx, (filename, _) in enumerate(subset, start=1):
            img_path = os.path.join(self.image_dir, filename)
            img_uint8 = np.array(Image.open(img_path).convert('RGB'))
            if self.enable_contrast_stretch:
                img_uint8 = contrast_stretch_mu_2sigma_uint8(img_uint8)
            img = img_uint8.astype(np.float32) / 255.0
            flat = img.reshape(-1, 3)
            sum_c += flat.sum(axis=0)
            sumsq_c += (flat ** 2).sum(axis=0)
            pixel_count += flat.shape[0]
            if idx % 200 == 0 or idx == total:
                self._log(f"Channel stats progress: {idx}/{total}")

        mean = sum_c / pixel_count
        var = sumsq_c / pixel_count - mean ** 2
        std = np.sqrt(np.clip(var, 1e-8, None))

        self._log(f"Channel stats done in {time.time() - start:.1f}s")

        if self.channel_stats_cache_path:
            try:
                with open(self.channel_stats_cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "mean": mean.tolist(),
                            "std": std.tolist(),
                            "preprocess_key": cache_key,
                        },
                        f,
                    )
            except Exception:
                pass

        return mean.tolist(), std.tolist()
