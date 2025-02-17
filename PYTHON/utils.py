#!/usr/bin/env python
"""
utils.py
----------
This module contains various helper functions for image preprocessing, augmentation,
and visualization. It now includes the get_train_transform and get_val_transform functions,
which define your Albumentations augmentation pipelines.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----------------------------
# Albumentations Transforms
# ----------------------------

def get_train_transform(img_size=224):
    """
    Returns an Albumentations Compose transform for training that applies:
      - Resize to img_size x img_size
      - Horizontal flip, random 90° rotations, and color jitter
      - Normalization using ImageNet mean and std
      - Conversion to a PyTorch tensor
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform(img_size=224):
    """
    Returns an Albumentations Compose transform for validation/testing that applies:
      - Resize to img_size x img_size
      - Normalization using ImageNet mean and std
      - Conversion to a PyTorch tensor
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ----------------------------
# Plotting Utilities
# ----------------------------

def plot_training_metrics(epochs, train_losses, val_losses, save_path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="Train Loss", color="cyan", lw=2)
    ax.plot(epochs, val_losses, label="Val Loss", color="magenta", lw=2)
    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)

def plot_regression_metrics(epochs, mae_scores, r2_scores, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, mae_scores, label="MAE", color="orange", lw=2)
    ax1.set_title("Mean Absolute Error Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE (microns)")
    ax1.legend()
    ax2.plot(epochs, r2_scores, label="R²", color="lime", lw=2)
    ax2.set_title("R² Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R²")
    ax2.legend()
    fig.savefig(save_path)
    plt.close(fig)
