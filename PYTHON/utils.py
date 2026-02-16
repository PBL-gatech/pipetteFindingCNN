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

def _default_imagenet_stats():
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_train_transform(img_size=224, mean=None, std=None, blur_prob=0.1):
    """
    Returns an Albumentations Compose transform for training that applies:
      - Resize to img_size (no random crop to preserve focus context)
      - Horizontal flip and occasional 90° rotation
      - Light Gaussian blur (reduced strength / probability)
      - Normalize with provided mean/std (defaults to ImageNet if None)
      - Conversion to a PyTorch tensor
    """
    mean = mean if mean is not None else _default_imagenet_stats()[0]
    std = std if std is not None else _default_imagenet_stats()[1]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.5), p=blur_prob),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_val_transform(img_size=224, mean=None, std=None):
    """
    Returns an Albumentations Compose transform for validation/testing that applies:
      - Resize to img_size x img_size
      - Normalize with provided mean/std (defaults to ImageNet if None)
      - Conversion to a PyTorch tensor
    """
    mean = mean if mean is not None else _default_imagenet_stats()[0]
    std = std if std is not None else _default_imagenet_stats()[1]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
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

def plot_regression_metrics(
    epochs,
    mae_scores,
    r2_scores,
    save_path,
    mae_scores_pos=None,
    mae_scores_neg=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, mae_scores, label="MAE", color="orange", lw=2)
    if mae_scores_pos is not None:
        ax1.plot(epochs, mae_scores_pos, label="MAE positive", color="cyan", lw=1.8, alpha=0.9)
    if mae_scores_neg is not None:
        ax1.plot(epochs, mae_scores_neg, label="MAE negative", color="magenta", lw=1.8, alpha=0.9)
    ax1.set_title("Mean Absolute Error Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE (microns)")
    ax1.legend()
    ax2.plot(epochs, r2_scores, label="R^2", color="lime", lw=2)
    ax2.set_title("R^2 Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R^2")
    ax2.legend()
    fig.savefig(save_path)
    plt.close(fig)
