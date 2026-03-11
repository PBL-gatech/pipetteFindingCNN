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


def contrast_stretch_mu_2sigma_uint8(image: np.ndarray) -> np.ndarray:
    """
    Paper-style per-image contrast stretch:
      low = mu - 2*sigma
      high = mu + 2*sigma
      out = clip((x - low) / (high - low), 0, 1)
    Returns uint8 image in [0, 255].
    """
    img = np.asarray(image)
    if img.size == 0:
        return img.copy()

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            max_val = float(np.nanmax(img)) if img.size else 0.0
            scale = 255.0 if max_val <= 1.0 else 1.0
            img_uint8 = np.clip(img * scale, 0, 255).astype(np.uint8)
        else:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    img_float = img_uint8.astype(np.float32)
    mu = float(np.mean(img_float))
    sigma = float(np.std(img_float))

    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-6:
        return img_uint8.copy()

    low = mu - 2.0 * sigma
    high = mu + 2.0 * sigma
    if high <= low + 1e-6:
        return img_uint8.copy()

    stretched = (img_float - low) / (high - low)
    stretched = np.clip(stretched, 0.0, 1.0)
    return np.round(stretched * 255.0).astype(np.uint8)


def _albumentations_contrast_stretch(image, **kwargs):
    return contrast_stretch_mu_2sigma_uint8(image)


def get_train_transform(
    img_size=224,
    mean=None,
    std=None,
    enable_contrast_stretch: bool = False,
    enable_aug_flip_rotate: bool = False,
):
    """
    Returns an Albumentations Compose transform for training that applies:
      - Optional contrast stretching (mu +/- 2*sigma)
      - Resize to img_size (no random crop to preserve focus context)
      - Horizontal and vertical flips (independent, so H+V can occur)
      - HSV jitter and optional 90-degree rotation
      - Normalize with provided mean/std (defaults to ImageNet if None)
      - Conversion to a PyTorch tensor
    """
    mean = mean if mean is not None else _default_imagenet_stats()[0]
    std = std if std is not None else _default_imagenet_stats()[1]
    transforms = []
    if enable_contrast_stretch:
        transforms.append(A.Lambda(image=_albumentations_contrast_stretch))

    transforms.append(A.Resize(img_size, img_size))

    transforms.extend(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=18,
                val_shift_limit=12,
                p=0.5,
            ),
        ]
    )
    if enable_aug_flip_rotate:
        transforms.append(A.RandomRotate90(p=0.3))

    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def get_val_transform(img_size=224, mean=None, std=None, enable_contrast_stretch: bool = False):
    """
    Returns an Albumentations Compose transform for validation/testing that applies:
      - Optional contrast stretching (mu +/- 2*sigma)
      - Resize to img_size x img_size
      - Normalize with provided mean/std (defaults to ImageNet if None)
      - Conversion to a PyTorch tensor
    """
    mean = mean if mean is not None else _default_imagenet_stats()[0]
    std = std if std is not None else _default_imagenet_stats()[1]
    transforms = []
    if enable_contrast_stretch:
        transforms.append(A.Lambda(image=_albumentations_contrast_stretch))
    transforms.extend(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)

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
    mae_scores_inner=None,
    mae_scores_outer=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, mae_scores, label="MAE", color="orange", lw=2)
    if mae_scores_pos is not None:
        ax1.plot(epochs, mae_scores_pos, label="MAE positive", color="cyan", lw=1.8, alpha=0.9)
    if mae_scores_neg is not None:
        ax1.plot(epochs, mae_scores_neg, label="MAE negative", color="magenta", lw=1.8, alpha=0.9)
    if mae_scores_inner is not None:
        ax1.plot(epochs, mae_scores_inner, label="MAE inner-band", color="yellow", lw=1.8, alpha=0.9)
    if mae_scores_outer is not None:
        ax1.plot(epochs, mae_scores_outer, label="MAE outer-band", color="deepskyblue", lw=1.8, alpha=0.9)
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


def plot_predictions_vs_targets(targets, predictions, save_path):
    """
    Save a scatter plot of predicted vs. true values with an ideal y=x line.
    Values are expected in microns (denormalized space).
    """
    targets_np = np.asarray(targets, dtype=np.float32).reshape(-1)
    preds_np = np.asarray(predictions, dtype=np.float32).reshape(-1)

    if targets_np.size == 0 or preds_np.size == 0:
        return

    n = min(targets_np.size, preds_np.size)
    targets_np = targets_np[:n]
    preds_np = preds_np[:n]

    combined = np.concatenate([targets_np, preds_np])
    axis_min = float(np.min(combined))
    axis_max = float(np.max(combined))
    padding = max((axis_max - axis_min) * 0.05, 1e-6)
    line_min = axis_min - padding
    line_max = axis_max + padding

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(
        targets_np,
        preds_np,
        s=14,
        alpha=0.65,
        color="deepskyblue",
        edgecolors="none",
        label="Test samples",
    )
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=1.6,
        color="red",
        label="Ideal (y=x)",
    )
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True z (microns)")
    ax.set_ylabel("Predicted z (microns)")
    ax.set_title("Test Predictions vs Ground Truth")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

