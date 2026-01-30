#!/usr/bin/env python
"""
evaluation.py
-------------
Evaluates a previously trained model on the test set, using the same
random splits and seed as training, but without retraining. Supports 1–3
regression outputs (x, y, z) based on the annotation columns.
"""

import glob
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import PipetteDataModule
from model import get_regression_model
from train import test_model


def _pick_checkpoint(run_folder, checkpoint_name=None):
    if checkpoint_name:
        candidate = os.path.join(run_folder, checkpoint_name)
        if os.path.isfile(candidate):
            return candidate
    pattern = os.path.join(run_folder, "best_model*.pth")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found in {run_folder}")
    return matches[0]


def evaluate_trained_model(
    run_folder,
    images_dir,
    annotations_csv,
    model_name="mobilenetv3_large_100",
    device="cuda",
    batch_size=32,
    seed=42,
    checkpoint_name=None
):
    """
    Evaluates the model saved in `run_folder` on the test split of your dataset.
    The dataset is re-split with the same seed, so you get the exact same test set as in training.
    """

    logging.info("Setting up data module with test_split=1.0 for evaluation")
    data_module = PipetteDataModule(
        image_dir=images_dir,
        annotation_file=annotations_csv,
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,
        seed=seed
    )
    _, _, test_dataset = data_module.setup()
    output_dim = data_module.target_dim

    ckpt_path = _pick_checkpoint(run_folder, checkpoint_name)
    logging.info(f"Loading checkpoint: {ckpt_path}")

    model = get_regression_model(model_name=model_name, pretrained=False, output_dim=output_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    criterion = nn.MSELoss()

    logging.info("Evaluating best model on the test set...")
    results = test_model(model, test_loader, device, criterion, run_folder)

    logging.info("Done evaluating!")
    logging.info(f"Test Loss (MSE): {results['Test Loss']:.4f}")
    logging.info(f"Test MAE:        {results['Test MAE']:.4f}")
    logging.info(f"Test RÂ²:        {results['Test RÂ²']:.4f}")
    return results


if __name__ == "__main__":
    # You can adjust these paths/values as needed
    logging.basicConfig(level=logging.INFO)

    run_folder = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilenetv3-20250215_160856"  # Folder where best_model*.pth is saved
    images_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\20191016"                           # Same images used in training
    annotations_csv = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\processed_20191016_final.csv"             # Same CSV used in training

    evaluate_trained_model(
        run_folder=run_folder,
        images_dir=images_dir,
        annotations_csv=annotations_csv,
        model_name="mobilenetv3_large_100",
        device="cuda",
        batch_size=16,
        seed=42
    )
