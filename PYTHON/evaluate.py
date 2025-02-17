#!/usr/bin/env python
"""
evaluation.py
-------------
Evaluates a previously trained model on the test set, using the same
random splits and seed as training, but without retraining.
"""

import logging
import os

import torch

# Import your data module and Trainer
from data import PipetteDataModule
from train import Trainer  # Trainer must have the 'test_model' method

def evaluate_trained_model(
    run_folder,
    images_dir,
    annotations_csv,
    model_name="mobilenetv3",
    device="cuda",
    batch_size=32,
    threshold=0.3,
    seed=42
):
    """
    Evaluates the model saved in `run_folder` (best_model.pth or current_model.pth)
    on the test split of your dataset. The dataset is re-split with the same seed,
    so you get the exact same test set as in training.
    """

    logging.info("Setting up data module with train_split=0.7, val_split=0.2, test_split=0.1")
    data_module = PipetteDataModule(
        image_dir=images_dir,
        annotation_file=annotations_csv,
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,
        seed=seed
    )
    # This replicates the same splits & random seed as in training
    _, _, test_dataset = data_module.setup()

    # Create a Trainer object (won't actually train—just for loading/checkpoint & testing)
    trainer = Trainer(
        model_name=model_name,
        train_dataset=None,   # Not needed for evaluation
        val_dataset=None,     # Not needed for evaluation
        device=device,
        batch_size=batch_size,
        learning_rate=1e-4,   # Not used here, but required
        num_epochs=1,         # Not used here
        threshold=threshold
    )

    # Point the Trainer to the folder that has best_model.pth/current_model.pth
    trainer.run_folder = run_folder

    logging.info("Evaluating best model on the test set...")
    test_loss, test_acc, test_mae, test_r2 = trainer.test_model(test_dataset,best_model="best_model_focus_22.pth")

    logging.info("Done evaluating!")
    logging.info(f"Test Loss (MSE in normalized space): {test_loss:.4f}")
    logging.info(f"Test Accuracy ±{threshold}µm:       {test_acc:.4f}")
    logging.info(f"Test MAE (µm):                      {test_mae:.4f}")
    logging.info(f"Test R²:                             {test_r2:.4f}")

if __name__ == "__main__":
    # You can adjust these paths/values as needed
    logging.basicConfig(level=logging.INFO)

    run_folder = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilenetv3-20250215_160856"  # Folder where best_model.pth is saved
    images_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\20191016"                           # Same images used in training
    annotations_csv = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\processed_20191016_final.csv"             # Same CSV used in training

    evaluate_trained_model(
        run_folder=run_folder,
        images_dir=images_dir,
        annotations_csv=annotations_csv,
        model_name="mobilenetv3",
        device="cuda",
        batch_size=16,
        threshold=0.5,
        seed=42
    )
