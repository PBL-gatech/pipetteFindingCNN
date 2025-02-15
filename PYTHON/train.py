#!/usr/bin/env python
"""
train.py
---------
This module defines a Trainer class that encapsulates the training logic for the defocus regression model.
It logs training loss, validation loss, accuracy within ±0.3 µm, MAE, and R^2. 
At the end of training, it plots these metrics and saves all results (including
the best checkpoint and current checkpoint) in a dedicated folder.
Data is split into 70% training, 20% validation, and 10% testing using a fixed seed.
The script uses the Adam optimizer, a step-based LR scheduler, mixed precision training,
and supports command-line configuration for testing different models.

This version of train.py maintains the interface required by your GUI frontend.
"""

import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import PipetteDataModule
from model import ModelFactory
import pickle
import torch.cuda.amp as amp
import random
import numpy as np
import logging
from tqdm import tqdm
import argparse

# Force matplotlib to use a non-interactive backend (important in headless environments)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Albumentations (for transforms)
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# Utility: Set Seeds
###############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

###############################################################################
# Normalization & Metrics
###############################################################################
SCALE_FACTOR = 40.0  # If defocus range is [-40, 40] microns

def normalize_z(z):
    """
    Convert z from [-40, 40] microns to [-1, 1].
    """
    return z / SCALE_FACTOR

def denormalize_z(z_norm):
    """
    Convert normalized z back to microns in [-40, 40].
    """
    return z_norm * SCALE_FACTOR

def compute_accuracy_within_threshold(preds, targets, threshold=0.3):
    """
    Computes the fraction of predictions with absolute error < threshold (in microns).
    Both preds and targets are in real space (microns).
    """
    correct = (torch.abs(preds - targets) < threshold).float()
    return correct.mean().item()

def compute_mae(preds, targets):
    """
    Computes Mean Absolute Error (MAE) between predictions and targets (both in microns).
    """
    return torch.mean(torch.abs(preds - targets)).item()

def compute_r2(preds, targets):
    """
    Computes R^2 for regression: 1 - SSR/SST
    preds, targets are real-space (microns).
    """
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2) + 1e-8
    r2 = 1 - ss_res / ss_tot
    return r2.item()

###############################################################################
# Trainer Class
###############################################################################
class Trainer:
    def __init__(self, model_name, train_dataset, val_dataset, device='cuda',
                 batch_size=32, learning_rate=1e-4, num_epochs=50, threshold=0.3,
                 min_delta=0.005, patience=15):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logging.info("Using CUDA" if torch.cuda.is_available() else "CUDA not available")
        
        self.model = ModelFactory.get_model(model_name, pretrained=True, output_dim=1).to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.min_delta = min_delta
        self.patience = patience

        # Create a new folder for this training run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_folder = os.path.join(os.getcwd(), "training")
        os.makedirs(training_folder, exist_ok=True)
        self.run_folder = os.path.join(training_folder, f"train-{model_name}-{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)
        logging.info(f"Run folder created at {self.run_folder}")

        # Save hyperparameters
        config = {
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "threshold": threshold,
            "min_delta": min_delta,
            "patience": patience,
            "device": device
        }
        config_path = os.path.join(self.run_folder, "config.txt")
        with open(config_path, "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        logging.info(f"Configuration saved to {config_path}")

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Step-based LR schedule: every 10 epochs, LR *= 0.1
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.accuracy_scores = []  # fraction within threshold
        self.mae_scores = []
        self.r2_scores = []

        # Early stopping
        self.best_val_loss = float('inf')
        self.no_improvement_counter = 0

    def _get_dataloaders(self):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=16, 
            pin_memory=True, 
            prefetch_factor=4,
            persistent_workers=True
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=16, 
            pin_memory=True, 
            prefetch_factor=4,
            persistent_workers=True
        )
        return train_loader, val_loader

    def train_one_epoch(self, train_loader, scaler):
        self.model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.device, non_blocking=True)

            # Normalize targets to [-1, 1]
            targets_norm = normalize_z(targets.to(self.device, non_blocking=True))

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs_norm = self.model(images).squeeze(1)  # raw output
                loss = self.criterion(outputs_norm, targets_norm)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        return running_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_preds_real, all_targets_real = [], []

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(self.device, non_blocking=True)

                # Keep real-space targets for metrics
                targets_real = targets.to(self.device, non_blocking=True).float()
                targets_norm = normalize_z(targets_real)

                preds_norm = self.model(images).squeeze(1)
                loss = self.criterion(preds_norm, targets_norm)
                val_loss += loss.item() * images.size(0)

                # Convert predictions to real space for metrics
                preds_real = denormalize_z(preds_norm)
                all_preds_real.append(preds_real)
                all_targets_real.append(targets_real)

        avg_loss = val_loss / len(val_loader.dataset)
        all_preds_real = torch.cat(all_preds_real)
        all_targets_real = torch.cat(all_targets_real)

        # Accuracy: fraction within ±0.3 µm
        acc = compute_accuracy_within_threshold(all_preds_real, all_targets_real, threshold=self.threshold)
        # MAE in microns
        mae = compute_mae(all_preds_real, all_targets_real)
        # R^2
        r2 = compute_r2(all_preds_real, all_targets_real)

        return avg_loss, acc, mae, r2

    def train(self):
        train_loader, val_loader = self._get_dataloaders()
        start_time = time.time()
        scaler = torch.amp.GradScaler()

        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch+1}/{self.num_epochs}")

            avg_train_loss = self.train_one_epoch(train_loader, scaler)
            self.train_losses.append(avg_train_loss)

            avg_val_loss, accuracy, mae, r2 = self.validate(val_loader)
            self.val_losses.append(avg_val_loss)
            self.accuracy_scores.append(accuracy)
            self.mae_scores.append(mae)
            self.r2_scores.append(r2)

            logging.info(
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Acc±{self.threshold}µm: {accuracy:.4f} | "
                f"MAE: {mae:.4f} | R^2: {r2:.4f}"
            )

            # ------------------------------------------------------------------
            # Save model checkpoints
            # ------------------------------------------------------------------
            # Always save "current_model.pth"
            current_path = os.path.join(self.run_folder, "current_model.pth")
            torch.save(self.model.state_dict(), current_path)

            # If this is the best so far, update "best_model.pth"
            if (self.best_val_loss - avg_val_loss) > self.min_delta:
                self.best_val_loss = avg_val_loss
                self.no_improvement_counter = 0
                best_path = os.path.join(self.run_folder, "best_model.pth")
                torch.save(self.model.state_dict(), best_path)
                logging.info(f"Best model updated at epoch {epoch+1}")
            else:
                self.no_improvement_counter += 1
                logging.info(f"No significant improvement for {self.no_improvement_counter} epoch(s).")

            if self.no_improvement_counter >= self.patience:
                logging.info("Early stopping triggered.")
                break

            self.scheduler.step()

        total_time = (time.time() - start_time) / 3600.0
        logging.info(f"Training complete in {total_time:.2f} hours")
        self._save_results()
        return self.train_losses, self.val_losses, self.accuracy_scores, self.mae_scores, self.r2_scores

    def _save_results(self):
        try:
            epochs = list(range(1, len(self.train_losses) + 1))
            plt.style.use("dark_background")

            # --- Figure 1: Training & Validation Loss ---
            fig1, ax1 = plt.subplots()
            ax1.plot(epochs, self.train_losses, label="Train Loss", color="cyan", lw=2)
            ax1.plot(epochs, self.val_losses, label="Val Loss", color="magenta", lw=2)
            ax1.set_title("Training & Validation Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss (MSE in normalized space)")
            ax1.legend()
            loss_graph_path = os.path.join(self.run_folder, "loss.png")
            fig1.savefig(loss_graph_path)
            plt.close(fig1)
            logging.info(f"Loss graph saved at {loss_graph_path}")

            # --- Figure 2: Accuracy & MAE ---
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
            ax2[0].plot(epochs, self.accuracy_scores, label=f"Accuracy ±{self.threshold}µm", color="yellow", lw=2)
            ax2[0].set_title("Accuracy Within Threshold Over Epochs")
            ax2[0].set_xlabel("Epoch")
            ax2[0].set_ylabel("Accuracy")
            ax2[0].legend()

            ax2[1].plot(epochs, self.mae_scores, label="MAE (µm)", color="orange", lw=2)
            ax2[1].set_title("MAE Over Epochs")
            ax2[1].set_xlabel("Epoch")
            ax2[1].set_ylabel("MAE (µm)")
            ax2[1].legend()

            metrics_graph_path = os.path.join(self.run_folder, "metrics.png")
            fig2.savefig(metrics_graph_path)
            plt.close(fig2)
            logging.info(f"Metrics graph saved at {metrics_graph_path}")

            # --- Figure 3: R^2 Over Epochs ---
            fig3, ax3 = plt.subplots()
            ax3.plot(epochs, self.r2_scores, label="R^2", color="lime", lw=2)
            ax3.set_title("R^2 Over Epochs")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("R^2")
            ax3.legend()
            r2_graph_path = os.path.join(self.run_folder, "r2.png")
            fig3.savefig(r2_graph_path)
            plt.close(fig3)
            logging.info(f"R^2 graph saved at {r2_graph_path}")

            # Save metrics to pickle
            metrics = {
                "epochs": epochs,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "accuracy": self.accuracy_scores,
                "mae": self.mae_scores,
                "r2": self.r2_scores
            }
            metrics_path = os.path.join(self.run_folder, "training_metrics.pkl")
            with open(metrics_path, "wb") as f:
                pickle.dump(metrics, f)
            logging.info(f"Training metrics saved at {metrics_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

###############################################################################
# Main: For Standalone CLI
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defocus Regression Training")
    parser.add_argument("--model_name", type=str, default="resnet101", 
                        help="Model name to use (e.g., resnet101, mobilenetv3, efficientnet_lite, convnext)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.3,  # ±0.3 µm
                        help="Threshold (in microns) for accuracy calculation.")
    parser.add_argument("--train_images_dir", type=str, required=True, 
                        help="Directory containing training images")
    parser.add_argument("--annotations_csv", type=str, required=True,
                        help="Path to CSV file with annotations (filename, defocus_microns)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # 1) Define Albumentations transforms for train vs. val/test
    # ------------------------------------------------------------------------
    def get_train_transform():
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def get_eval_transform():
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    # 2) Create PipetteDataModule with separate transforms
    data_module = PipetteDataModule(
        image_dir=args.train_images_dir,
        annotation_file=args.annotations_csv,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        seed=42,
        train_transform=get_train_transform(),
        val_transform=get_eval_transform(),
        test_transform=get_eval_transform()
    )

    # 3) Set up (load) the datasets
    train_dataset, val_dataset, test_dataset = data_module.setup()

    # 4) Create Trainer
    trainer = Trainer(
        model_name=args.model_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        threshold=args.threshold
    )

    # 5) Train
    trainer.train()

    # ------------------------------------------------------------------------
    # 6) Final Test Set Evaluation with Best Model
    # ------------------------------------------------------------------------
    best_model_path = os.path.join(trainer.run_folder, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(torch.load(best_model_path))
        logging.info("Loaded best_model.pth for final testing.")
    else:
        logging.warning("No best_model.pth found, using current_model.pth instead.")
        trainer.model.load_state_dict(torch.load(os.path.join(trainer.run_folder, "current_model.pth")))

    trainer.model.eval()
    test_loader = DataLoader(test_dataset, batch_size=trainer.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    test_loss = 0.0
    all_test_preds_real, all_test_targets_real = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(trainer.device, non_blocking=True)
            targets_real = targets.to(trainer.device, non_blocking=True).float()
            targets_norm = normalize_z(targets_real)

            preds_norm = trainer.model(images).squeeze(1)
            loss = trainer.criterion(preds_norm, targets_norm)
            test_loss += loss.item() * images.size(0)

            preds_real = denormalize_z(preds_norm)
            all_test_preds_real.append(preds_real)
            all_test_targets_real.append(targets_real)

    avg_test_loss = test_loss / len(test_loader.dataset)
    all_test_preds_real = torch.cat(all_test_preds_real)
    all_test_targets_real = torch.cat(all_test_targets_real)

    test_acc = compute_accuracy_within_threshold(all_test_preds_real, all_test_targets_real, threshold=args.threshold)
    test_mae = compute_mae(all_test_preds_real, all_test_targets_real)
    test_r2 = compute_r2(all_test_preds_real, all_test_targets_real)

    test_results = (
        f"Test Loss (MSE in normalized space): {avg_test_loss:.4f}\n"
        f"Test Accuracy ±{args.threshold}µm: {test_acc:.4f}\n"
        f"Test MAE (µm): {test_mae:.4f}\n"
        f"Test R^2: {test_r2:.4f}"
    )
    logging.info(test_results)

    test_results_path = os.path.join(trainer.run_folder, "test_results.txt")
    try:
        with open(test_results_path, "w") as f:
            f.write(test_results)
        logging.info(f"Test results saved at {test_results_path}")
    except Exception as e:
        logging.error(f"Error saving test results: {e}")
