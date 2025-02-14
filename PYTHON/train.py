#!/usr/bin/env python
"""
train.py
---------
This module defines a Trainer class that encapsulates the training logic for the defocus regression model.
It logs training loss, validation loss, F1 score, MAE, and additionally computes accuracy, precision, and recall
(over epochs based on a threshold). At the end of training, it plots these metrics and saves all results (including
the best checkpoint and test results) in a dedicated folder.
Data is split into 70% training, 20% validation, and 10% testing using a fixed seed.
The script uses the Adam optimizer, a cosine annealing scheduler, mixed precision training,
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def compute_accuracy_within_threshold(preds, targets, threshold=0.1):
    """Computes the fraction of predictions with absolute error less than the threshold."""
    correct = (torch.abs(preds - targets) < threshold).float()
    return correct.mean().item()

def compute_mae(preds, targets):
    """Computes Mean Absolute Error (MAE) between predictions and targets."""
    return torch.mean(torch.abs(preds - targets)).item()

class Trainer:
    def __init__(self, model_name, train_dataset, val_dataset, device='cuda',
                 batch_size=32, learning_rate=1e-4, num_epochs=50, threshold=0.1,
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

        # Create a new folder for this training run: training/train-<model_name>-<timestamp>
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_folder = os.path.join(os.getcwd(), "training")
        os.makedirs(training_folder, exist_ok=True)
        self.run_folder = os.path.join(training_folder, f"train-{model_name}-{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)
        logging.info(f"Run folder created at {self.run_folder}")

        # Save hyperparameter settings to a config file
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
        # Using a cosine annealing LR scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        self.train_losses = []
        self.val_losses = []
        self.f1_scores = []  # computed as fraction within threshold
        self.mae_scores = []
        # For additional metrics (accuracy, precision, recall) we use the same value for simplicity
        self.accuracy_scores = []
        self.precision_scores = []
        self.recall_scores = []

        # Early stopping variables
        self.best_val_loss = float('inf')
        self.no_improvement_counter = 0

    def _get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=4)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=4)
        return train_loader, val_loader

    def train_one_epoch(self, train_loader, scaler):
        self.model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = self.model(images).squeeze(1)
                loss = self.criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
        return running_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                preds = self.model(images).squeeze(1)
                loss = self.criterion(preds, targets)
                val_loss += loss.item() * images.size(0)
                all_preds.append(preds)
                all_targets.append(targets)
        avg_loss = val_loss / len(val_loader.dataset)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metric = compute_accuracy_within_threshold(all_preds, all_targets, threshold=self.threshold)
        mae = compute_mae(all_preds, all_targets)
        return avg_loss, metric, mae

    def train(self):
        """A complete training loop that runs through epochs, saves the best model, and outputs graphs."""
        train_loader, val_loader = self._get_dataloaders()
        start_time = time.time()
        scaler = torch.amp.GradScaler()

        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch+1}/{self.num_epochs}")
            avg_train_loss = self.train_one_epoch(train_loader, scaler)
            self.train_losses.append(avg_train_loss)

            avg_val_loss, metric, mae = self.validate(val_loader)
            self.val_losses.append(avg_val_loss)
            self.f1_scores.append(metric)
            self.mae_scores.append(mae)
            self.accuracy_scores.append(metric)
            self.precision_scores.append(metric)
            self.recall_scores.append(metric)

            logging.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Metric: {metric:.4f} | MAE: {mae:.4f}")

            # Save model checkpoint if improvement is significant
            if self.best_val_loss - avg_val_loss > self.min_delta:
                self.best_val_loss = avg_val_loss
                self.no_improvement_counter = 0
                save_path = os.path.join(self.run_folder, f"best_model_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), save_path)
                logging.info(f"Best model saved at epoch {epoch+1} to {save_path}")
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
        return self.train_losses, self.val_losses, self.f1_scores, self.mae_scores

    def _save_results(self):
        try:
            epochs = list(range(1, len(self.train_losses) + 1))
            plt.style.use("dark_background")
            # Graph 1: Training & Validation Loss
            fig1, ax1 = plt.subplots()
            ax1.plot(epochs, self.train_losses, label="Train Loss", color="cyan", lw=2)
            ax1.plot(epochs, self.val_losses, label="Val Loss", color="magenta", lw=2)
            ax1.set_title("Training & Validation Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            loss_graph_path = os.path.join(self.run_folder, "loss.png")
            fig1.savefig(loss_graph_path)
            plt.close(fig1)
            logging.info(f"Loss graph saved at {loss_graph_path}")

            # Graph 2: F1 Score & MAE
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
            ax2[0].plot(epochs, self.f1_scores, label="Metric (Acc/Prec/Recall)", color="yellow", lw=2)
            ax2[0].set_title("Accuracy/Precision/Recall Over Epochs")
            ax2[0].set_xlabel("Epoch")
            ax2[0].set_ylabel("Metric")
            ax2[0].legend()
            ax2[1].plot(epochs, self.mae_scores, label="MAE", color="orange", lw=2)
            ax2[1].set_title("MAE Over Epochs")
            ax2[1].set_xlabel("Epoch")
            ax2[1].set_ylabel("MAE")
            ax2[1].legend()
            metrics_graph_path = os.path.join(self.run_folder, "metrics.png")
            fig2.savefig(metrics_graph_path)
            plt.close(fig2)
            logging.info(f"Metrics graph saved at {metrics_graph_path}")

            # Graph 3: Accuracy, Precision, & Recall
            fig3, ax3 = plt.subplots()
            ax3.plot(epochs, self.accuracy_scores, label="Accuracy", color="lime", lw=2)
            ax3.plot(epochs, self.precision_scores, label="Precision", color="cyan", lw=2)
            ax3.plot(epochs, self.recall_scores, label="Recall", color="magenta", lw=2)
            ax3.set_title("Accuracy, Precision, & Recall Over Epochs")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Score")
            ax3.legend()
            extra_graph_path = os.path.join(self.run_folder, "acc_prec_rec.png")
            fig3.savefig(extra_graph_path)
            plt.close(fig3)
            logging.info(f"Accuracy/Precision/Recall graph saved at {extra_graph_path}")

            metrics = {
                "epochs": epochs,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "metric": self.f1_scores,
                "mae": self.mae_scores,
                "accuracy": self.accuracy_scores,
                "precision": self.precision_scores,
                "recall": self.recall_scores
            }
            metrics_path = os.path.join(self.run_folder, "training_metrics.pkl")
            with open(metrics_path, "wb") as f:
                pickle.dump(metrics, f)
            logging.info(f"Training metrics saved at {metrics_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

# This main block is only executed when running train.py standalone.
# It does not interfere when the module is imported by your GUI.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defocus Regression Training")
    parser.add_argument("--model_name", type=str, default="resnet101", 
                        help="Model name to use (e.g., resnet101, mobilenetv3, efficientnet_lite, convnext)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--train_images_dir", type=str, required=True, 
                        help="Directory containing training images")
    parser.add_argument("--annotations_csv", type=str, required=True,
                        help="Path to CSV file with annotations (filename, defocus_microns)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from data import PipetteDataModule
    data_module = PipetteDataModule(args.train_images_dir, args.annotations_csv,
                                    train_split=0.7, val_split=0.2, test_split=0.1, seed=42)
    train_dataset, val_dataset, test_dataset = data_module.setup()

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
    trainer.train()

    # Test set evaluation
    test_loader = DataLoader(test_dataset, batch_size=trainer.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    trainer.model.eval()
    test_loss = 0.0
    all_test_preds, all_test_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(trainer.device, non_blocking=True)
            targets = targets.to(trainer.device, non_blocking=True)
            preds = trainer.model(images).squeeze(1)
            loss = trainer.criterion(preds, targets)
            test_loss += loss.item() * images.size(0)
            all_test_preds.append(preds)
            all_test_targets.append(targets)
    avg_test_loss = test_loss / len(test_loader.dataset)
    all_test_preds = torch.cat(all_test_preds)
    all_test_targets = torch.cat(all_test_targets)
    test_metric = compute_accuracy_within_threshold(all_test_preds, all_test_targets, threshold=trainer.threshold)
    test_mae = compute_mae(all_test_preds, all_test_targets)
    test_results = (f"Test Loss: {avg_test_loss:.4f}\n"
                    f"Test Metric (Acc/Prec/Recall): {test_metric:.4f}\n"
                    f"Test MAE: {test_mae:.4f}")
    logging.info(test_results)

    test_results_path = os.path.join(trainer.run_folder, "test_results.txt")
    try:
        with open(test_results_path, "w") as f:
            f.write(test_results)
        logging.info(f"Test results saved at {test_results_path}")
    except Exception as e:
        logging.error(f"Error saving test results: {e}")
