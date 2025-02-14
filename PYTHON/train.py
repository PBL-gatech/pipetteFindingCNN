"""
train.py
---------
This module defines a Trainer class that encapsulates the training logic for the defocus regression model.
It logs training loss, validation loss, and an "F1 score" (computed as the fraction of predictions
with absolute error less than a specified threshold). At the end of training, it plots these metrics.
Data is split into 70% training, 20% validation, and 10% testing using a fixed seed.
It now uses the Adam optimizer, a larger batch size, increased number of DataLoader workers,
and prints the CUDA status. GPU utilization improvements are attempted via enabling cuDNN benchmarking.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import PipetteDataModule
from model import ModelFactory
import matplotlib.pyplot as plt
import pickle
import torch.cuda.amp as amp

def compute_accuracy_within_threshold(preds, targets, threshold=0.1):
    """
    Computes the fraction of predictions with absolute error less than the threshold.
    This metric serves as a proxy for accuracy.
    """
    correct = (torch.abs(preds - targets) < threshold).float()
    return correct.mean().item()

class Trainer:
    def __init__(self, model_name, train_dataset, val_dataset, device='cuda',
                 batch_size=32, learning_rate=1e-4, num_epochs=50, threshold=0.1,
                 model_save_folder=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using CUDA")
        else:
            print("CUDA not available")
            
        # Instantiate the single-output model for defocus regression
        self.model = ModelFactory.get_model(model_name, pretrained=True, output_dim=1).to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold

        # If no folder specified, create a "checkpoints" folder in the current working directory.
        if model_save_folder is None:
            self.model_save_folder = os.path.join(os.getcwd(), "checkpoints")
        else:
            self.model_save_folder = model_save_folder
        os.makedirs(self.model_save_folder, exist_ok=True)

        self.criterion = nn.MSELoss()  # Using MSE Loss for single-value regression
        # Use Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_losses = []
        self.val_losses = []
        self.f1_scores = []

    def _get_dataloaders(self):
        # Increase num_workers to 8 for faster data loading
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)
        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self._get_dataloaders()
        best_val_loss = float('inf')
        start_time = time.time()

        scaler = torch.amp.GradScaler(device=self.device)  # Updated AMP scaler usage

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, targets in train_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    outputs = outputs.squeeze(1)  # Convert to shape: (batch_size,)
                    loss = self.criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)

            avg_train_loss = running_loss / len(train_loader.dataset)
            self.train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    preds = self.model(images).squeeze(1)
                    loss = self.criterion(preds, targets)
                    val_loss += loss.item() * images.size(0)
                    all_preds.append(preds)
                    all_targets.append(targets)

            avg_val_loss = val_loss / len(val_loader.dataset)
            self.val_losses.append(avg_val_loss)
            # Concatenate predictions and targets to compute the F1-like metric
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            epoch_f1 = compute_accuracy_within_threshold(all_preds, all_targets, threshold=self.threshold)
            self.f1_scores.append(epoch_f1)

            print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1: {epoch_f1:.4f}")

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(self.model_save_folder, f"best_model_focus_{epoch+1}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch+1} to {save_path}")

        total_time = (time.time() - start_time) / 3600.0
        print(f"Training complete in {total_time:.2f} hours")

        # Plot the metrics after training is done
        epochs = list(range(1, self.num_epochs + 1))
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(epochs, self.train_losses, label="Train Loss", color="cyan", lw=2)
        ax1.plot(epochs, self.val_losses, label="Val Loss", color="magenta", lw=2)
        ax1.set_title("Training & Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(epochs, self.f1_scores, label="F1 Score", color="yellow", lw=2)
        ax2.set_title("F1 Score Over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1 Score")
        ax2.legend()

        fig.tight_layout()
        plt.show()

        return self.train_losses, self.val_losses, self.f1_scores

if __name__ == "__main__":
    # Enable cuDNN benchmarking for potentially higher GPU utilization (if input sizes are constant)
    torch.backends.cudnn.benchmark = True

    from torch.utils.data import DataLoader  # Needed for test evaluation below

    # Example usage:
    train_images_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\2025_02_14-13_10\camera_frames"
    annotations_csv = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\pipette_z_data21425.csv"  # CSV columns: filename, defocus_microns

    # Create the data module with a 70/20/10 split using a fixed seed
    from data import PipetteDataModule
    data_module = PipetteDataModule(train_images_dir, annotations_csv,
                                    train_split=0.7, val_split=0.2, test_split=0.1, seed=42)
    train_dataset, val_dataset, test_dataset = data_module.setup()

    # Specify the checkpoints folder (if you want a custom location, change the path here)
    checkpoints_folder = os.path.join(os.getcwd(), "checkpoints")

    trainer = Trainer(
        model_name="resnet101",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device='cuda',
        batch_size=32,        # Increased batch size
        learning_rate=1e-4,
        num_epochs=50,        # Number of epochs
        threshold=0.1,        # Tolerance threshold for computing the F1-like metric
        model_save_folder=checkpoints_folder  # Specify folder for saving models
    )
    trainer.train()

    # Evaluate on the test set after training
    test_loader = DataLoader(test_dataset, batch_size=trainer.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    trainer.model.eval()
    test_loss = 0.0
    all_test_preds = []
    all_test_targets = []
    with torch.no_grad():
        for images, targets in test_loader:
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
    test_f1 = compute_accuracy_within_threshold(all_test_preds, all_test_targets, threshold=trainer.threshold)
    print(f"Test Loss: {avg_test_loss:.4f}, Test F1: {test_f1:.4f}")

    # Optionally, pickle the training metrics for later analysis
    metrics = {
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "f1_scores": trainer.f1_scores
    }
    with open("training_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
