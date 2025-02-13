"""
train.py
---------
This module defines a Trainer class that encapsulates the training logic for the pipette tip regression model.
"""

# train.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PYTHON.data import PipetteDataModule
from model import ModelFactory

class Trainer:
    def __init__(self, model_name, train_dataset, val_dataset, device='cuda',
                 batch_size=16, learning_rate=1e-4, num_epochs=10):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ModelFactory.get_model(model_name, pretrained=True).to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()  # Single-value regression
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def _get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=4)
        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self._get_dataloaders()
        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, targets in train_loader:
                images = images.to(self.device)               # [batch_size, 3, 224, 224]
                targets = targets.to(self.device)             # [batch_size]
                self.optimizer.zero_grad()

                # model outputs shape [batch_size, 1]
                outputs = self.model(images)  # shape: (B,1)
                outputs = outputs.squeeze(1)  # shape: (B,) to match MSELoss input

                loss = self.criterion(outputs, targets) 
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            avg_train_loss = running_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    preds = self.model(images).squeeze(1)
                    loss = self.criterion(preds, targets)
                    val_loss += loss.item() * images.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)

            print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"best_model_focus_{epoch+1}.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch+1} to {save_path}")

        total_time = (time.time() - start_time) / 3600.0
        print(f"Training complete in {total_time:.2f} hours")


if __name__ == "__main__":
    # Example usage:
    train_images_dir = "/path/to/train_images"
    annotations_csv = "/path/to/defocus_annotations.csv"  # columns: filename, defocus_microns

    data_module = PipetteDataModule(train_images_dir, annotations_csv, val_split=0.1)
    train_dataset, val_dataset = data_module.setup()

    trainer = Trainer(
        model_name="resnet101",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device='cuda',
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=10
    )
    trainer.train()
