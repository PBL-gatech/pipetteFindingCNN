import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
import time

# Define the function for training the CNN
def train_cnn(train_dir, val_dir, use_resnet=True, batch_size=16, epochs=60, learning_rate=1e-4):
    """
    Train a CNN for regression using a pretrained model.

    Parameters:
    - train_dir (str): Path to the training dataset.
    - val_dir (str): Path to the validation dataset.
    - use_resnet (bool): If True, use ResNet101; if False, use NASNet (not supported here).
    - batch_size (int): Size of each training batch.
    - epochs (int): Number of epochs for training.
    - learning_rate (float): Initial learning rate for training.
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained network
    if use_resnet:
        model = models.resnet101(pretrained=True)
        print("Using ResNet101")
    else:
        raise NotImplementedError("NASNetLarge is not directly available in torchvision.")
    
    # Modify the last layers to match regression task (replace classification layers)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, 3)  # Predicting x, y, z coordinates
    )
    model = model.to(device)
    
    # Data augmentation and normalization for training and validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Using standard ImageNet normalization
    ])
    
    # Load training and validation datasets
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')
    print("Beginning Training...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # Move data to the GPU
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Loss calculation
            # Assuming labels have the format [batch_size, 3] for x, y, z coordinates
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Print epoch statistics
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_epoch_{epoch + 1}.pth")
            print(f"Model saved at epoch {epoch + 1}")

    total_time = (time.time() - start_time) / 3600
    print(f"Training complete in {total_time:.2f} hours")

# Example usage
if __name__ == "__main__":
    train_dir = "/path/to/train_data"
    val_dir = "/path/to/val_data"
    train_cnn(train_dir, val_dir)
