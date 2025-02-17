#!/usr/bin/env python3
import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import get_regression_model  # Creates model with a regression head
from data import PipetteDataModule  # Your data module with 70/20/10 split
from utils import plot_training_metrics, plot_regression_metrics, get_train_transform, get_val_transform
import timm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

# --- Run folder creation ---
def create_run_folder(model_name, config=None):
    training_dir = os.path.join(os.getcwd(), "training")
    os.makedirs(training_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(training_dir, f"train-{model_name}-{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    # Save the configuration if provided
    if config is not None:
        config_path = os.path.join(run_folder, "config.txt")
        with open(config_path, "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
    return run_folder


# --- Training and Validation Loop with timm optimizations ---
def train_and_validate(model, train_loader, val_loader, device, run_folder, num_epochs=50, update=False, update_callback=None):
    # Create the optimizer directly
    optimizer = create_optimizer_v2(model, "adamw", lr=1e-4, weight_decay=2e-5)
    
    # Create the scheduler directly (returns a tuple, we only need the scheduler)
    scheduler, _ = create_scheduler_v2(optimizer, "cosine", num_epochs=num_epochs, updates_per_epoch=len(train_loader))
    
    criterion = nn.MSELoss()  # Regression loss
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float('inf')
    best_checkpoint = None
    
    train_losses, val_losses = [], []
    mae_scores, r2_scores = [], []
    epochs_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(enabled=(scaler is not None),device_type='cuda'):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, targets)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                with torch.amp.autocast(enabled=(scaler is not None),device_type='cuda'):
                    outputs = model(images).squeeze(1)
                    loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        preds = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        mae = torch.mean(torch.abs(preds - targets_cat)).item()
        ss_res = torch.sum((targets_cat - preds)**2)
        ss_tot = torch.sum((targets_cat - torch.mean(targets_cat))**2) + 1e-8
        r2 = 1 - ss_res / ss_tot

        mae_scores.append(mae)
        r2_scores.append(r2.item())
        epochs_list.append(epoch+1)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")
        
        scheduler.step(epoch + 1)

        if update and (update_callback is not None):
            update_callback(epoch+1, train_loss, val_loss, mae, r2.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = os.path.join(run_folder, f"best_model_focus_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_checkpoint)
            print(f"Best model updated and saved to {best_checkpoint}")

        plot_training_metrics(epochs_list, train_losses, val_losses, os.path.join(run_folder, "loss.png"))
        plot_regression_metrics(epochs_list, mae_scores, r2_scores, os.path.join(run_folder, "metrics.png"))
    
    return best_checkpoint, epochs_list, train_losses, val_losses, mae_scores, r2_scores

# --- Testing Function ---
def test_model(model, test_loader, device, criterion, run_folder):
    model.eval()
    test_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * images.size(0)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    test_loss /= len(test_loader.dataset)
    preds = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    mae = torch.mean(torch.abs(preds - targets_cat)).item()
    ss_res = torch.sum((targets_cat - preds)**2)
    ss_tot = torch.sum((targets_cat - torch.mean(targets_cat))**2) + 1e-8
    r2 = 1 - ss_res/ss_tot
    results = {"Test Loss": test_loss, "Test MAE": mae, "Test R²": r2.item()}
    with open(os.path.join(run_folder, "test_results.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print("Final test results saved to", os.path.join(run_folder, "test_results.txt"))
    return results

# --- Main ---
if __name__ == '__main__':
    image_dir = "path/to/images"
    annotation_file = "path/to/annotations.csv"
    
    data_module = PipetteDataModule(
        image_dir, 
        annotation_file,
        train_split=0.7, 
        val_split=0.2, 
        test_split=0.1, 
        seed=42,
        train_transform=get_train_transform(img_size=224),
        val_transform=get_val_transform(img_size=224),
        test_transform=get_val_transform(img_size=224)
    )
    train_dataset, val_dataset, test_dataset = data_module.setup()
    
    model_name = "mobilenetv3_large_100"
    config = {
        "model_name": model_name,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "threshold": 0.3,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    run_folder = create_run_folder(model_name, config=config)
    print("Run folder created at:", run_folder)
    
    model = get_regression_model(model_name=model_name, pretrained=True, output_dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    best_checkpoint, epochs_list, train_losses, val_losses, mae_scores, r2_scores = train_and_validate(
        model, train_loader, val_loader, device, run_folder, num_epochs=50
    )
    
    model.load_state_dict(torch.load(best_checkpoint))
    criterion = nn.MSELoss()
    test_results = test_model(model, test_loader, device, criterion, run_folder)