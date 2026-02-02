#!/usr/bin/env python3
import os
import json
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model import get_regression_model  # Creates model with a regression head
from data import PipetteDataModule  # Your data module with 70/20/10 split
from utils import plot_training_metrics, plot_regression_metrics, get_train_transform, get_val_transform
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from functools import partial

# Enable faster CUDA kernels when available
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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


def default_loader_kwargs(device, cache_images: bool = False):
    # When caching images, keep workers at 0 to avoid duplicating the cache per process
    if cache_images:
        return {"num_workers": 0, "pin_memory": device.type == "cuda"}
    num_workers = os.cpu_count() or 0
    num_workers = max(2, min(num_workers - 2, 16)) if num_workers > 0 else 0
    kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda"}
    if num_workers > 0:
        # persistent_workers can misbehave on Windows; enable only on POSIX
        if os.name != "nt":
            kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def compute_focus_params(dataset, inner_band_um: float, manual_weight_ratio: float | None = None,
                         outer_band_um: float | None = None):
    """
    Derive outer band and weight ratio:
      - outer band defaults to the max |z| seen in the dataset (fallback: inner band)
      - weight ratio defaults to outer/inner so inner samples are penalized more
      - manual_weight_ratio overrides auto when provided (>0)
    """
    annotations = getattr(dataset, "annotations", None)
    if outer_band_um is None:
        if annotations:
            max_abs = max(abs(z) for _, z in annotations)
        else:
            max_abs = inner_band_um
        outer_band_um = max_abs
    if manual_weight_ratio is not None and manual_weight_ratio > 0:
        weight_ratio = manual_weight_ratio
    else:
        weight_ratio = max(outer_band_um / max(inner_band_um, 1e-6), 1.0)
    return outer_band_um, weight_ratio


def weighted_huber_loss(outputs, targets, beta, z_mean, z_std, inner_band_um, weight_ratio):
    """
    Per-sample Huber (SmoothL1) with higher weight inside the focus band.
    Targets/outputs are in normalized space; weighting is computed in microns.
    """
    microns = targets * z_std + z_mean
    inner_w = torch.as_tensor(weight_ratio, device=targets.device, dtype=outputs.dtype)
    outer_w = torch.as_tensor(1.0, device=targets.device, dtype=outputs.dtype)
    weights = torch.where(torch.abs(microns) <= inner_band_um, inner_w, outer_w)
    per_sample = F.smooth_l1_loss(outputs, targets, beta=beta, reduction="none")
    return (weights * per_sample).mean()


# --- Training and Validation Loop with timm optimizations ---
def train_and_validate(model, train_loader, val_loader, device, run_folder,
                       num_epochs=50, update=False, update_callback=None,
                       huber_beta: float = 1.0, learning_rate: float = 1e-4,
                       logger=print, compile_model: bool = False,
                       focus_inner_um: float = 3.0,
                       focus_outer_um: float | None = None,
                       focus_weight_ratio: float | None = None):
    # Ensure model is on device and in channels-last for better GPU throughput
    logger(f"Device selected: {device} | CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        try:
            logger(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    else:
        logger("CUDA not in use; training on CPU.")
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logger("Model compiled with torch.compile")
        except Exception as e:
            logger(f"torch.compile failed, continuing without it: {e}")
    # Create the optimizer directly
    optimizer = create_optimizer_v2(model, "adamw", lr=learning_rate, weight_decay=2e-5)
    
    # Create the scheduler with a short warmup to stabilize training
    scheduler, _ = create_scheduler_v2(
        optimizer,
        "cosine",
        num_epochs=num_epochs,
        updates_per_epoch=len(train_loader),
        warmup_epochs=5,
    )
    
    # Weighted Huber (SmoothL1) emphasizing the inner focus band
    z_mean = getattr(train_loader.dataset, "z_mean", 0.0)
    z_std = getattr(train_loader.dataset, "z_std", 1.0)
    focus_outer_um, focus_weight_ratio = compute_focus_params(
        train_loader.dataset, focus_inner_um, manual_weight_ratio=focus_weight_ratio,
        outer_band_um=focus_outer_um
    )
    logger(f"Weighted Huber: inner |z| <= {focus_inner_um}µm weighted x{focus_weight_ratio:.2f}; "
           f"outer span up to ~{focus_outer_um:.2f}µm; beta={huber_beta}")

    criterion = partial(
        weighted_huber_loss,
        beta=huber_beta,
        z_mean=z_mean,
        z_std=z_std,
        inner_band_um=focus_inner_um,
        weight_ratio=focus_weight_ratio,
    )
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float('inf')
    best_checkpoint = None
    
    train_losses, val_losses = [], []
    mae_scores, r2_scores = [], []
    epochs_list = []

    num_updates = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device=device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(enabled=(scaler is not None), device_type=device.type):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, targets)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            num_updates += 1
            scheduler.step_update(num_updates=num_updates)
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Val   {epoch+1}/{num_epochs}", leave=False):
                images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device=device, non_blocking=True)
                with torch.amp.autocast(enabled=(scaler is not None), device_type=device.type):
                    outputs = model(images).squeeze(1)
                    loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        preds = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)

        # Denormalize to microns for reporting
        z_mean = getattr(train_loader.dataset, "z_mean", 0.0)
        z_std = getattr(train_loader.dataset, "z_std", 1.0)
        preds_real = preds * z_std + z_mean
        targets_real = targets_cat * z_std + z_mean

        mae = torch.mean(torch.abs(preds_real - targets_real)).item()
        ss_res = torch.sum((targets_real - preds_real) ** 2)
        ss_tot = torch.sum((targets_real - torch.mean(targets_real)) ** 2) + 1e-8
        r2 = 1 - ss_res / ss_tot

        mae_scores.append(mae)
        r2_scores.append(r2.item())
        epochs_list.append(epoch+1)

        epoch_time = time.time() - epoch_start
        imgs_per_sec = len(train_loader.dataset) / epoch_time if epoch_time > 0 else 0.0
        logger(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  MAE: {mae:.4f}  R^2: {r2:.4f}  | imgs/sec: {imgs_per_sec:.1f}")
        
        scheduler.step(epoch + 1)

        if update and (update_callback is not None):
            update_callback(epoch+1, train_loss, val_loss, mae, r2.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = os.path.join(run_folder, f"best_model_focus_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_checkpoint)
            logger(f"Best model updated and saved to {best_checkpoint}")

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
            images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device=device, non_blocking=True)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * images.size(0)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    test_loss /= len(test_loader.dataset)
    preds = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)

    # Denormalize to microns for reporting
    z_mean = getattr(test_loader.dataset, "z_mean", 0.0)
    z_std = getattr(test_loader.dataset, "z_std", 1.0)
    preds_real = preds * z_std + z_mean
    targets_real = targets_cat * z_std + z_mean

    mae = torch.mean(torch.abs(preds_real - targets_real)).item()
    ss_res = torch.sum((targets_real - preds_real) ** 2)
    ss_tot = torch.sum((targets_real - torch.mean(targets_real)) ** 2) + 1e-8
    r2 = 1 - ss_res/ss_tot
    results = {"Test Loss": test_loss, "Test MAE": mae, "Test R2": r2.item()}
    with open(os.path.join(run_folder, "test_results.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print("Final test results saved to", os.path.join(run_folder, "test_results.txt"))
    return results

# --- Main ---
if __name__ == '__main__':
    image_dir = "path/to/images"
    annotation_file = "path/to/annotations.csv"
    model_name = "mobilenetv3_large_100"
    learning_rate = 1e-4
    num_epochs = 50
    huber_beta = 1.0
    focus_inner_um = 3.0
    config = {
        "model_name": model_name,
        "batch_size": 32,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "img_size": 224,
        "huber_beta": huber_beta,
        "focus_inner_um": focus_inner_um,
    }
    run_folder = create_run_folder(model_name, config=config)
    print("Run folder created at:", run_folder)

    cache_images = True  # enable RAM cache for CLI example
    data_module = PipetteDataModule(
        image_dir,
        annotation_file,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        seed=42,
        default_img_size=224,
        split_save_path=os.path.join(run_folder, "data_splits.pkl"),
        channel_stats_cache_path=os.path.join(run_folder, "channel_stats.pkl"),
        channel_stats_max_images=2000,
        cache_images=cache_images,
    )
    train_dataset, val_dataset, test_dataset = data_module.setup()
    focus_outer_um, focus_weight_ratio = compute_focus_params(train_dataset, focus_inner_um)

    # Persist channel normalization stats for inference
    if data_module.channel_mean is not None and data_module.channel_std is not None:
        with open(os.path.join(run_folder, "channel_norm.json"), "w") as f:
            json.dump({"mean": data_module.channel_mean, "std": data_module.channel_std}, f, indent=2)
    
    model = get_regression_model(model_name=model_name, pretrained=True, output_dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, memory_format=torch.channels_last)
    
    loader_kwargs = default_loader_kwargs(device, cache_images=cache_images)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, **loader_kwargs)
    
    best_checkpoint, epochs_list, train_losses, val_losses, mae_scores, r2_scores = train_and_validate(
        model, train_loader, val_loader, device, run_folder, num_epochs=num_epochs,
        learning_rate=learning_rate, huber_beta=huber_beta,
        focus_inner_um=focus_inner_um, focus_outer_um=focus_outer_um,
        focus_weight_ratio=focus_weight_ratio,
    )
    
    model.load_state_dict(torch.load(best_checkpoint))
    criterion = partial(
        weighted_huber_loss,
        beta=huber_beta,
        z_mean=getattr(train_dataset, "z_mean", 0.0),
        z_std=getattr(train_dataset, "z_std", 1.0),
        inner_band_um=focus_inner_um,
        weight_ratio=focus_weight_ratio,
    )
    test_results = test_model(model, test_loader, device, criterion, run_folder)
