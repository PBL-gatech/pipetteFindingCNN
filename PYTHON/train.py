#!/usr/bin/env python3
import os
import json
import datetime
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from model import build_model
from data import PipetteDataModule


def create_run_folder(model_name, config=None):
    training_dir = os.path.join(os.getcwd(), "training")
    os.makedirs(training_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(training_dir, f"train-{model_name}-{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    if config is not None:
        with open(os.path.join(run_folder, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    return run_folder


def _stack(tensors):
    return torch.cat(tensors) if len(tensors) > 0 else torch.tensor([])


def _compute_metrics(pred_x, pred_y, pred_z, gt_x, gt_y, gt_z):
    mae_x = torch.mean(torch.abs(pred_x - gt_x)).item()
    mae_y = torch.mean(torch.abs(pred_y - gt_y)).item()
    mae_z = torch.mean(torch.abs(pred_z - gt_z)).item()

    def _r2(pred, target):
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - target.mean()) ** 2) + 1e-8
        return (1 - ss_res / ss_tot).item()

    return {
        "MAE_x_px": mae_x,
        "MAE_y_px": mae_y,
        "MAE_z_um": mae_z,
        "R2_x": _r2(pred_x, gt_x),
        "R2_y": _r2(pred_y, gt_y),
        "R2_z": _r2(pred_z, gt_z),
    }


def _write_history(run_folder, history):
    path = os.path.join(run_folder, "metrics_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def train_and_validate(
    model,
    train_loader,
    val_loader,
    device,
    run_folder,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=2e-5,
    mixed_precision=True,
    update_callback=None,
):
    optimizer = create_optimizer_v2(model, "adamw", lr=learning_rate, weight_decay=weight_decay)
    scheduler, _ = create_scheduler_v2(
        optimizer, "cosine", num_epochs=num_epochs, updates_per_epoch=len(train_loader)
    )

    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    best_val_loss = float("inf")
    best_checkpoint = None
    global_step = 0

    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "val_loss": [],
        "train_MAE_x_px": [],
        "train_MAE_y_px": [],
        "train_MAE_z_um": [],
        "val_MAE_x_px": [],
        "val_MAE_y_px": [],
        "val_MAE_z_um": [],
        "train_R2_x": [],
        "train_R2_y": [],
        "train_R2_z": [],
        "val_R2_x": [],
        "val_R2_y": [],
        "val_R2_z": [],
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        preds_x, preds_y, preds_z, gts_x, gts_y, gts_z = [], [], [], [], [], []

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_dict = {"xy": targets[:, :2], "z": targets[:, 2]}

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                output = model(images, targets=target_dict, compute_loss=True)
                loss = output["loss"]

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            preds_x.append(output["x_px"].detach().cpu())
            preds_y.append(output["y_px"].detach().cpu())
            preds_z.append(output["defocus_microns"].detach().cpu())
            gts_x.append(target_dict["xy"][:, 0].detach().cpu())
            gts_y.append(target_dict["xy"][:, 1].detach().cpu())
            gts_z.append(target_dict["z"].detach().cpu())

            global_step += 1
            if scheduler is not None and hasattr(scheduler, "step_update"):
                scheduler.step_update(global_step)

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_metrics = _compute_metrics(
            _stack(preds_x), _stack(preds_y), _stack(preds_z), _stack(gts_x), _stack(gts_y), _stack(gts_z)
        )

        model.eval()
        val_loss_sum = 0.0
        v_preds_x, v_preds_y, v_preds_z, v_gts_x, v_gts_y, v_gts_z = [], [], [], [], [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_dict = {"xy": targets[:, :2], "z": targets[:, 2]}
                with autocast_ctx():
                    output = model(images, targets=target_dict, compute_loss=True)
                    loss = output["loss"]
                val_loss_sum += loss.item() * images.size(0)
                v_preds_x.append(output["x_px"].detach().cpu())
                v_preds_y.append(output["y_px"].detach().cpu())
                v_preds_z.append(output["defocus_microns"].detach().cpu())
                v_gts_x.append(target_dict["xy"][:, 0].detach().cpu())
                v_gts_y.append(target_dict["xy"][:, 1].detach().cpu())
                v_gts_z.append(target_dict["z"].detach().cpu())

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_metrics = _compute_metrics(
            _stack(v_preds_x), _stack(v_preds_y), _stack(v_preds_z),
            _stack(v_gts_x), _stack(v_gts_y), _stack(v_gts_z)
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["lr"].append(current_lr)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_MAE_x_px"].append(train_metrics["MAE_x_px"])
        history["train_MAE_y_px"].append(train_metrics["MAE_y_px"])
        history["train_MAE_z_um"].append(train_metrics["MAE_z_um"])
        history["val_MAE_x_px"].append(val_metrics["MAE_x_px"])
        history["val_MAE_y_px"].append(val_metrics["MAE_y_px"])
        history["val_MAE_z_um"].append(val_metrics["MAE_z_um"])
        history["train_R2_x"].append(train_metrics["R2_x"])
        history["train_R2_y"].append(train_metrics["R2_y"])
        history["train_R2_z"].append(train_metrics["R2_z"])
        history["val_R2_x"].append(val_metrics["R2_x"])
        history["val_R2_y"].append(val_metrics["R2_y"])
        history["val_R2_z"].append(val_metrics["R2_z"])

        _write_history(run_folder, history)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"MAE(px) x {val_metrics['MAE_x_px']:.2f} y {val_metrics['MAE_y_px']:.2f} | "
            f"MAE_z {val_metrics['MAE_z_um']:.3f} µm | "
            f"R2 x {val_metrics['R2_x']:.3f} y {val_metrics['R2_y']:.3f} z {val_metrics['R2_z']:.3f} | "
            f"LR {current_lr:.2e}"
        )

        if update_callback is not None:
            update_payload = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "MAE_x_px": val_metrics["MAE_x_px"],
                "MAE_y_px": val_metrics["MAE_y_px"],
                "MAE_z_um": val_metrics["MAE_z_um"],
                "R2_x": val_metrics["R2_x"],
                "R2_y": val_metrics["R2_y"],
                "R2_z": val_metrics["R2_z"],
            }
            update_callback(update_payload)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = os.path.join(run_folder, f"best_model_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), best_checkpoint)

    return best_checkpoint, history


def test_model(model, test_loader, device, run_folder=None):
    model.eval()
    loss_sum = 0.0
    preds_x, preds_y, preds_z, gts_x, gts_y, gts_z = [], [], [], [], [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_dict = {"xy": targets[:, :2], "z": targets[:, 2]}
            output = model(images, targets=target_dict, compute_loss=True)
            loss = output["loss"]
            loss_sum += loss.item() * images.size(0)
            preds_x.append(output["x_px"].detach().cpu())
            preds_y.append(output["y_px"].detach().cpu())
            preds_z.append(output["defocus_microns"].detach().cpu())
            gts_x.append(target_dict["xy"][:, 0].detach().cpu())
            gts_y.append(target_dict["xy"][:, 1].detach().cpu())
            gts_z.append(target_dict["z"].detach().cpu())

    test_loss = loss_sum / len(test_loader.dataset)
    metrics = _compute_metrics(
        _stack(preds_x), _stack(preds_y), _stack(preds_z),
        _stack(gts_x), _stack(gts_y), _stack(gts_z)
    )

    results = {"Test Loss": test_loss, **metrics}
    if run_folder:
        path = os.path.join(run_folder, "test_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test results saved to {path}")
    return results


if __name__ == "__main__":
    image_dir = "path/to/images"
    annotation_file = "path/to/annotations.csv"

    data_module = PipetteDataModule(
        image_dir,
        annotation_file,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        seed=42,
        img_size=224,
    )
    train_dataset, val_dataset, test_dataset = data_module.setup()

    model = build_model(
        model_name="mobilenetv3_large_100",
        pretrained=True,
        heatmap_sigma=2.0,
        heatmap_stride=4,
        lambda_z=1.0,
        huber_beta=1.0,
    )
    model.set_z_stats(*data_module.get_z_stats())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    config = {
        "model_name": "mobilenetv3_large_100",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 2e-5,
        "num_epochs": 50,
        "heatmap_sigma": 2.0,
        "heatmap_stride": 4,
        "lambda_z": 1.0,
        "huber_beta": 1.0,
        "device": str(device),
    }
    run_folder = create_run_folder(config["model_name"], config=config)

    best_checkpoint, history = train_and_validate(
        model,
        train_loader,
        val_loader,
        device,
        run_folder,
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        mixed_precision=True,
    )

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_model(model, test_loader, device, run_folder=run_folder)
