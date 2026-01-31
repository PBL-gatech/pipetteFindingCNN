#!/usr/bin/env python
"""
Evaluate a trained heatmap+z model on the test split.
"""
import json
import os
import glob

import torch
from torch.utils.data import DataLoader

from data import PipetteDataModule
from model import build_model
from train import test_model


def _pick_checkpoint(run_folder, checkpoint_name=None):
    if checkpoint_name:
        candidate = os.path.join(run_folder, checkpoint_name)
        if os.path.isfile(candidate):
            return candidate
    pattern = os.path.join(run_folder, "best_model_epoch*.pth")
    matches = sorted([p for p in glob.glob(pattern)], key=os.path.getmtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found in {run_folder}")
    return matches[0]


def _load_config(run_folder):
    cfg_path = os.path.join(run_folder, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}


def evaluate_trained_model(
    run_folder,
    images_dir,
    annotations_csv,
    device="cuda",
    batch_size=32,
    num_workers=4,
    checkpoint_name=None,
):
    config = _load_config(run_folder)
    model_name = config.get("model_name", "mobilenetv3_large_100")
    heatmap_sigma = config.get("heatmap_sigma", 2.0)
    heatmap_stride = config.get("heatmap_stride", 4)
    lambda_z = config.get("lambda_z", 1.0)
    huber_beta = config.get("huber_beta", 1.0)
    img_size = config.get("img_size", 224)

    model = build_model(
        model_name=model_name,
        pretrained=False,
        heatmap_sigma=heatmap_sigma,
        heatmap_stride=heatmap_stride,
        lambda_z=lambda_z,
        huber_beta=huber_beta,
    )

    ckpt_path = _pick_checkpoint(run_folder, checkpoint_name)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    data_module = PipetteDataModule(
        images_dir,
        annotations_csv,
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,
        seed=42,
        img_size=img_size,
        flip_p=0.0,
        rotate90_p=0.0,
        z_mean=model.z_mean.item(),
        z_std=model.z_std.item(),
    )
    _, _, test_dataset = data_module.setup()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device_torch = torch.device(device)

    results = test_model(model, test_loader, device_torch, run_folder=run_folder)
    return results


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_name", default=None)
    args = parser.parse_args()

    evaluate_trained_model(
        run_folder=args.run_folder,
        images_dir=args.images_dir,
        annotations_csv=args.annotations_csv,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_name=args.checkpoint_name,
    )
