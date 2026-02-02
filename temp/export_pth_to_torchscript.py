#!/usr/bin/env python
"""
Minimal converter: state_dict .pth/.pt -> TorchScript .pt using this repo's model.

Edit the four constants below to point at your files, then run:
  py temp/export_pth_to_torchscript.py
"""

import importlib.util
from pathlib import Path
import torch

# ----- EDIT THESE PATHS / SETTINGS -----
CHECKPOINT = Path(r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilevitv2_050-20260202_143309\best_model_focus_epoch49.pth")
OUTPUT     = Path(r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\training\train-mobilevitv2_050-20260202_143309\PipetteFocuserNet.pt")
MODEL_NAME = "mobilevitv2_050"  # set to match config.txt if different
IMG_SIZE   = 224                # set to training img_size if different
# --------------------------------------


def load_state_dict(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        return ckpt.get("state_dict", ckpt)
    return ckpt


def build_model(repo_root: Path, model_name: str) -> torch.nn.Module:
    """
    Build the regression model using this repo's PYTHON/model.py so any local
    customizations used during training are preserved.
    """
    model_py = (repo_root / "PYTHON" / "model.py").resolve()
    if not model_py.is_file():
        raise FileNotFoundError(f"Expected model.py at {model_py}; run from the pipetteFindingCNN repo or adjust repo_root.")

    spec = importlib.util.spec_from_file_location("pipette_model_module", model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_regression_model = getattr(module, "get_regression_model", None)
    if get_regression_model is None:
        raise ImportError(f"get_regression_model not found in {model_py}")
    return get_regression_model(model_name=model_name, pretrained=False, output_dim=1)


def main():
    checkpoint_path = CHECKPOINT
    output_path = OUTPUT
    model_name = MODEL_NAME
    img_size = int(IMG_SIZE)

    repo_root = Path(__file__).resolve().parent.parent
    state_dict = load_state_dict(checkpoint_path)
    model = build_model(repo_root, model_name)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(output_path)
    print(f"Saved TorchScript model to {output_path}")


if __name__ == "__main__":
    main()
