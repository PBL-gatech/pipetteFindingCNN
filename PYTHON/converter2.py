#!/usr/bin/env python
"""
Minimal converter: state_dict .pth/.pt -> TorchScript .pt using this repo's model.

Can be executed directly or imported:
- CLI: py converter2.py --checkpoint path\best_model.pth --output path\model.pt
- Function: convert_checkpoint_to_torchscript(...)
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Optional

import torch

# ----- DEFAULT SETTINGS -----
DEFAULT_MODEL_NAME = "mobilevitv2_050"  # set to match config if different
DEFAULT_IMG_SIZE = 224                 # set to training img_size if different
DEFAULT_OUTPUT_NAME = "PipetteFocuserNet.pt"
# ----------------------------


def load_state_dict(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        return ckpt.get("state_dict", ckpt)
    return ckpt


def build_model(repo_root: Path, model_name: str) -> torch.nn.Module:
    """
    Build the regression model from PYTHON/model.py so training-time customizations are
    preserved.
    """
    model_py = (repo_root / "PYTHON" / "model.py").resolve()
    if not model_py.is_file():
        raise FileNotFoundError(
            f"Expected model.py at {model_py}; run from the pipetteFindingCNN repo or adjust repo_root."
        )

    spec = importlib.util.spec_from_file_location("pipette_model_module", model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_regression_model = getattr(module, "get_regression_model", None)
    if get_regression_model is None:
        raise ImportError(f"get_regression_model not found in {model_py}")

    return get_regression_model(model_name=model_name, pretrained=False, output_dim=1)


def convert_checkpoint_to_torchscript(
    checkpoint_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    img_size: int = DEFAULT_IMG_SIZE,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Convert a checkpoint at ``checkpoint_path`` to a TorchScript model at
    ``output_path`` and return the saved path.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    repo_root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parent.parent

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = load_state_dict(checkpoint_path)
    model = build_model(repo_root=repo_root, model_name=model_name)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy = torch.zeros(1, 3, int(img_size), int(img_size))
    scripted = torch.jit.trace(model, dummy)
    scripted.save(str(output_path))
    print(f"Saved TorchScript model to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export checkpoint to TorchScript.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent.parent
            / "training"
            / "train-mobilevitv2_050-20260212_185415"
            / "best_model_focus_epoch49.pth"
        ),
        help="Path to input checkpoint",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / DEFAULT_OUTPUT_NAME),
        help="Path for exported TorchScript model",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name used during training")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input image size (square)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint_to_torchscript(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        model_name=args.model_name,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
