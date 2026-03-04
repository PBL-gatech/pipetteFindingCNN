"""
PyQt GUI that combines data preparation and z-balancing.

This tool reuses:
- data_preparer.py for movement parsing and per-session zeroing/alignment
- balancer_z.py for dataset balancing and artifact/statistics output

It also supports a CLI plot mode:
- compute focus scores for images listed in a final CSV
- plot filter scores vs recorded position (Laplacian, LoG, Sobel, Gabor)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

from data_preparer import (
    compute_laplacian_variance,
    crop_tip_roi_256,
    load_focus_points_from_final_csv,
    extract_image_data,
    find_closest_movement_record,
    load_calibration_matrix_xy,
    load_movement_data,
)
from balancer_z import CAP_PER_BIN, CULL_LIMIT, SEED, TARGET_COL, balance_defocus
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
CAMERA_FRAMES_MANIFEST = "camera_frames_manifest.csv"
CROPPED_FRAMES_MANIFEST = "cropped_frames_manifest.csv"
CROPPED_CAMERA_FRAMES_DIR = "cropped_camera_frames"
DEFAULT_CROP_SIZE = 256
PREPROCESS_STATE_FILE = "preprocess_state.json"
CROPPED_OUTLIER_ZSCORE = 3.0
FOCUS_METRIC_ORDER = (
    "laplacian_variance",
    "log_variance",
    "sobel_score",
    "gabor_score",
)
FOCUS_METRIC_SPECS: dict[str, dict[str, str]] = {
    "laplacian_variance": {
        "display_name": "Laplacian",
        "score_label": "Laplacian variance",
        "plot_title": "Laplacian Variance vs Recorded Position",
        "file_stem": "laplacian_variance_vs_position",
    },
    "log_variance": {
        "display_name": "LoG",
        "score_label": "LoG variance",
        "plot_title": "LoG Variance vs Recorded Position",
        "file_stem": "log_variance_vs_position",
    },
    "sobel_score": {
        "display_name": "Sobel",
        "score_label": "Sobel score",
        "plot_title": "Sobel Score vs Recorded Position",
        "file_stem": "sobel_score_vs_position",
    },
    "gabor_score": {
        "display_name": "Gabor",
        "score_label": "Gabor score",
        "plot_title": "Gabor Score vs Recorded Position",
        "file_stem": "gabor_score_vs_position",
    },
}


def find_image_folder(session_dir: str) -> Optional[str]:
    """Return the image folder path for a session, if present."""
    candidate_folders = (
        "camera_frames",
        "P_DET_IMAGES",
        os.path.join("camera_frames", "P_DET_IMAGES"),
    )
    for folder_name in candidate_folders:
        candidate = os.path.join(session_dir, folder_name)
        if os.path.isdir(candidate):
            return candidate
    return None


def is_valid_session_dir(session_dir: str) -> bool:
    """A valid session must contain movement_recording.csv and an image folder."""
    movement_path = os.path.join(session_dir, "movement_recording.csv")
    return os.path.isfile(movement_path) and find_image_folder(session_dir) is not None


def collect_image_paths(image_root: str) -> list[str]:
    """Recursively collect image files under image_root."""
    image_paths: list[str] = []
    for root, _, files in os.walk(image_root):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, filename))
    return image_paths


def sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def unique_output_name(preferred_name: str, session_tag: str, used_names: set[str]) -> str:
    """Return a unique filename for the combined camera_frames folder."""
    if preferred_name not in used_names:
        return preferred_name

    root, ext = os.path.splitext(preferred_name)
    base_candidate = f"{session_tag}__{preferred_name}"
    if base_candidate not in used_names:
        return base_candidate

    suffix = 2
    while True:
        candidate = f"{session_tag}__{root}_{suffix}{ext}"
        if candidate not in used_names:
            return candidate
        suffix += 1


def load_manifest(manifest_path: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    if not os.path.isfile(manifest_path):
        return entries

    with open(manifest_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_path = row.get("source_path")
            output_name = row.get("output_name")
            if not source_path or not output_name:
                continue
            entries[source_path] = {
                "output_name": output_name,
                "size_bytes": row.get("size_bytes", ""),
                "mtime_ns": row.get("mtime_ns", ""),
            }
    return entries


def save_manifest(manifest_path: str, entries: dict[str, dict[str, str]]) -> None:
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_path", "output_name", "size_bytes", "mtime_ns"])
        for source_path in sorted(entries.keys()):
            row = entries[source_path]
            writer.writerow(
                [
                    source_path,
                    row.get("output_name", ""),
                    row.get("size_bytes", ""),
                    row.get("mtime_ns", ""),
                ]
            )


def load_cropped_manifest(manifest_path: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    if not os.path.isfile(manifest_path):
        return entries

    with open(manifest_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_path = row.get("source_path")
            output_name = row.get("output_name")
            if not source_path or not output_name:
                continue
            entries[source_path] = {
                "output_name": output_name,
                "size_bytes": row.get("size_bytes", ""),
                "mtime_ns": row.get("mtime_ns", ""),
                "calibration_signature": row.get("calibration_signature", ""),
                "crop_size": row.get("crop_size", ""),
                "laplacian_variance": row.get("laplacian_variance", ""),
                "outlier_removed": row.get("outlier_removed", "0"),
            }
    return entries


def save_cropped_manifest(manifest_path: str, entries: dict[str, dict[str, str]]) -> None:
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_path",
                "output_name",
                "size_bytes",
                "mtime_ns",
                "calibration_signature",
                "crop_size",
                "laplacian_variance",
                "outlier_removed",
            ]
        )
        for source_path in sorted(entries.keys()):
            row = entries[source_path]
            writer.writerow(
                [
                    source_path,
                    row.get("output_name", ""),
                    row.get("size_bytes", ""),
                    row.get("mtime_ns", ""),
                    row.get("calibration_signature", ""),
                    row.get("crop_size", ""),
                    row.get("laplacian_variance", ""),
                    row.get("outlier_removed", "0"),
                ]
            )


def build_file_signature(file_path: str) -> str:
    normalized = os.path.normpath(os.path.abspath(file_path))
    stat = os.stat(normalized)
    return f"{normalized}|{stat.st_size}|{stat.st_mtime_ns}"


def hash_rows(rows: list[tuple[str, float, float, float, float]]) -> str:
    digest = hashlib.sha256()
    for filename, defocus, pipette_x, pipette_y, pipette_z in sorted(rows, key=lambda row: row[0]):
        digest.update(
            f"{filename}|{defocus:.9f}|{pipette_x:.9f}|{pipette_y:.9f}|{pipette_z:.9f}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def load_preprocess_state(state_path: str) -> Optional[dict[str, object]]:
    if not os.path.isfile(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def save_preprocess_state(state_path: str, state: dict[str, object]) -> None:
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def has_reusable_artifacts(state: dict[str, object], run_signature: str) -> bool:
    if not state:
        return False
    if str(state.get("run_signature", "")) != run_signature:
        return False

    balanced_csv_path = str(state.get("balanced_csv_path", ""))
    if not balanced_csv_path or not os.path.isfile(balanced_csv_path):
        return False

    metric_artifacts = state.get("metric_artifacts")
    if not isinstance(metric_artifacts, dict):
        return False
    for metric_name in FOCUS_METRIC_ORDER:
        metric_entry = metric_artifacts.get(metric_name)
        if not isinstance(metric_entry, dict):
            return False
        plot_path = str(metric_entry.get("plot_path", ""))
        csv_path = str(metric_entry.get("csv_path", ""))
        if not plot_path or not csv_path:
            return False
        if not os.path.isfile(plot_path) or not os.path.isfile(csv_path):
            return False

    return True


def _get_focus_metric_spec(metric_name: str) -> dict[str, str]:
    metric_spec = FOCUS_METRIC_SPECS.get(metric_name)
    if metric_spec is None:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Expected one of: {FOCUS_METRIC_ORDER}"
        )
    return metric_spec


def save_focus_metric_points_csv(
    points: list[dict[str, str | float]],
    output_csv: Path,
    position_col: str,
    metric_name: str,
) -> None:
    _get_focus_metric_spec(metric_name)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "image_path", position_col, metric_name])
        for item in points:
            writer.writerow(
                [
                    item["filename"],
                    item["image_path"],
                    item["position_value"],
                    item[metric_name],
                ]
            )


def save_focus_metric_plot(
    points: list[dict[str, str | float]],
    output_plot: Path,
    position_col: str,
    metric_name: str,
    show_plot: bool,
) -> None:
    import matplotlib.pyplot as plt

    metric_spec = _get_focus_metric_spec(metric_name)
    x_values = [float(item["position_value"]) for item in points]
    y_values = [float(item[metric_name]) for item in points]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_values, y_values, s=12, alpha=0.7)

    best_point = max(points, key=lambda item: float(item[metric_name]))
    ax.scatter(
        [float(best_point["position_value"])],
        [float(best_point[metric_name])],
        marker="*",
        s=220,
        color="red",
        label=(
            f"max {metric_spec['score_label']} @ "
            f"{position_col}={float(best_point['position_value']):.4f}"
        ),
        zorder=5,
    )

    ax.set_title(metric_spec["plot_title"])
    ax.set_xlabel(position_col)
    ax.set_ylabel(metric_spec["score_label"])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    if show_plot:
        plt.show()
    plt.close(fig)


def save_laplacian_points_csv(
    points: list[dict[str, str | float]],
    output_csv: Path,
    position_col: str,
) -> None:
    save_focus_metric_points_csv(
        points=points,
        output_csv=output_csv,
        position_col=position_col,
        metric_name="laplacian_variance",
    )


def save_laplacian_plot(
    points: list[dict[str, str | float]],
    output_plot: Path,
    position_col: str,
    show_plot: bool,
) -> None:
    save_focus_metric_plot(
        points=points,
        output_plot=output_plot,
        position_col=position_col,
        metric_name="laplacian_variance",
        show_plot=show_plot,
    )


def run_laplacian_plot_from_csv(
    csv_path: Path,
    images_dir_arg: Optional[str],
    position_col_arg: Optional[str],
    median_ksize: int,
    max_rows: Optional[int],
    output_plot_arg: Optional[str],
    output_csv_arg: Optional[str],
    show_plot: bool,
) -> dict[str, object]:
    position_col, images_dir, points = load_focus_points_from_final_csv(
        csv_path=csv_path,
        images_dir_arg=images_dir_arg,
        requested_position_column=position_col_arg,
        median_ksize=median_ksize,
        max_rows=max_rows,
    )
    print(f"Using CSV: {csv_path}")
    print(f"Using images directory: {images_dir}")
    print(f"Using position column: {position_col}")

    if not points:
        raise ValueError("No usable rows found. Check CSV/image paths and selected columns.")

    if output_plot_arg:
        output_plot = Path(output_plot_arg).expanduser().resolve()
    else:
        output_plot = csv_path.with_name(f"{csv_path.stem}_laplacian_vs_position.png")
    save_laplacian_plot(
        points=points,
        output_plot=output_plot,
        position_col=position_col,
        show_plot=show_plot,
    )
    print(f"Saved plot: {output_plot}")

    if output_csv_arg:
        output_csv = Path(output_csv_arg).expanduser().resolve()
        save_laplacian_points_csv(points=points, output_csv=output_csv, position_col=position_col)
        print(f"Saved CSV: {output_csv}")

    best_point = max(points, key=lambda item: float(item["laplacian_variance"]))
    print(
        "Best focus candidate: "
        f"file={best_point['filename']}, "
        f"{position_col}={float(best_point['position_value']):.4f}, "
        f"lap_var={float(best_point['laplacian_variance']):.4f}"
    )
    return {
        "position_col": position_col,
        "images_dir": str(images_dir),
        "output_plot": str(output_plot),
        "output_csv": str(output_csv) if output_csv_arg else None,
        "points_count": len(points),
    }


def run_focus_metrics_plots_from_csv(
    csv_path: Path,
    images_dir_arg: Optional[str],
    position_col_arg: Optional[str],
    median_ksize: int,
    max_rows: Optional[int],
    output_dir_arg: Optional[str],
    show_plot: bool,
) -> dict[str, object]:
    position_col, images_dir, points = load_focus_points_from_final_csv(
        csv_path=csv_path,
        images_dir_arg=images_dir_arg,
        requested_position_column=position_col_arg,
        median_ksize=median_ksize,
        max_rows=max_rows,
    )
    print(f"Using CSV: {csv_path}")
    print(f"Using images directory: {images_dir}")
    print(f"Using position column: {position_col}")

    if not points:
        raise ValueError("No usable rows found. Check CSV/image paths and selected columns.")

    if output_dir_arg:
        output_dir = Path(output_dir_arg).expanduser().resolve()
    else:
        output_dir = csv_path.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_outputs: dict[str, dict[str, object]] = {}
    for metric_name in FOCUS_METRIC_ORDER:
        metric_spec = _get_focus_metric_spec(metric_name)
        plot_path = output_dir / f"{metric_spec['file_stem']}.png"
        csv_path_metric = output_dir / f"{metric_spec['file_stem']}.csv"

        save_focus_metric_plot(
            points=points,
            output_plot=plot_path,
            position_col=position_col,
            metric_name=metric_name,
            show_plot=show_plot,
        )
        save_focus_metric_points_csv(
            points=points,
            output_csv=csv_path_metric,
            position_col=position_col,
            metric_name=metric_name,
        )

        best_point = max(points, key=lambda item: float(item[metric_name]))
        print(f"Saved {metric_spec['display_name']} plot: {plot_path}")
        print(f"Saved {metric_spec['display_name']} CSV: {csv_path_metric}")
        print(
            f"Best {metric_spec['display_name']} candidate: "
            f"file={best_point['filename']}, "
            f"{position_col}={float(best_point['position_value']):.4f}, "
            f"score={float(best_point[metric_name]):.4f}"
        )

        metric_outputs[metric_name] = {
            "output_plot": str(plot_path),
            "output_csv": str(csv_path_metric),
            "best_filename": str(best_point["filename"]),
            "best_position": float(best_point["position_value"]),
            "best_score": float(best_point[metric_name]),
            "points_count": len(points),
        }

    return {
        "position_col": position_col,
        "images_dir": str(images_dir),
        "points_count": len(points),
        "metrics": metric_outputs,
    }


def run_cli_if_requested(argv: list[str]) -> Optional[int]:
    if "--plot-laplacian" not in argv and "--plot-focus-metrics" not in argv:
        return None

    parser = argparse.ArgumentParser(
        description=(
            "DataPreprocesser CLI mode. Use --plot-laplacian (single metric) or "
            "--plot-focus-metrics (Laplacian, LoG, Sobel, Gabor)."
        )
    )
    parser.add_argument(
        "--plot-laplacian",
        action="store_true",
        help="Enable Laplacian-vs-position plotting mode.",
    )
    parser.add_argument(
        "--plot-focus-metrics",
        action="store_true",
        help="Generate Laplacian, LoG, Sobel, and Gabor plots/CSVs together.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to final dataset CSV (for example: pipette_z_data.csv).",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory containing images listed by CSV filename. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--position-col",
        default=None,
        help="Position column for X axis. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--median-ksize",
        type=int,
        default=5,
        help="Odd kernel size for median blur before Laplacian (default: 5).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows processed from the CSV.",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional output plot path. Default: <csv_stem>_laplacian_vs_position.png",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path with computed Laplacian variance values.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory for --plot-focus-metrics. "
            "Defaults to the CSV parent folder."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively.",
    )
    args = parser.parse_args(argv)

    if not args.plot_laplacian and not args.plot_focus_metrics:
        return None
    if not args.csv:
        parser.error("--csv is required when --plot-laplacian or --plot-focus-metrics is set.")

    try:
        csv_path = Path(args.csv).expanduser().resolve()
        if args.plot_focus_metrics:
            run_focus_metrics_plots_from_csv(
                csv_path=csv_path,
                images_dir_arg=args.images_dir,
                position_col_arg=args.position_col,
                median_ksize=args.median_ksize,
                max_rows=args.max_rows,
                output_dir_arg=args.output_dir,
                show_plot=args.show,
            )
        if args.plot_laplacian:
            run_laplacian_plot_from_csv(
                csv_path=csv_path,
                images_dir_arg=args.images_dir,
                position_col_arg=args.position_col,
                median_ksize=args.median_ksize,
                max_rows=args.max_rows,
                output_plot_arg=args.output_plot,
                output_csv_arg=args.output_csv,
                show_plot=args.show,
            )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


class PreprocessWorker(QObject):
    log = pyqtSignal(str)
    success = pyqtSignal(str, str)
    failure = pyqtSignal(str)
    done = pyqtSignal()

    def __init__(
        self,
        session_dirs: list[str],
        output_dir: str,
        cull_limit: Optional[float],
        cap_per_bin: int,
        seed: int,
        show_plots: bool,
        enable_cropped_tip_roi: bool,
        calibration_path: Optional[str],
        crop_size: int,
    ):
        super().__init__()
        self.session_dirs = session_dirs
        self.output_dir = output_dir
        self.cull_limit = cull_limit
        self.cap_per_bin = cap_per_bin
        self.seed = seed
        self.show_plots = show_plots
        self.enable_cropped_tip_roi = enable_cropped_tip_roi
        self.calibration_path = calibration_path
        self.crop_size = int(crop_size)

    def run(self) -> None:
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            state_path = os.path.join(self.output_dir, PREPROCESS_STATE_FILE)
            previous_state = load_preprocess_state(state_path)
            combined_camera_frames_dir = os.path.join(self.output_dir, "camera_frames")
            os.makedirs(combined_camera_frames_dir, exist_ok=True)
            combined_cropped_frames_dir: Optional[str] = None
            manip_matrix_xy = None
            calibration_signature = ""
            cropped_manifest: dict[str, dict[str, str]] = {}
            cropped_manifest_path: Optional[str] = None
            cropped_laplacian_by_output_name: dict[str, float] = {}
            source_path_by_output_name: dict[str, str] = {}
            known_outlier_output_names: set[str] = set()
            if self.enable_cropped_tip_roi:
                if not self.calibration_path:
                    raise ValueError("Calibration path is required when cropped ROI is enabled.")
                manip_matrix_xy = load_calibration_matrix_xy(self.calibration_path)
                calibration_signature = build_file_signature(self.calibration_path)
                combined_cropped_frames_dir = os.path.join(
                    self.output_dir, CROPPED_CAMERA_FRAMES_DIR
                )
                os.makedirs(combined_cropped_frames_dir, exist_ok=True)
                cropped_manifest_path = os.path.join(self.output_dir, CROPPED_FRAMES_MANIFEST)
                cropped_manifest = load_cropped_manifest(cropped_manifest_path)
            manifest_path = os.path.join(self.output_dir, CAMERA_FRAMES_MANIFEST)
            manifest = load_manifest(manifest_path)

            all_rows: list[tuple[str, float, float, float, float]] = []
            used_output_filenames: set[str] = {
                name
                for name in os.listdir(combined_camera_frames_dir)
                if os.path.isfile(os.path.join(combined_camera_frames_dir, name))
            }

            for idx, session_dir in enumerate(self.session_dirs, start=1):
                self.log.emit(f"[{idx}/{len(self.session_dirs)}] Processing: {session_dir}")

                movement_path = os.path.join(session_dir, "movement_recording.csv")
                image_dir = find_image_folder(session_dir)
                if image_dir is None:
                    raise FileNotFoundError(
                        "Missing image folder in "
                        f"{session_dir} (expected camera_frames, P_DET_IMAGES, "
                        "or camera_frames/P_DET_IMAGES)."
                    )
                if not os.path.isfile(movement_path):
                    raise FileNotFoundError(f"Missing movement file: {movement_path}")

                movement_data = load_movement_data(movement_path)
                if not movement_data:
                    raise ValueError(f"No movement data loaded from {movement_path}")

                image_paths = collect_image_paths(image_dir)
                if not image_paths:
                    raise ValueError(f"No image files found in {image_dir}")

                image_files_with_timestamp: list[tuple[str, float, str]] = []
                for image_path in image_paths:
                    image_name = os.path.basename(image_path)
                    _, timestamp = extract_image_data(image_name)
                    image_files_with_timestamp.append((image_name, timestamp, image_path))
                image_files_with_timestamp.sort(key=lambda item: item[1])

                session_tag = sanitize_name(os.path.basename(os.path.normpath(session_dir)))
                session_rows: list[tuple[str, float, float, float, float]] = []
                session_copy_count = 0
                session_skip_count = 0
                session_crop_count = 0
                session_crop_skip_count = 0
                session_crop_skip_reasons: dict[str, int] = {}
                for image_name, timestamp, image_path in image_files_with_timestamp:
                    movement_record = find_closest_movement_record(timestamp, movement_data)
                    pipette_x = float(movement_record["pipette"][0])
                    pipette_y = float(movement_record["pipette"][1])
                    pipette_z = float(movement_record["pipette"][2])
                    session_rows.append(
                        (
                            image_name,
                            pipette_z,
                            pipette_x,
                            pipette_y,
                            pipette_z,
                        )
                    )

                    source_path = os.path.normpath(os.path.abspath(image_path))
                    source_stat = os.stat(image_path)
                    source_size = str(source_stat.st_size)
                    source_mtime_ns = str(source_stat.st_mtime_ns)

                    manifest_entry = manifest.get(source_path)
                    reused_existing_copy = False
                    if manifest_entry:
                        candidate_output_name = manifest_entry.get("output_name", "")
                        candidate_output_path = os.path.join(
                            combined_camera_frames_dir, candidate_output_name
                        )
                        if (
                            candidate_output_name
                            and os.path.isfile(candidate_output_path)
                            and manifest_entry.get("size_bytes") == source_size
                            and manifest_entry.get("mtime_ns") == source_mtime_ns
                        ):
                            output_name = candidate_output_name
                            reused_existing_copy = True
                        else:
                            output_name = unique_output_name(
                                preferred_name=image_name,
                                session_tag=session_tag,
                                used_names=used_output_filenames,
                            )
                    else:
                        output_name = unique_output_name(
                            preferred_name=image_name,
                            session_tag=session_tag,
                            used_names=used_output_filenames,
                        )

                    output_image_path = os.path.join(combined_camera_frames_dir, output_name)
                    if not reused_existing_copy:
                        shutil.copy2(image_path, output_image_path)
                        session_copy_count += 1
                    else:
                        session_skip_count += 1

                    if self.enable_cropped_tip_roi and manip_matrix_xy is not None:
                        if combined_cropped_frames_dir is None:
                            raise RuntimeError("Unexpected missing cropped output directory.")
                        source_path_by_output_name[output_name] = source_path
                        cropped_output_path = os.path.join(combined_cropped_frames_dir, output_name)
                        reused_existing_crop = False
                        cropped_entry = cropped_manifest.get(source_path)
                        if (
                            cropped_entry
                            and cropped_entry.get("output_name") == output_name
                            and cropped_entry.get("size_bytes") == source_size
                            and cropped_entry.get("mtime_ns") == source_mtime_ns
                            and cropped_entry.get("calibration_signature") == calibration_signature
                            and cropped_entry.get("crop_size") == str(self.crop_size)
                        ):
                            if cropped_entry.get("outlier_removed") == "1":
                                known_outlier_output_names.add(output_name)
                                reused_existing_crop = True
                                session_crop_skip_count += 1
                                session_crop_skip_reasons["known_outlier"] = (
                                    session_crop_skip_reasons.get("known_outlier", 0) + 1
                                )
                                lap_var_text = cropped_entry.get("laplacian_variance", "")
                                if lap_var_text:
                                    try:
                                        cropped_laplacian_by_output_name[output_name] = float(
                                            lap_var_text
                                        )
                                    except ValueError:
                                        pass
                                if os.path.isfile(cropped_output_path):
                                    os.remove(cropped_output_path)
                            elif os.path.isfile(cropped_output_path):
                                reused_existing_crop = True
                                session_crop_skip_count += 1
                                session_crop_skip_reasons["already_cropped"] = (
                                    session_crop_skip_reasons.get("already_cropped", 0) + 1
                                )
                                lap_var_text = cropped_entry.get("laplacian_variance", "")
                                if lap_var_text:
                                    try:
                                        cropped_laplacian_by_output_name[output_name] = float(
                                            lap_var_text
                                        )
                                    except ValueError:
                                        pass
                                else:
                                    try:
                                        lap_var = compute_laplacian_variance(
                                            cropped_output_path, median_ksize=5
                                        )
                                        cropped_laplacian_by_output_name[output_name] = lap_var
                                        cropped_entry["laplacian_variance"] = f"{lap_var:.12f}"
                                        cropped_manifest[source_path] = cropped_entry
                                    except Exception:
                                        pass

                        if not reused_existing_crop:
                            crop_ok, crop_reason = crop_tip_roi_256(
                                image_path=image_path,
                                output_path=cropped_output_path,
                                manip_matrix_xy=manip_matrix_xy,
                                zeroed_pipette_xyz=(pipette_x, pipette_y, pipette_z),
                                crop_size=self.crop_size,
                            )
                            if crop_ok:
                                session_crop_count += 1
                                lap_var_value = ""
                                try:
                                    lap_var = compute_laplacian_variance(
                                        cropped_output_path, median_ksize=5
                                    )
                                    lap_var_value = f"{lap_var:.12f}"
                                    cropped_laplacian_by_output_name[output_name] = lap_var
                                except Exception:
                                    if os.path.isfile(cropped_output_path):
                                        os.remove(cropped_output_path)
                                    crop_ok = False
                                    crop_reason = "laplacian_failed"
                                cropped_manifest[source_path] = {
                                    "output_name": output_name,
                                    "size_bytes": source_size,
                                    "mtime_ns": source_mtime_ns,
                                    "calibration_signature": calibration_signature,
                                    "crop_size": str(self.crop_size),
                                    "laplacian_variance": lap_var_value,
                                    "outlier_removed": "0",
                                }
                            if not crop_ok:
                                session_crop_skip_count += 1
                                session_crop_skip_reasons[crop_reason] = (
                                    session_crop_skip_reasons.get(crop_reason, 0) + 1
                                )

                    all_rows.append(
                        (
                            output_name,
                            pipette_z,
                            pipette_x,
                            pipette_y,
                            pipette_z,
                        )
                    )
                    used_output_filenames.add(output_name)
                    manifest[source_path] = {
                        "output_name": output_name,
                        "size_bytes": source_size,
                        "mtime_ns": source_mtime_ns,
                    }

                # Keep data_preparer.py behavior: save per-session CSV near source images.
                session_csv_path = os.path.join(image_dir, "pipette_z_data.csv")
                with open(session_csv_path, "w", newline="") as session_csv:
                    writer = csv.writer(session_csv)
                    writer.writerow(
                        [
                            "filename",
                            "defocus_microns",
                            "pipette_x_microns",
                            "pipette_y_microns",
                            "pipette_z_microns",
                        ]
                    )
                    writer.writerows(session_rows)

                self.log.emit(
                    f"Saved {len(session_rows)} rows to session CSV: {session_csv_path}"
                )
                self.log.emit(
                    f"Copied {session_copy_count} image(s) to: {combined_camera_frames_dir}"
                )
                self.log.emit(
                    f"Skipped {session_skip_count} already-copied image(s)."
                )
                if self.enable_cropped_tip_roi:
                    self.log.emit(
                        "Cropped ROI images written: "
                        f"{session_crop_count}, skipped: {session_crop_skip_count}."
                    )
                    if session_crop_skip_reasons:
                        self.log.emit(f"Crop skip reasons: {session_crop_skip_reasons}")

            if not all_rows:
                raise ValueError("No rows were generated from selected sessions.")

            outlier_output_names: set[str] = set(known_outlier_output_names)
            if self.enable_cropped_tip_roi:
                laplacian_values = list(cropped_laplacian_by_output_name.values())
                if len(laplacian_values) >= 2:
                    mean_lap = sum(laplacian_values) / len(laplacian_values)
                    variance_lap = sum(
                        (value - mean_lap) ** 2 for value in laplacian_values
                    ) / len(laplacian_values)
                    std_lap = variance_lap ** 0.5
                    if std_lap > 0.0:
                        lower = mean_lap - CROPPED_OUTLIER_ZSCORE * std_lap
                        upper = mean_lap + CROPPED_OUTLIER_ZSCORE * std_lap
                        for output_name, lap_var in cropped_laplacian_by_output_name.items():
                            if lap_var < lower or lap_var > upper:
                                outlier_output_names.add(output_name)
                        self.log.emit(
                            "Cropped Laplacian outlier filter "
                            f"(mean={mean_lap:.4f}, std={std_lap:.4f}, "
                            f"threshold={CROPPED_OUTLIER_ZSCORE}x SD) marked "
                            f"{len(outlier_output_names)} image(s)."
                        )

                if combined_cropped_frames_dir and outlier_output_names:
                    for output_name in outlier_output_names:
                        cropped_file_path = os.path.join(combined_cropped_frames_dir, output_name)
                        if os.path.isfile(cropped_file_path):
                            os.remove(cropped_file_path)
                        source_path = source_path_by_output_name.get(output_name)
                        if source_path:
                            entry = cropped_manifest.get(source_path, {})
                            entry["outlier_removed"] = "1"
                            cropped_manifest[source_path] = entry

                if outlier_output_names:
                    rows_before_outlier_filter = len(all_rows)
                    all_rows = [
                        row for row in all_rows if row[0] not in outlier_output_names
                    ]
                    removed_count = rows_before_outlier_filter - len(all_rows)
                    self.log.emit(
                        f"Removed {removed_count} row(s) from combined CSV due to cropped outliers."
                    )
                    if removed_count == 0 and known_outlier_output_names:
                        self.log.emit(
                            "No combined rows matched cached outlier markers in this run."
                        )

            if not all_rows:
                raise ValueError("No rows left after applying cropped outlier filtering.")

            save_manifest(manifest_path, manifest)
            if self.enable_cropped_tip_roi and cropped_manifest_path:
                save_cropped_manifest(cropped_manifest_path, cropped_manifest)

            combined_csv_path = os.path.join(self.output_dir, "pipette_z_data.csv")
            with open(combined_csv_path, "w", newline="") as combined_csv:
                writer = csv.writer(combined_csv)
                writer.writerow(
                    [
                        "filename",
                        "defocus_microns",
                        "pipette_x_microns",
                        "pipette_y_microns",
                        "pipette_z_microns",
                    ]
                )
                writer.writerows(all_rows)

            self.log.emit(
                f"Saved combined CSV with {len(all_rows)} rows: {combined_csv_path}"
            )
            self.log.emit(
                f"Combined camera_frames folder: {combined_camera_frames_dir}"
            )
            if self.enable_cropped_tip_roi and combined_cropped_frames_dir:
                self.log.emit(
                    f"Combined {CROPPED_CAMERA_FRAMES_DIR} folder: {combined_cropped_frames_dir}"
                )
            focus_images_dir = combined_camera_frames_dir
            if self.enable_cropped_tip_roi:
                if not combined_cropped_frames_dir or not os.path.isdir(combined_cropped_frames_dir):
                    raise FileNotFoundError(
                        f"Missing {CROPPED_CAMERA_FRAMES_DIR} folder: {combined_cropped_frames_dir}"
                    )
                focus_images_dir = combined_cropped_frames_dir
            self.log.emit(
                "Using cropped ROI images for focus-filter artifacts: "
                f"{focus_images_dir}"
                if self.enable_cropped_tip_roi
                else "Using full camera_frames images for focus-filter artifacts: "
                f"{focus_images_dir}"
            )

            rows_hash = hash_rows(all_rows)
            signature_payload = {
                "rows_hash": rows_hash,
                "cull_limit": self.cull_limit,
                "cap_per_bin": int(self.cap_per_bin),
                "seed": int(self.seed),
                "use_cropped_images": bool(self.enable_cropped_tip_roi),
                "calibration_signature": calibration_signature if self.enable_cropped_tip_roi else "",
                "crop_size": int(self.crop_size) if self.enable_cropped_tip_roi else 0,
                "cropped_outlier_zscore": CROPPED_OUTLIER_ZSCORE if self.enable_cropped_tip_roi else 0.0,
                "focus_metric_names": list(FOCUS_METRIC_ORDER),
            }
            run_signature = hashlib.sha256(
                json.dumps(signature_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()

            if previous_state and has_reusable_artifacts(previous_state, run_signature):
                balanced_csv_path = str(previous_state["balanced_csv_path"])
                artifacts_dir = str(previous_state.get("artifacts_dir", os.path.dirname(balanced_csv_path)))
                metric_artifacts = previous_state.get("metric_artifacts", {})
                self.log.emit("Skipping balancing + focus-filter artifacts (cache hit).")
                self.log.emit(f"Balanced CSV (cached): {balanced_csv_path}")
                for metric_name in FOCUS_METRIC_ORDER:
                    metric_spec = _get_focus_metric_spec(metric_name)
                    metric_entry = (
                        metric_artifacts.get(metric_name, {})
                        if isinstance(metric_artifacts, dict)
                        else {}
                    )
                    self.log.emit(
                        f"{metric_spec['display_name']} plot (cached): "
                        f"{metric_entry.get('plot_path', '')}"
                    )
                    self.log.emit(
                        f"{metric_spec['display_name']} CSV (cached): "
                        f"{metric_entry.get('csv_path', '')}"
                    )
            else:
                self.log.emit("Balancing combined dataset...")
                # balancer_z uses matplotlib; force a non-GUI backend in worker threads.
                import matplotlib.pyplot as plt

                plt.switch_backend("Agg")

                balanced_csv_path = balance_defocus(
                    input_path=combined_csv_path,
                    target_col=TARGET_COL,
                    cull_limit=self.cull_limit,
                    cap_per_bin=self.cap_per_bin,
                    seed=self.seed,
                    show_plots=False,
                )

                artifacts_dir = os.path.dirname(balanced_csv_path)
                self.log.emit("Generating focus-filter artifacts from balanced CSV...")
                focus_metrics_info = run_focus_metrics_plots_from_csv(
                    csv_path=Path(balanced_csv_path),
                    images_dir_arg=focus_images_dir,
                    position_col_arg=None,
                    median_ksize=5,
                    max_rows=None,
                    output_dir_arg=artifacts_dir,
                    show_plot=False,
                )
                metric_outputs = focus_metrics_info.get("metrics", {})
                if not isinstance(metric_outputs, dict):
                    raise ValueError("Unexpected focus metrics output format.")
                metric_artifacts_state: dict[str, dict[str, str]] = {}
                for metric_name in FOCUS_METRIC_ORDER:
                    metric_spec = _get_focus_metric_spec(metric_name)
                    metric_info = metric_outputs.get(metric_name, {})
                    if not isinstance(metric_info, dict):
                        raise ValueError(f"Missing output info for metric: {metric_name}")
                    metric_plot_path = str(metric_info.get("output_plot", ""))
                    metric_csv_path = str(metric_info.get("output_csv", ""))
                    metric_points_count = int(metric_info.get("points_count", 0))
                    self.log.emit(
                        f"{metric_spec['display_name']} artifact ready: "
                        f"{metric_plot_path} ({metric_points_count} points, "
                        f"x={focus_metrics_info['position_col']})"
                    )
                    self.log.emit(
                        f"{metric_spec['display_name']} values CSV: {metric_csv_path}"
                    )
                    metric_artifacts_state[metric_name] = {
                        "plot_path": metric_plot_path,
                        "csv_path": metric_csv_path,
                    }

                laplacian_artifact = metric_artifacts_state.get("laplacian_variance", {})
                self.log.emit(f"Balancing complete. Artifacts: {artifacts_dir}")
                save_preprocess_state(
                    state_path,
                    {
                        "run_signature": run_signature,
                        "rows_hash": rows_hash,
                        "balanced_csv_path": balanced_csv_path,
                        "artifacts_dir": artifacts_dir,
                        "laplacian_plot_path": laplacian_artifact.get("plot_path", ""),
                        "laplacian_csv_path": laplacian_artifact.get("csv_path", ""),
                        "laplacian_images_dir": focus_images_dir,
                        "metric_artifacts": metric_artifacts_state,
                    },
                )
            self.success.emit(combined_csv_path, balanced_csv_path)
        except Exception:
            self.failure.emit(traceback.format_exc())
        finally:
            self.done.emit()


class DataPreprocesserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread: Optional[QThread] = None
        self.worker: Optional[PreprocessWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("Data Preprocesser")
        self.resize(980, 700)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        folders_label = QLabel("Selected date_time session folders:")
        root_layout.addWidget(folders_label)

        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        root_layout.addWidget(self.session_list)

        folder_buttons_layout = QHBoxLayout()
        self.add_session_button = QPushButton("Add Session Folder(s)")
        self.add_parent_button = QPushButton("Add From Parent Folder")
        self.remove_selected_button = QPushButton("Remove Selected")
        self.clear_button = QPushButton("Clear")

        folder_buttons_layout.addWidget(self.add_session_button)
        folder_buttons_layout.addWidget(self.add_parent_button)
        folder_buttons_layout.addWidget(self.remove_selected_button)
        folder_buttons_layout.addWidget(self.clear_button)
        root_layout.addLayout(folder_buttons_layout)

        options_layout = QGridLayout()
        root_layout.addLayout(options_layout)

        output_label = QLabel("Combined output folder:")
        self.output_dir_edit = QLineEdit()
        self.output_browse_button = QPushButton("Browse")

        options_layout.addWidget(output_label, 0, 0)
        options_layout.addWidget(self.output_dir_edit, 0, 1)
        options_layout.addWidget(self.output_browse_button, 0, 2)

        self.enable_cull_checkbox = QCheckBox("Enable cull limit (+/- microns)")
        self.enable_cull_checkbox.setChecked(CULL_LIMIT is not None)
        self.cull_spin = QDoubleSpinBox()
        self.cull_spin.setDecimals(3)
        self.cull_spin.setRange(0.0, 1000000.0)
        self.cull_spin.setValue(float(CULL_LIMIT or 0.0))
        self.cull_spin.setEnabled(self.enable_cull_checkbox.isChecked())

        options_layout.addWidget(self.enable_cull_checkbox, 1, 0)
        options_layout.addWidget(self.cull_spin, 1, 1)

        self.enable_cropped_roi_checkbox = QCheckBox(
            f"Enable calibrated tip ROI crops ({DEFAULT_CROP_SIZE}x{DEFAULT_CROP_SIZE})"
        )
        self.enable_cropped_roi_checkbox.setChecked(False)
        options_layout.addWidget(self.enable_cropped_roi_checkbox, 2, 0, 1, 2)

        calibration_label = QLabel("Calibration JSON:")
        self.calibration_path_edit = QLineEdit()
        self.calibration_browse_button = QPushButton("Browse")
        self.calibration_path_edit.setPlaceholderText(
            "Path to calibration.json containing manip/pipette matrix M"
        )
        self.calibration_path_edit.setEnabled(False)
        self.calibration_browse_button.setEnabled(False)
        options_layout.addWidget(calibration_label, 3, 0)
        options_layout.addWidget(self.calibration_path_edit, 3, 1)
        options_layout.addWidget(self.calibration_browse_button, 3, 2)

        cap_label = QLabel("Cap per bin:")
        self.cap_spin = QSpinBox()
        self.cap_spin.setRange(1, 1000000)
        self.cap_spin.setValue(int(CAP_PER_BIN))
        options_layout.addWidget(cap_label, 4, 0)
        options_layout.addWidget(self.cap_spin, 4, 1)

        seed_label = QLabel("Random seed:")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(int(SEED))
        options_layout.addWidget(seed_label, 5, 0)
        options_layout.addWidget(self.seed_spin, 5, 1)

        self.show_plots_checkbox = QCheckBox(
            "Interactive plot windows (disabled during background run)"
        )
        self.show_plots_checkbox.setChecked(False)
        self.show_plots_checkbox.setEnabled(False)
        options_layout.addWidget(self.show_plots_checkbox, 6, 0)

        self.run_button = QPushButton("Run Preprocess + Balance")
        root_layout.addWidget(self.run_button)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        root_layout.addWidget(self.log_text)

        self.add_session_button.clicked.connect(self.add_multiple_sessions)
        self.add_parent_button.clicked.connect(self.add_from_parent_folder)
        self.remove_selected_button.clicked.connect(self.remove_selected_sessions)
        self.clear_button.clicked.connect(self.session_list.clear)
        self.output_browse_button.clicked.connect(self.browse_output_dir)
        self.enable_cull_checkbox.toggled.connect(self.cull_spin.setEnabled)
        self.enable_cropped_roi_checkbox.toggled.connect(self._set_crop_controls_enabled)
        self.calibration_browse_button.clicked.connect(self.browse_calibration_file)
        self.run_button.clicked.connect(self.run_pipeline)

    def append_log(self, message: str) -> None:
        self.log_text.append(message)

    def add_multiple_sessions(self) -> None:
        dialog = QFileDialog(self, "Select one or more date_time session folders")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        # Native Windows folder picker does not support multi-select directories.
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)

        for view in dialog.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if not dialog.exec_():
            return

        selected_dirs = dialog.selectedFiles()
        if not selected_dirs:
            return

        added_count = 0
        invalid_dirs = []
        for selected_dir in selected_dirs:
            if not is_valid_session_dir(selected_dir):
                invalid_dirs.append(selected_dir)
                continue
            if self._add_session_if_missing(selected_dir):
                added_count += 1

        if added_count > 0:
            self.append_log(f"Added {added_count} session folder(s).")

        if invalid_dirs:
            invalid_preview = "\n".join(invalid_dirs[:8])
            suffix = "\n..." if len(invalid_dirs) > 8 else ""
            QMessageBox.warning(
                self,
                "Some Folders Skipped",
                "These folders are invalid (need movement_recording.csv and image folder):\n\n"
                f"{invalid_preview}{suffix}",
            )

    def add_from_parent_folder(self) -> None:
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Select parent folder containing date_time sessions"
        )
        if not parent_dir:
            return

        added_count = 0
        for child_name in sorted(os.listdir(parent_dir)):
            child_path = os.path.join(parent_dir, child_name)
            if os.path.isdir(child_path) and is_valid_session_dir(child_path):
                if self._add_session_if_missing(child_path):
                    added_count += 1

        if added_count == 0:
            QMessageBox.information(
                self,
                "No Sessions Added",
                "No valid session folders were found (or all were already listed).",
            )
        else:
            self.append_log(f"Added {added_count} session folder(s) from: {parent_dir}")

    def _add_session_if_missing(self, session_dir: str) -> bool:
        normalized = os.path.normpath(os.path.abspath(session_dir))
        existing = {
            os.path.normpath(os.path.abspath(self.session_list.item(i).text()))
            for i in range(self.session_list.count())
        }
        if normalized in existing:
            return False

        self.session_list.addItem(normalized)
        self._set_default_output_dir_if_empty()
        return True

    def _set_default_output_dir_if_empty(self) -> None:
        if self.output_dir_edit.text().strip():
            return
        if self.session_list.count() == 0:
            return
        first_session = self.session_list.item(0).text()
        parent = os.path.dirname(first_session)
        default_output = os.path.join(parent, "combined")
        self.output_dir_edit.setText(default_output)

    def _set_crop_controls_enabled(self, enabled: bool) -> None:
        self.calibration_path_edit.setEnabled(enabled)
        self.calibration_browse_button.setEnabled(enabled)

    def browse_output_dir(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select output folder for combined CSV and balancing artifacts"
        )
        if selected_dir:
            self.output_dir_edit.setText(os.path.normpath(selected_dir))

    def browse_calibration_file(self) -> None:
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select calibration JSON file",
            self.calibration_path_edit.text().strip(),
            "JSON Files (*.json);;All Files (*)",
        )
        if selected_path:
            self.calibration_path_edit.setText(os.path.normpath(selected_path))

    def remove_selected_sessions(self) -> None:
        selected_items = self.session_list.selectedItems()
        for item in selected_items:
            row = self.session_list.row(item)
            self.session_list.takeItem(row)

    def _session_dirs(self) -> list[str]:
        return [self.session_list.item(i).text() for i in range(self.session_list.count())]

    def _set_controls_enabled(self, enabled: bool) -> None:
        controls = [
            self.add_session_button,
            self.add_parent_button,
            self.remove_selected_button,
            self.clear_button,
            self.output_dir_edit,
            self.output_browse_button,
            self.enable_cull_checkbox,
            self.cull_spin,
            self.enable_cropped_roi_checkbox,
            self.calibration_path_edit,
            self.calibration_browse_button,
            self.cap_spin,
            self.seed_spin,
            self.show_plots_checkbox,
            self.run_button,
            self.session_list,
        ]
        for control in controls:
            control.setEnabled(enabled)
        if enabled:
            self.cull_spin.setEnabled(self.enable_cull_checkbox.isChecked())
            self._set_crop_controls_enabled(self.enable_cropped_roi_checkbox.isChecked())

    def run_pipeline(self) -> None:
        session_dirs = self._session_dirs()
        if not session_dirs:
            QMessageBox.warning(self, "No Sessions", "Add at least one session folder.")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            self._set_default_output_dir_if_empty()
            output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "No Output Folder", "Select an output folder.")
            return

        cull_limit = self.cull_spin.value() if self.enable_cull_checkbox.isChecked() else None
        cap_per_bin = self.cap_spin.value()
        seed = self.seed_spin.value()
        show_plots = False
        enable_cropped_tip_roi = self.enable_cropped_roi_checkbox.isChecked()
        calibration_path = self.calibration_path_edit.text().strip() or None

        if enable_cropped_tip_roi:
            if not calibration_path:
                QMessageBox.warning(
                    self,
                    "Missing Calibration File",
                    "Select a calibration JSON file when cropped ROI is enabled.",
                )
                return
            if not os.path.isfile(calibration_path):
                QMessageBox.warning(
                    self,
                    "Invalid Calibration File",
                    f"Calibration file not found:\n{calibration_path}",
                )
                return

        self.log_text.clear()
        self.append_log("Starting preprocessing + balancing...")
        self.append_log(f"Sessions selected: {len(session_dirs)}")
        self.append_log(f"Combined output folder: {output_dir}")
        if enable_cropped_tip_roi:
            self.append_log(
                f"Calibrated ROI crop enabled ({DEFAULT_CROP_SIZE}x{DEFAULT_CROP_SIZE})."
            )
            self.append_log(f"Calibration JSON: {calibration_path}")

        self.thread = QThread(self)
        self.worker = PreprocessWorker(
            session_dirs=session_dirs,
            output_dir=output_dir,
            cull_limit=cull_limit,
            cap_per_bin=cap_per_bin,
            seed=seed,
            show_plots=show_plots,
            enable_cropped_tip_roi=enable_cropped_tip_roi,
            calibration_path=calibration_path,
            crop_size=DEFAULT_CROP_SIZE,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.success.connect(self.on_success)
        self.worker.failure.connect(self.on_failure)
        self.worker.done.connect(self.thread.quit)
        self.worker.done.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)

        self._set_controls_enabled(False)
        self.thread.start()

    def on_success(self, combined_csv: str, balanced_csv: str) -> None:
        artifacts_dir = os.path.dirname(balanced_csv)
        combined_camera_frames_dir = os.path.join(os.path.dirname(combined_csv), "camera_frames")
        combined_cropped_frames_dir = os.path.join(
            os.path.dirname(combined_csv), CROPPED_CAMERA_FRAMES_DIR
        )
        self.append_log(f"Combined CSV: {combined_csv}")
        self.append_log(f"Combined images: {combined_camera_frames_dir}")
        if self.enable_cropped_roi_checkbox.isChecked():
            self.append_log(f"Combined cropped images: {combined_cropped_frames_dir}")
        self.append_log(f"Balanced CSV: {balanced_csv}")
        self.append_log(f"Artifacts folder: {artifacts_dir}")
        cropped_section = ""
        if self.enable_cropped_roi_checkbox.isChecked():
            cropped_section = f"Combined {CROPPED_CAMERA_FRAMES_DIR}:\n{combined_cropped_frames_dir}\n\n"
        QMessageBox.information(
            self,
            "Completed",
            "Preprocessing and balancing finished.\n\n"
            f"Combined CSV:\n{combined_csv}\n\n"
            f"Combined camera_frames:\n{combined_camera_frames_dir}\n\n"
            f"{cropped_section}"
            f"Balanced CSV:\n{balanced_csv}\n\n"
            f"Artifacts folder:\n{artifacts_dir}",
        )

    def on_failure(self, error_trace: str) -> None:
        self.append_log("ERROR")
        self.append_log(error_trace)
        QMessageBox.critical(
            self,
            "Failed",
            "Processing failed. See log output for details.",
        )

    def on_thread_finished(self) -> None:
        self._set_controls_enabled(True)
        self.thread = None
        self.worker = None


def run_app() -> int:
    app = QApplication(sys.argv)
    window = DataPreprocesserWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    cli_exit = run_cli_if_requested(sys.argv[1:])
    if cli_exit is None:
        sys.exit(run_app())
    sys.exit(cli_exit)
