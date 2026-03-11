"""
Balance a defocus CSV (filename, defocus_microns, optional pipette x/y/z).

Edit the CONFIG section below to point at the CSV you want to balance.
The script trims extreme defocus values, caps each bin to a fixed sample
count, and saves a balanced CSV alongside the input.
"""

import os
import time
import threading
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ CONFIG ------------
FILE_PATH = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\combined\pipette_z_data.csv"
TARGET_COL = "defocus_microns"  # falls back to last column if missing
CULL_LIMIT = 30              # set to None to disable symmetric culling
# Variable binning: fine near focus, coarse in the tails
INNER_HALF_WIDTH = 3.0          # microns, range around 0 with fine bins
INNER_STEP = 0.3                # microns, bin width inside [-INNER_HALF_WIDTH, INNER_HALF_WIDTH]
OUTER_STEP = 1.0                # microns, bin width outside the inner band
CAP_PER_BIN = 50
MIN_LABEL_SPACING_UM = 0.25
SEED = 42
SHOW_PLOTS = True
# ------------ CONFIG ------------


def save_histogram(series, bins, title, filepath, xlabel="Value", show_plot=True):
    """Save a histogram for the provided series and optionally display it."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(series, bins=bins, edgecolor="k", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath)
    print(f"Saved histogram: {filepath}")
    if show_plot:
        plt.show()
    plt.close(fig)


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present in df, else None."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def resolve_projection_columns(df: pd.DataFrame, target_col: str) -> tuple[str | None, str | None, str]:
    """
    Resolve x/y/z columns for projection diagnostics.

    z defaults to target_col, then falls back to known z aliases.
    """
    x_col = find_first_existing_column(
        df,
        [
            "pipette_x_microns",
            "pipette_x",
            "pi_x",
            "x_microns",
            "x",
        ],
    )
    y_col = find_first_existing_column(
        df,
        [
            "pipette_y_microns",
            "pipette_y",
            "pi_y",
            "y_microns",
            "y",
        ],
    )

    z_candidates = [target_col, "pipette_z_microns", "pipette_z", "pi_z", "z_microns", "z"]
    z_col = find_first_existing_column(df, z_candidates)
    if z_col is None:
        raise ValueError(
            f"No z/target column found. Checked target_col='{target_col}' and known z aliases."
        )
    return x_col, y_col, z_col


def save_projection_panels(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    title: str,
    filepath: str,
    show_plot: bool = False,
) -> None:
    """Save x-y, x-z, and y-z 2D projection panels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_specs = [
        (x_col, y_col, "X-Y projection"),
        (x_col, z_col, "X-Z projection"),
        (y_col, z_col, "Y-Z projection"),
    ]

    n_points = len(df)
    marker_size = 4 if n_points > 50000 else 7
    marker_alpha = 0.22 if n_points > 50000 else 0.30

    for ax, (x_axis, y_axis, panel_title) in zip(axes, plot_specs):
        ax.scatter(
            df[x_axis],
            df[y_axis],
            s=marker_size,
            alpha=marker_alpha,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(panel_title)
        ax.grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(filepath, dpi=200)
    print(f"Saved projections: {filepath}")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_projection_placeholder(
    title: str,
    filepath: str,
    message: str,
    show_plot: bool = False,
) -> None:
    """Save a placeholder projection image when x/y columns are unavailable."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, panel_title in zip(axes, ["X-Y projection", "X-Z projection", "Y-Z projection"]):
        ax.set_title(panel_title)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(filepath, dpi=200)
    print(f"Saved projections placeholder: {filepath}")
    if show_plot:
        plt.show()
    plt.close(fig)


def build_variable_bins(
    series: pd.Series,
    inner_half_width: float = 3.0,
    inner_step: float = 0.3,
    outer_step: float = 1.0,
) -> np.ndarray:
    """
    Create symmetric, non-uniform bins: fine near zero, coarse in the tails.

    - Inner region: [-inner_half_width, inner_half_width] with inner_step spacing.
    - Outer regions: [-max_abs, -inner_half_width) and (inner_half_width, max_abs]
      with outer_step spacing.
    """
    max_abs = max(abs(series.min()), abs(series.max()))
    max_abs = max(max_abs, inner_half_width)  # ensure outer exists if data is tiny
    # Align to step sizes
    max_abs = np.ceil(max_abs / outer_step) * outer_step

    outer_neg = np.arange(-max_abs, -inner_half_width, outer_step)
    inner = np.arange(-inner_half_width, inner_half_width + inner_step, inner_step)
    outer_pos = np.arange(inner_half_width, max_abs + outer_step, outer_step)

    edges = np.concatenate([outer_neg, inner, outer_pos])
    # Remove any duplicates caused by step alignment
    return np.unique(np.round(edges, 6))


def normalize_source_folders(source_session_dirs: Iterable[str] | None) -> list[str]:
    """Normalize source folder paths while preserving input order."""
    if source_session_dirs is None:
        return []

    normalized_folders: list[str] = []
    seen: set[str] = set()
    for folder in source_session_dirs:
        folder_text = str(folder).strip()
        if not folder_text:
            continue
        normalized = os.path.normpath(os.path.abspath(folder_text))
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_folders.append(normalized)
    return normalized_folders


def enforce_min_label_spacing(
    df: pd.DataFrame,
    target_col: str,
    min_label_spacing_um: float,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """
    Keep rows so adjacent sorted labels are at least min_label_spacing_um apart.

    The dataframe is first shuffled deterministically to break ties, then stably
    sorted by target column. Selected rows are returned in original index order.
    """
    if min_label_spacing_um <= 0.0 or df.empty:
        return df.copy(), 0

    shuffled = df.sample(frac=1.0, random_state=seed)
    sorted_df = shuffled.sort_values(by=target_col, kind="mergesort")

    keep_indices: list[int] = []
    last_anchor_value: float | None = None
    # Treat tiny floating-point jitter as the same z-level.
    label_tolerance_um = max(1e-6, min_label_spacing_um * 1e-6)
    values = sorted_df[target_col].to_numpy(dtype=np.float64, copy=False)
    for index, value in zip(sorted_df.index.tolist(), values):
        if not np.isfinite(value):
            keep_indices.append(index)
            continue
        if last_anchor_value is None:
            keep_indices.append(index)
            last_anchor_value = float(value)
            continue

        delta = float(value) - last_anchor_value
        # Keep all duplicates/repeats of the currently accepted z-level.
        if abs(delta) <= label_tolerance_um:
            keep_indices.append(index)
            continue
        # Only open a new accepted z-level when spacing is met.
        if delta >= (min_label_spacing_um - label_tolerance_um):
            keep_indices.append(index)
            last_anchor_value = float(value)

    filtered = df.loc[keep_indices].sort_index(kind="mergesort").copy()
    removed_count = len(df) - len(filtered)
    return filtered, removed_count


def balance_defocus(
    input_path: str,
    target_col: str = "defocus_microns",
    cull_limit: float | None = 80.0,
    cap_per_bin: int = 50,
    min_label_spacing_um: float = MIN_LABEL_SPACING_UM,
    seed: int = 42,
    show_plots: bool = False,
    source_session_dirs: Iterable[str] | None = None,
) -> str:
    if threading.current_thread() is not threading.main_thread():
        try:
            plt.switch_backend("Agg")
        except Exception as exc:
            print(f"Warning: could not switch matplotlib backend to Agg: {exc}")
        if show_plots:
            print("show_plots=True requested outside main thread; disabling interactive plot display.")
            show_plots = False

    source_df = pd.read_csv(input_path)
    rows_before_cull = len(source_df)

    directory = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    run_ts = int(time.time())
    artifacts_dir = os.path.join(directory, f"{os.path.splitext(base)[0]}_sampled_{run_ts}_artifacts")
    output_path = os.path.join(artifacts_dir, base.replace(".csv", "_sampled.csv"))
    os.makedirs(artifacts_dir, exist_ok=True)
    normalized_source_folders = normalize_source_folders(source_session_dirs)
    source_folders_csv_path = os.path.join(artifacts_dir, "label_source_folders.csv")
    source_folders_df = pd.DataFrame(
        {
            "source_folder_index": list(range(1, len(normalized_source_folders) + 1)),
            "source_folder": normalized_source_folders,
        }
    )
    source_folders_df.to_csv(source_folders_csv_path, index=False)

    if target_col not in source_df.columns:
        target_col = source_df.columns[-1]
        print(f"Target column '{target_col}' inferred from last column.")

    x_col, y_col, z_col = resolve_projection_columns(source_df, target_col)
    if z_col != target_col:
        print(f"Using '{z_col}' as target column for balancing.")
        target_col = z_col

    df_before_cull = source_df.copy()

    # Optional symmetric culling
    if cull_limit is not None:
        df = df_before_cull[
            (df_before_cull[target_col] >= -cull_limit) & (df_before_cull[target_col] <= cull_limit)
        ].copy()
        print(f"After culling to +/-{cull_limit}: {len(df)} rows remain.")
    else:
        df = df_before_cull.copy()
        print(f"No culling applied. Rows loaded: {len(df)}.")

    if df.empty:
        raise ValueError("No data left after culling; adjust limits and retry.")

    if x_col is None or y_col is None:
        print(
            "X/Y columns not found for projection scatter plots. "
            f"Detected x_col={x_col}, y_col={y_col}, z_col={target_col}."
        )

    bins = build_variable_bins(
        df[target_col],
        inner_half_width=INNER_HALF_WIDTH,
        inner_step=INNER_STEP,
        outer_step=OUTER_STEP,
    )

    hist_before_path = os.path.join(artifacts_dir, "hist_before_balancing.png")
    save_histogram(
        df[target_col],
        bins=bins,
        title="Histogram before balancing (post-cull)",
        filepath=hist_before_path,
        xlabel=target_col,
        show_plot=show_plots,
    )

    df["bin"] = pd.cut(
        df[target_col],
        bins=bins,
        include_lowest=True,
        right=False,
    )

    bin_categories = df["bin"].cat.categories

    capped_df = df.groupby("bin", group_keys=False, observed=False).apply(
        lambda group: group.sample(
            n=min(len(group), cap_per_bin),
            random_state=seed,
        )
    )
    rows_after_cap = len(capped_df)
    sampled_df, rows_removed_by_spacing = enforce_min_label_spacing(
        capped_df,
        target_col=target_col,
        min_label_spacing_um=float(min_label_spacing_um),
        seed=seed,
    )
    rows_after_spacing = len(sampled_df)
    if min_label_spacing_um > 0.0:
        print(
            f"Min label spacing {min_label_spacing_um}um removed "
            f"{rows_removed_by_spacing} row(s) after capping."
        )
    else:
        print("Min label spacing disabled (<= 0).")

    hist_after_path = os.path.join(artifacts_dir, "hist_after_balancing.png")
    save_histogram(
        sampled_df[target_col],
        bins=bins,
        title=(
            f"Histogram after balancing (cap {cap_per_bin} per bin, "
            f"min spacing {min_label_spacing_um}um)"
            if min_label_spacing_um > 0.0
            else f"Histogram after balancing (cap {cap_per_bin} per bin)"
        ),
        filepath=hist_after_path,
        xlabel=target_col,
        show_plot=show_plots,
    )

    projections_before_balancing_path = os.path.join(
        artifacts_dir,
        "projections_before_balancing.png",
    )
    projections_after_balancing_path = os.path.join(
        artifacts_dir,
        "projections_after_balancing.png",
    )
    projections_generated = x_col is not None and y_col is not None
    if projections_generated:
        save_projection_panels(
            df,
            x_col=x_col,
            y_col=y_col,
            z_col=target_col,
            title="Pipette position projections before balancing (post-cull)",
            filepath=projections_before_balancing_path,
            show_plot=show_plots,
        )
        save_projection_panels(
            sampled_df,
            x_col=x_col,
            y_col=y_col,
            z_col=target_col,
            title=(
                f"Pipette position projections after balancing (cap {cap_per_bin}, "
                f"min spacing {min_label_spacing_um}um)"
                if min_label_spacing_um > 0.0
                else f"Pipette position projections after balancing (cap {cap_per_bin} per bin)"
            ),
            filepath=projections_after_balancing_path,
            show_plot=show_plots,
        )
    else:
        missing_message = (
            "Projection scatter unavailable.\n"
            f"Need X and Y columns.\nFound x_col={x_col}, y_col={y_col}."
        )
        save_projection_placeholder(
            title="Pipette position projections before balancing",
            filepath=projections_before_balancing_path,
            message=missing_message,
            show_plot=show_plots,
        )
        save_projection_placeholder(
            title="Pipette position projections after balancing",
            filepath=projections_after_balancing_path,
            message=missing_message,
            show_plot=show_plots,
        )

    # Save descriptive statistics and bin counts
    overall_stats = pd.DataFrame(
        {
            "before": df[target_col].describe(),
            "after": sampled_df[target_col].describe(),
        }
    )

    bin_counts = pd.DataFrame(
        {
            "before": df["bin"].value_counts().reindex(bin_categories, fill_value=0),
            "after": sampled_df["bin"].value_counts().reindex(bin_categories, fill_value=0),
        }
    )

    stats_df = pd.concat({"overall": overall_stats, "bin_counts": bin_counts})
    stats_path = os.path.join(artifacts_dir, "sample_statistics.csv")
    stats_df.to_csv(stats_path)

    balanced_df = sampled_df.drop(columns=["bin"])

    balanced_df.to_csv(output_path, index=False)

    # Record the recipe / parameters used for this sampling run
    recipe_items = [
        ("source_file", input_path),
        ("balanced_file", output_path),
        ("artifacts_dir", artifacts_dir),
        ("timestamp_unix", run_ts),
        ("cull_limit", cull_limit),
        ("projection_x_col", x_col or ""),
        ("projection_y_col", y_col or ""),
        ("projection_z_col", target_col),
        ("projection_plots_generated", projections_generated),
        ("hist_before_balancing_path", hist_before_path),
        ("hist_after_balancing_path", hist_after_path),
        ("projections_before_balancing_path", projections_before_balancing_path),
        ("projections_after_balancing_path", projections_after_balancing_path),
        ("inner_half_width", INNER_HALF_WIDTH),
        ("inner_step", INNER_STEP),
        ("outer_step", OUTER_STEP),
        ("cap_per_bin", cap_per_bin),
        ("min_label_spacing_um", min_label_spacing_um),
        ("seed", seed),
        ("rows_before_cull", rows_before_cull),
        ("rows_after_cull", len(df)),
        ("rows_removed_by_cull", rows_before_cull - len(df)),
        ("rows_after_cap", rows_after_cap),
        ("rows_after_spacing", rows_after_spacing),
        ("rows_removed_by_spacing", rows_removed_by_spacing),
        ("bin_count", len(bins) - 1),
        ("label_source_folder_count", len(normalized_source_folders)),
        ("label_source_folders_csv", source_folders_csv_path),
    ]
    recipe_items.extend(
        (f"label_source_folder_{idx:03d}", folder)
        for idx, folder in enumerate(normalized_source_folders, start=1)
    )
    recipe_df = pd.DataFrame(recipe_items, columns=["key", "value"])
    recipe_path = os.path.join(artifacts_dir, "recipe.csv")
    recipe_df.to_csv(recipe_path, index=False)

    print(f"Balanced rows: {len(balanced_df)}")
    print(f"Bins used: {len(bins) - 1}")
    print(f"Saved balanced CSV to: {output_path}")
    print(f"Artifacts (histograms, stats) saved to: {artifacts_dir}")
    print(f"Source folder list CSV saved to: {source_folders_csv_path}")
    print(f"Recipe saved to: {recipe_path}")

    return output_path


if __name__ == "__main__":
    balance_defocus(
        input_path=FILE_PATH,
        target_col=TARGET_COL,
        cull_limit=CULL_LIMIT,
        cap_per_bin=CAP_PER_BIN,
        min_label_spacing_um=MIN_LABEL_SPACING_UM,
        seed=SEED,
        show_plots=SHOW_PLOTS,
    )
