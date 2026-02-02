"""
Balance a z-only defocus CSV (filename, defocus_microns).

Edit the CONFIG section below to point at the CSV you want to balance.
The script trims extreme defocus values, caps each bin to a fixed sample
count, and saves a balanced CSV alongside the input.
"""

import os
import time
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


def balance_defocus(
    input_path: str,
    target_col: str = "defocus_microns",
    cull_limit: float | None = 80.0,
    cap_per_bin: int = 50,
    seed: int = 42,
    show_plots: bool = False,
) -> str:
    df = pd.read_csv(input_path)
    rows_before_cull = len(df)

    directory = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    run_ts = int(time.time())
    artifacts_dir = os.path.join(directory, f"{os.path.splitext(base)[0]}_sampled_{run_ts}_artifacts")
    output_path = os.path.join(artifacts_dir, base.replace(".csv", "_sampled.csv"))
    os.makedirs(artifacts_dir, exist_ok=True)

    if target_col not in df.columns:
        target_col = df.columns[-1]
        print(f"Target column '{target_col}' inferred from last column.")

    # Optional symmetric culling
    if cull_limit is not None:
        df = df[(df[target_col] >= -cull_limit) & (df[target_col] <= cull_limit)]
        print(f"After culling to +/-{cull_limit}: {len(df)} rows remain.")
    else:
        print(f"No culling applied. Rows loaded: {len(df)}.")

    if df.empty:
        raise ValueError("No data left after culling; adjust limits and retry.")

    bins = build_variable_bins(
        df[target_col],
        inner_half_width=INNER_HALF_WIDTH,
        inner_step=INNER_STEP,
        outer_step=OUTER_STEP,
    )

    hist_before_path = os.path.join(artifacts_dir, "hist_before.png")
    save_histogram(
        df[target_col],
        bins=bins,
        title="Histogram before balancing",
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

    hist_after_path = os.path.join(artifacts_dir, "hist_after.png")
    save_histogram(
        capped_df[target_col],
        bins=bins,
        title=f"Histogram after balancing (cap {cap_per_bin} per bin)",
        filepath=hist_after_path,
        xlabel=target_col,
        show_plot=show_plots,
    )

    # Save descriptive statistics and bin counts
    overall_stats = pd.DataFrame(
        {
            "before": df[target_col].describe(),
            "after": capped_df[target_col].describe(),
        }
    )

    bin_counts = pd.DataFrame(
        {
            "before": df["bin"].value_counts().reindex(bin_categories, fill_value=0),
            "after": capped_df["bin"].value_counts().reindex(bin_categories, fill_value=0),
        }
    )

    stats_df = pd.concat({"overall": overall_stats, "bin_counts": bin_counts})
    stats_path = os.path.join(artifacts_dir, "sample_statistics.csv")
    stats_df.to_csv(stats_path)

    balanced_df = capped_df.drop(columns=["bin"])

    balanced_df.to_csv(output_path, index=False)

    # Record the recipe / parameters used for this sampling run
    recipe_items = [
        ("source_file", input_path),
        ("balanced_file", output_path),
        ("artifacts_dir", artifacts_dir),
        ("timestamp_unix", run_ts),
        ("cull_limit", cull_limit),
        ("inner_half_width", INNER_HALF_WIDTH),
        ("inner_step", INNER_STEP),
        ("outer_step", OUTER_STEP),
        ("cap_per_bin", cap_per_bin),
        ("seed", seed),
        ("rows_before_cull", rows_before_cull),
        ("rows_after_cull", len(df)),
        ("rows_after_cap", len(balanced_df)),
        ("bin_count", len(bins) - 1),
    ]
    recipe_df = pd.DataFrame(recipe_items, columns=["key", "value"])
    recipe_path = os.path.join(artifacts_dir, "recipe.csv")
    recipe_df.to_csv(recipe_path, index=False)

    print(f"Balanced rows: {len(balanced_df)}")
    print(f"Bins used: {len(bins) - 1}")
    print(f"Saved balanced CSV to: {output_path}")
    print(f"Artifacts (histograms, stats) saved to: {artifacts_dir}")
    print(f"Recipe saved to: {recipe_path}")

    return output_path


if __name__ == "__main__":
    balance_defocus(
        input_path=FILE_PATH,
        target_col=TARGET_COL,
        cull_limit=CULL_LIMIT,
        cap_per_bin=CAP_PER_BIN,
        seed=SEED,
        show_plots=SHOW_PLOTS,
    )
