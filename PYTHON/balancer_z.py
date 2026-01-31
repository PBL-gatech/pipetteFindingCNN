"""
Balance a z-only defocus CSV (filename, defocus_microns).

Edit the CONFIG section below to point at the CSV you want to balance.
The script trims extreme defocus values, caps each 1 micron bin to a
fixed sample count, and saves a balanced CSV alongside the input.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ CONFIG ------------
FILE_PATH = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\combined\pipette_z_data.csv"
TARGET_COL = "defocus_microns"  # falls back to last column if missing
CULL_LIMIT = 35               # set to None to disable symmetric culling
BIN_WIDTH = 1.0                 # microns
CAP_PER_BIN = 50
SEED = 42
SHOW_PLOTS = True
# ------------ CONFIG ------------


def build_bins(series: pd.Series, bin_width: float) -> np.ndarray:
    """Create symmetric bins around zero that cover the series range."""
    max_abs = max(abs(series.min()), abs(series.max()))
    max_abs = np.ceil(max_abs / bin_width) * bin_width  # expand to full bin
    return np.arange(-max_abs, max_abs + bin_width, bin_width)


def balance_defocus(
    input_path: str,
    target_col: str = "defocus_microns",
    cull_limit: float | None = 80.0,
    bin_width: float = 1.0,
    cap_per_bin: int = 50,
    seed: int = 42,
    show_plots: bool = False,
) -> str:
    df = pd.read_csv(input_path)

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

    bins = build_bins(df[target_col], bin_width)

    if show_plots:
        plt.figure(figsize=(10, 4))
        plt.hist(df[target_col], bins=bins, edgecolor="k", alpha=0.7)
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.title("Histogram before balancing")
        plt.tight_layout()
        plt.show()

    df["bin"] = pd.cut(
        df[target_col],
        bins=bins,
        include_lowest=True,
        right=False,
    )

    capped_df = df.groupby("bin", group_keys=False, observed=False).apply(
        lambda group: group.sample(
            n=min(len(group), cap_per_bin),
            random_state=seed,
        )
    )

    if show_plots:
        plt.figure(figsize=(10, 4))
        plt.hist(capped_df[target_col], bins=bins, edgecolor="k", alpha=0.7)
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.title(f"Histogram after balancing (cap {cap_per_bin} per bin)")
        plt.tight_layout()
        plt.show()

    balanced_df = capped_df.drop(columns=["bin"])

    directory = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    output_path = os.path.join(directory, base.replace(".csv", "_sampled.csv"))
    balanced_df.to_csv(output_path, index=False)

    print(f"Balanced rows: {len(balanced_df)}")
    print(f"Bins used: {len(bins) - 1} (width {bin_width})")
    print(f"Saved balanced CSV to: {output_path}")

    return output_path


if __name__ == "__main__":
    balance_defocus(
        input_path=FILE_PATH,
        target_col=TARGET_COL,
        cull_limit=CULL_LIMIT,
        bin_width=BIN_WIDTH,
        cap_per_bin=CAP_PER_BIN,
        seed=SEED,
        show_plots=SHOW_PLOTS,
    )
