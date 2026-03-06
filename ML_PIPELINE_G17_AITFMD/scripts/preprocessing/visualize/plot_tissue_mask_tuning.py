"""Plot and rank tissue-mask parameter tuning results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    fmt: str = ".3f",
) -> None:
    pivot = df.pivot(index="min_object_size", columns="min_hole_size", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("min_hole_size")
    ax.set_ylabel("min_object_size")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, format(float(val), fmt), ha="center", va="center", color="white", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tissue-mask tuning summary.")
    parser.add_argument(
        "--summary-csv",
        type=str,
        required=True,
        help="Path to tuning summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/masks/tuning_plots",
        help="Directory where report plots and ranked CSV are saved.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    required_cols = {
        "run_name",
        "min_object_size",
        "min_hole_size",
        "all_mean",
        "all_p10",
        "all_lt_01",
        "T_mean",
        "NT_mean",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")

    # Helpful derived metrics for ranking/reporting.
    ranked = df.copy()
    ranked["class_gap"] = ranked["T_mean"] - ranked["NT_mean"]
    ranked["score"] = (
        ranked["all_mean"]
        + 0.50 * ranked["all_p10"]
        - 0.010 * ranked["all_lt_01"]
        - 0.30 * ranked["class_gap"]
    )
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.to_csv(out_dir / "ranked_summary.csv", index=False)

    _save_heatmap(
        ranked,
        value_col="all_mean",
        title="Tissue Ratio Mean (all_mean)",
        out_path=out_dir / "heatmap_all_mean.png",
    )
    _save_heatmap(
        ranked,
        value_col="all_p10",
        title="Lower Tail Tissue Ratio (all_p10)",
        out_path=out_dir / "heatmap_all_p10.png",
    )
    _save_heatmap(
        ranked,
        value_col="class_gap",
        title="Class Gap (T_mean - NT_mean)",
        out_path=out_dir / "heatmap_class_gap.png",
    )

    # Trade-off view: overall tissue retention vs low-ratio failures.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ranked["all_lt_01"], ranked["all_mean"], s=45)
    for _, row in ranked.iterrows():
        ax.annotate(str(row["run_name"]), (row["all_lt_01"], row["all_mean"]), fontsize=8, alpha=0.8)
    ax.set_xlabel("all_lt_01 (count of very low-ratio masks)")
    ax.set_ylabel("all_mean")
    ax.set_title("Trade-off: Tissue Retention vs Very Low-Ratio Masks")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_tradeoff_mean_vs_lt01.png", dpi=160)
    plt.close(fig)

    print(f"Saved report plots to: {out_dir}")
    print(f"Saved ranked summary to: {out_dir / 'ranked_summary.csv'}")
    print("\nTop 5 runs by score:")
    print(ranked[["run_name", "score", "all_mean", "all_p10", "all_lt_01", "class_gap"]].head(5))


if __name__ == "__main__":
    main()
