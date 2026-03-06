"""Visualize wavelet-reduced cubes as pseudo-RGB previews."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    lo = np.percentile(img, 2.0)
    hi = np.percentile(img, 98.0)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _cube_to_rgb(cube: np.ndarray) -> np.ndarray:
    """Map (H, W, C) wavelet cube to pseudo-RGB in [0, 1]."""
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (H, W, C), got {cube.shape}")
    c = cube.shape[2]
    if c < 3:
        gray = _normalize01(cube.mean(axis=2))
        return np.stack([gray, gray, gray], axis=-1)

    ridx = int(round(0.75 * (c - 1)))
    gidx = int(round(0.50 * (c - 1)))
    bidx = int(round(0.25 * (c - 1)))
    rgb = np.stack([cube[:, :, ridx], cube[:, :, gidx], cube[:, :, bidx]], axis=-1)
    return _normalize01(rgb)


def _save_preview(cube_path: Path, out_path: Path, title: str) -> None:
    cube = np.load(cube_path, allow_pickle=False)
    rgb = _cube_to_rgb(cube)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pseudo-RGB previews from wavelet manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to wavelet manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/wavelet_previews",
        help="Output directory for PNG previews.",
    )
    parser.add_argument("--max-samples", type=int, default=20, help="Max previews to write.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random row sampling.")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test", None],
        help="Optional split filter before sampling.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {"patient_id", "roi_name", "split", "output_path", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    written_df = df[df["status"] == "written"].copy()
    if args.split is not None:
        written_df = written_df[written_df["split"] == args.split].copy()
    if written_df.empty:
        raise ValueError("No matching rows with status='written' after filtering.")

    n = min(args.max_samples, len(written_df))
    selected = written_df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    saved = 0
    for _, row in selected.iterrows():
        patient_id = str(row["patient_id"])
        roi_name = str(row["roi_name"])
        split = str(row["split"])
        cube_path = Path(str(row["output_path"]))
        if not cube_path.exists():
            print(f"Skipping missing file: {cube_path}")
            continue

        name = f"{split}__{patient_id}__{roi_name}.png"
        out_path = output_dir / name
        _save_preview(cube_path=cube_path, out_path=out_path, title=f"{split} | {patient_id}/{roi_name}")
        saved += 1

    print(f"Saved previews: {saved}")
    print(f"Preview directory: {output_dir}")


if __name__ == "__main__":
    main()
