"""Compare float32 vs float16 avg3 subsets using PCA explained variance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pca import fit_pca_from_pixels, flatten_cube


def _collect_npy_paths(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.npy")
        if not p.name.startswith(".") and not p.name.startswith("._")
    )


def _sample_pixels(paths: list[Path], pixels_per_roi: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []

    for path in paths:
        cube = np.load(path)
        flat = flatten_cube(cube.astype(np.float32, copy=False))
        if flat.shape[0] > pixels_per_roi:
            idx = rng.choice(flat.shape[0], size=pixels_per_roi, replace=False)
            flat = flat[idx]
        chunks.append(flat)

    if not chunks:
        raise ValueError("No cubes found for sampling.")
    return np.concatenate(chunks, axis=0)


def _format_vec(vec: np.ndarray, limit: int = 8) -> str:
    shown = vec[:limit]
    body = ", ".join(f"{x:.6f}" for x in shown)
    suffix = ", ..." if len(vec) > limit else ""
    return f"[{body}{suffix}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PCA variance for avg3 float32 vs float16.")
    parser.add_argument("--float32-root", required=True, type=str)
    parser.add_argument("--float16-root", required=True, type=str)
    parser.add_argument("--n-components", type=int, default=24)
    parser.add_argument("--pixels-per-roi", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    float32_root = Path(args.float32_root).resolve()
    float16_root = Path(args.float16_root).resolve()

    paths32 = _collect_npy_paths(float32_root)
    paths16 = _collect_npy_paths(float16_root)
    common = sorted(set(p.relative_to(float32_root) for p in paths32) & set(p.relative_to(float16_root) for p in paths16))
    if not common:
        raise FileNotFoundError("No matching .npy files found between float32 and float16 roots.")

    full32 = [float32_root / rel for rel in common]
    full16 = [float16_root / rel for rel in common]

    x32 = _sample_pixels(full32, pixels_per_roi=args.pixels_per_roi, seed=args.seed)
    x16 = _sample_pixels(full16, pixels_per_roi=args.pixels_per_roi, seed=args.seed)

    if x32.shape != x16.shape:
        raise ValueError(f"Sampled matrices have different shapes: {x32.shape} vs {x16.shape}")

    pca32 = fit_pca_from_pixels(x32, n_components=args.n_components, random_state=args.seed)
    pca16 = fit_pca_from_pixels(x16, n_components=args.n_components, random_state=args.seed)

    evr32 = pca32.explained_variance_ratio_
    evr16 = pca16.explained_variance_ratio_
    abs_diff = np.abs(evr32 - evr16)

    print(f"Compared {len(common)} matching cubes")
    print(f"Sample matrix shape: {x32.shape}")
    print(f"n_components: {args.n_components}")
    print("")
    print(f"sum(evr) float32: {evr32.sum():.6f}")
    print(f"sum(evr) float16: {evr16.sum():.6f}")
    print(f"abs diff (sum): {abs(evr32.sum() - evr16.sum()):.6f}")
    print(f"abs diff (mean per component): {abs_diff.mean():.6f}")
    print(f"abs diff (max component): {abs_diff.max():.6f}")
    print("")
    print(f"evr float32: {_format_vec(evr32)}")
    print(f"evr float16: {_format_vec(evr16)}")
    print(f"abs diff:    {_format_vec(abs_diff)}")


if __name__ == "__main__":
    main()
