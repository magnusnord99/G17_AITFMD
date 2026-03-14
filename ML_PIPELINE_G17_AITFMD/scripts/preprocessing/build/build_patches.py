"""Build patches from preprocessed cubes, optionally filtered by tissue mask."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.patching import get_patient_splits, make_patches


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_from_config(config_path: Path, raw_path: str) -> Path:
    """Resolve path relative to config file directory."""
    return (config_path.parent / raw_path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build patches from preprocessed cubes. Config in preprocessing.yaml."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing/preprocessing.yaml",
        help="Path to preprocessing YAML config.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override input cube root (e.g. data/processed/pca32).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output patch root.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Override mask root (e.g. data/interim/masks/final_mask).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Override splits CSV path.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Override output subdir under patches_dir (e.g. pca32).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix for output dir (e.g. _masked). Output becomes {output_subdir}{suffix}.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    cfg = _load_yaml(config_path)

    paths = cfg["paths"]
    patch_cfg = cfg["patching"]

    # Resolve base paths (relative to config file)
    processed_dir = _resolve_from_config(config_path, paths["processed_dir"])
    patches_dir = _resolve_from_config(config_path, paths["patches_dir"])
    masks_dir = _resolve_from_config(config_path, paths["masks_dir"])
    splits_dir = _resolve_from_config(config_path, paths["splits_dir"])

    # Input: cubes to patch
    if args.input:
        input_root = Path(args.input).expanduser().resolve()
    else:
        input_root = processed_dir / patch_cfg["input_subdir"]

    # Output: where to write patches
    output_subdir = args.output_subdir or patch_cfg["output_subdir"]
    if args.output_suffix:
        output_subdir = f"{output_subdir}{args.output_suffix}"
    if args.output:
        output_root = Path(args.output).expanduser().resolve()
    else:
        output_root = patches_dir / output_subdir

    # Mask: tissue masks for filtering
    if args.mask:
        mask_root = Path(args.mask).expanduser().resolve()
    else:
        mask_root = masks_dir / patch_cfg["mask_subdir"]

    # Splits
    if args.splits:
        splits_path = Path(args.splits).expanduser().resolve()
    else:
        splits_path = splits_dir / patch_cfg["splits_file"]

    splits = get_patient_splits(splits_path)
    min_tissue_ratio = float(patch_cfg["min_tissue_ratio"])
    effective_mask_root = mask_root if mask_root.exists() else None

    if effective_mask_root:
        print(f"Using tissue masks from {effective_mask_root} (min_tissue_ratio={min_tissue_ratio})")
    else:
        print(f"Mask root {mask_root} not found; including all patches (no mask filter)")

    manifest_df = make_patches(
        input_root=input_root,
        output_root=output_root,
        splits=splits,
        patch_size_h=int(patch_cfg["patch_h"]),
        patch_size_w=int(patch_cfg["patch_w"]),
        stride_h=int(patch_cfg["stride_h"]),
        stride_w=int(patch_cfg["stride_w"]),
        mask_root=effective_mask_root,
        min_tissue_ratio=min_tissue_ratio,
    )
    print(f"saved patches: {len(manifest_df)}")
    print(f"manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
