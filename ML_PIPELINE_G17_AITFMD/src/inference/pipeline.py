"""
Preprocessing pipeline for single ROI: kalibrering → clipping → avg3 → spektral reduksjon (AE/PCA) → masking → patchifisering.

GUI sender input-sti; pipeline returnerer patches og metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from joblib import load

from src.preprocessing.calibrateClip import calibrate_cube, clip_cube, load_envi_cube
from src.preprocessing.pca import transform_cube_with_pca
from src.preprocessing.patching import iter_patches
from src.preprocessing.spectral_transform import reduce_bands_neighbor_average
from src.preprocessing.tissue_mask import build_tissue_mask


def _resolve_path(config_path: Path, raw_path: str, project_root: Path | None = None) -> Path:
    """Resolve path relative to project root."""
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    root = project_root if project_root is not None else config_path.parent.parent.parent
    return (root / raw_path).resolve()


def _roi_paths_from_input(input_path: str) -> tuple[Path, Path, Path, Path, Path]:
    """
    Resolve ROI folder and raw/dark/white paths from input.
    Input can be: path to ROI folder, or path to raw.hdr.
    """
    p = Path(input_path).resolve()
    if p.is_file():
        roi_dir = p.parent
        if p.name == "raw.hdr":
            raw_hdr = p
            raw_bin = roi_dir / "raw"
        else:
            raise ValueError(f"Expected raw.hdr or ROI folder, got {p}")
    elif p.is_dir():
        roi_dir = p
        raw_hdr = roi_dir / "raw.hdr"
        raw_bin = roi_dir / "raw"
    else:
        raise FileNotFoundError(f"Input does not exist: {p}")

    dark_hdr = roi_dir / "darkReference.hdr"
    dark_bin = roi_dir / "darkReference"
    white_hdr = roi_dir / "whiteReference.hdr"
    white_bin = roi_dir / "whiteReference"

    for name, path in [
        ("raw", raw_hdr),
        ("raw", raw_bin),
        ("darkReference", dark_hdr),
        ("darkReference", dark_bin),
        ("whiteReference", white_hdr),
        ("whiteReference", white_bin),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    return raw_hdr, raw_bin, dark_hdr, dark_bin, white_hdr, white_bin


def _transform_with_ae(cube: np.ndarray, model_path: Path) -> np.ndarray:
    """Transform avg3 cube (H,W,275) to (H,W,16) using trained AE encoder."""
    from src.preprocessing.autoencoder import ConvAutoencoder

    ckpt = torch.load(model_path, map_location="cpu")
    in_ch = ckpt.get("in_channels", 275)
    latent_ch = ckpt.get("latent_channels", 16)

    model = ConvAutoencoder(in_channels=in_ch, latent_channels=latent_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    h, w, c = cube.shape
    x = torch.from_numpy(cube.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        z = model.encode(x)
    out = z.squeeze(0).permute(1, 2, 0).numpy()
    return out.astype(np.float32)


def preprocess_single_roi(
    input_path: str,
    config_path: Path,
    project_root: Path | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray, dict]:
    """
    Full preprocessing for one ROI: calibrate → clip → avg3 → spektral reduksjon (AE/PCA) → mask → patches.

    Returns:
        patches: list of (H, W, C) arrays
        patch_coords: list of (y, x) top-left for each patch
        reduced_cube: (H, W, 16) for optional heatmap/overlay
        metadata: dict with num_patches, cube_shape, etc.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths_cfg = cfg["paths"]
    pre_cfg = cfg["preprocessing"]
    cal = pre_cfg["calibration"]
    spec = pre_cfg["spectral"]
    mask_cfg = pre_cfg["tissue_mask"]
    patch_cfg = pre_cfg["patching"]

    reducer = str(paths_cfg.get("spectral_reducer", "ae"))

    def _res(p: str) -> Path:
        return _resolve_path(config_path, p, project_root)

    raw_hdr, raw_bin, dark_hdr, dark_bin, white_hdr, white_bin = _roi_paths_from_input(input_path)

    raw = load_envi_cube(raw_hdr, raw_bin)
    dark = load_envi_cube(dark_hdr, dark_bin)
    white = load_envi_cube(white_hdr, white_bin)

    cube = calibrate_cube(raw, dark, white, eps=float(cal["eps"]))
    cube = clip_cube(cube, clip_min=float(cal["clip_min"]), clip_max=float(cal["clip_max"]))

    cube = reduce_bands_neighbor_average(cube, window=int(spec["reduction_window"]))
    mask = build_tissue_mask(
        cube,
        method=str(mask_cfg["method"]),
        min_object_size=int(mask_cfg["min_object_size"]),
        min_hole_size=int(mask_cfg["min_hole_size"]),
        tissue_side=str(mask_cfg.get("tissue_side", "dark")),
    )

    if reducer == "ae":
        ae_path = _res(paths_cfg["ae_model"])
        if not ae_path.exists():
            raise FileNotFoundError(f"AE model not found: {ae_path}")
        reduced_cube = _transform_with_ae(cube.astype(np.float32), ae_path)
    else:
        pca_path = _res(paths_cfg["pca_model"])
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        pca_model = load(pca_path)
        reduced_cube = transform_cube_with_pca(cube.astype(np.float32), pca_model)

    patch_h = int(patch_cfg["patch_h"])
    patch_w = int(patch_cfg["patch_w"])
    stride_h = int(patch_cfg["stride_h"])
    stride_w = int(patch_cfg["stride_w"])
    min_tissue = float(patch_cfg["min_tissue_ratio"])

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for patch, y, x in iter_patches(
        reduced_cube,
        patch_h=patch_h,
        patch_w=patch_w,
        stride_h=stride_h,
        stride_w=stride_w,
        mask=mask,
        min_tissue_ratio=min_tissue,
    ):
        patches.append(patch)
        coords.append((y, x))

    metadata = {
        "num_patches": len(patches),
        "cube_shape": list(reduced_cube.shape),
        "input_path": str(input_path),
        "spectral_reducer": reducer,
    }

    return patches, coords, reduced_cube, metadata
