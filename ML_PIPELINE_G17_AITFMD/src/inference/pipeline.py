"""
Preprocessing pipeline for single ROI.

Steg lastes fra configs/preprocessing/pipeline.yaml (eller overstyrt i inference-config).
Hvert steg kan slås av med enabled: false.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from joblib import load

from src.preprocessing.calibrateClip import calibrate_cube, clip_cube, load_envi_cube
from src.preprocessing.pca import transform_cube_with_pca
from src.preprocessing.patching import count_patch_grid, iter_patches
from src.preprocessing.spectral_transform import reduce_bands_neighbor_average
from src.preprocessing.tissue_mask import build_tissue_mask, tissue_ratio


def _load_pipeline_config(inference_config: dict, config_path: Path, project_root: Path) -> dict:
    """Last pipeline-config fra preprocessing.pipeline_config eller default pipeline.yaml."""
    pre = inference_config.get("preprocessing", {})
    pipeline_path = pre.get("pipeline_config", "configs/preprocessing/pipeline.yaml")
    p = Path(pipeline_path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Pipeline-config ikke funnet: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


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


def _transform_with_wavelet(cube: np.ndarray, wave_cfg: dict) -> np.ndarray:
    """Transform avg3 cube med wavelet."""
    from src.preprocessing.wavelet import (
        reduce_cube_wavelet_1d,
        reduce_cube_wavelet_approx_padded,
        reduce_cube_wavelet_approx_detail_padded,
    )
    mode = str(wave_cfg.get("feature_mode", "approx_padded"))
    target = int(wave_cfg.get("target_bands", 16))
    wavelet = str(wave_cfg.get("wavelet", "db2"))
    pywt_mode = str(wave_cfg.get("mode", "periodization"))
    pad_mode = str(wave_cfg.get("pad_mode", "edge"))
    if mode == "approx_padded":
        return reduce_cube_wavelet_approx_padded(
            cube, target_bands=target, wavelet=wavelet, mode=pywt_mode, pad_mode=pad_mode
        )
    if mode == "approx_detail":
        return reduce_cube_wavelet_approx_detail_padded(
            cube, target_bands=target, wavelet=wavelet, mode=pywt_mode, pad_mode=pad_mode
        )
    return reduce_cube_wavelet_1d(cube, target_bands=target, wavelet=wavelet, mode=pywt_mode)


def preprocess_single_roi(
    input_path: str,
    config_path: Path,
    project_root: Path | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray, dict]:
    """
    Full preprocessing for one ROI. Steg og parametre leses fra pipeline-config.

    Returns:
        patches: list of (H, W, C) arrays
        patch_coords: list of (y, x) top-left for each patch
        reduced_cube: (H, W, C) for optional heatmap/overlay
        metadata: dict with num_patches, cube_shape, etc.
    """
    root = project_root if project_root is not None else config_path.parent.parent.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pl_cfg = _load_pipeline_config(cfg, config_path, root)
    pipe = pl_cfg["pipeline"].copy()
    pre_cfg = cfg.get("preprocessing", {})
    pipeline_config_relative = str(pre_cfg.get("pipeline_config", "configs/preprocessing/pipeline.yaml"))
    paths_cfg = cfg.get("paths", {})

    # Overstyr patch-størrelse fra modell-config (én kilde ved trening og inferanse)
    model_cfg_path = cfg.get("model", {}).get("model_config_path")
    if model_cfg_path:
        mc_path = Path(model_cfg_path).expanduser()
        if not mc_path.is_absolute():
            mc_path = (root / mc_path).resolve()
        if mc_path.exists():
            model_cfg = yaml.safe_load(mc_path.read_text(encoding="utf-8"))
            model_input = model_cfg.get("input", {})
            if model_input:
                pipe.setdefault("patching", {})
                pipe["patching"]["patch_h"] = int(model_input.get("patch_h", 64))
                pipe["patching"]["patch_w"] = int(model_input.get("patch_w", 64))

    def _res(p: str) -> Path:
        return _resolve_path(config_path, p, root)

    def _path(key: str, default: str) -> Path:
        spec = pipe.get("spectral_reduction", {})
        val = spec.get(key) or paths_cfg.get(key) or default
        return _res(str(val))

    raw_hdr, raw_bin, dark_hdr, dark_bin, white_hdr, white_bin = _roi_paths_from_input(input_path)

    raw = load_envi_cube(raw_hdr, raw_bin)
    dark = load_envi_cube(dark_hdr, dark_bin)
    white = load_envi_cube(white_hdr, white_bin)

    cal = pipe.get("calibration", {})
    cube = calibrate_cube(raw, dark, white, eps=float(cal.get("eps", 1.0e-8)))

    clip_cfg = pipe.get("clip", {})
    if clip_cfg.get("enabled", True):
        cube = clip_cube(
            cube,
            clip_min=float(clip_cfg.get("clip_min", 0.0)),
            clip_max=float(clip_cfg.get("clip_max", 1.0)),
        )

    avg3_cfg = pipe.get("avg3", {})
    if avg3_cfg.get("enabled", True):
        cube = reduce_bands_neighbor_average(cube, window=int(avg3_cfg.get("reduction_window", 3)))

    mask_cfg = pipe.get("tissue_mask", {})
    mask = None
    tissue_mask_meta: dict
    if mask_cfg.get("enabled", True):
        mask = build_tissue_mask(
            cube,
            method=str(mask_cfg.get("method", "mean_std_percentile")),
            min_object_size=int(mask_cfg.get("min_object_size", 500)),
            min_hole_size=int(mask_cfg.get("min_hole_size", 1000)),
            tissue_side=str(mask_cfg.get("tissue_side", "dark")),
        )
        tr = tissue_ratio(mask)
        tissue_mask_meta = {
            "enabled": True,
            "method": str(mask_cfg.get("method", "mean_std_percentile")),
            "tissue_fraction": round(tr, 6),
            "background_fraction": round(1.0 - tr, 6),
            "tissue_percent": round(100.0 * tr, 2),
            "background_percent": round(100.0 * (1.0 - tr), 2),
        }
    else:
        tissue_mask_meta = {
            "enabled": False,
            "method": None,
            "tissue_fraction": 1.0,
            "background_fraction": 0.0,
            "tissue_percent": 100.0,
            "background_percent": 0.0,
            "note": "Tissue mask disabled; no pixels classified as background by mask.",
        }

    spec_cfg = pipe.get("spectral_reduction", {})
    reducer = str(spec_cfg.get("reducer", paths_cfg.get("spectral_reducer", "pca")))
    if spec_cfg.get("enabled", True):
        if reducer == "ae":
            ae_path = _path("ae_model", "models/ae_avg3_16.pt")
            if not ae_path.exists():
                raise FileNotFoundError(f"AE model not found: {ae_path}")
            cube = _transform_with_ae(cube.astype(np.float32), ae_path)
        elif reducer == "wavelet":
            wave_cfg = spec_cfg.get("wavelet", {})
            cube = _transform_with_wavelet(cube.astype(np.float32), wave_cfg)
        else:
            pca_path = _path("pca_model", "models/pca_avg3_16.joblib")
            if not pca_path.exists():
                raise FileNotFoundError(f"PCA model not found: {pca_path}")
            pca_model = load(pca_path)
            cube = transform_cube_with_pca(cube.astype(np.float32), pca_model)
    reduced_cube = cube

    patch_cfg = pipe.get("patching", {})
    patch_h = int(patch_cfg.get("patch_h", 64))
    patch_w = int(patch_cfg.get("patch_w", 64))
    stride_h = int(patch_cfg.get("stride_h", 32))
    stride_w = int(patch_cfg.get("stride_w", 32))
    min_tissue = float(patch_cfg.get("min_tissue_ratio", 0.60))

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    patch_stats: dict[str, int]
    if patch_cfg.get("enabled", True):
        patch_stats = count_patch_grid(
            reduced_cube,
            patch_h=patch_h,
            patch_w=patch_w,
            stride_h=stride_h,
            stride_w=stride_w,
            mask=mask,
            min_tissue_ratio=min_tissue,
        )
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
    else:
        patch_stats = {"total_possible": 0, "filtered_by_tissue": 0, "evaluated": 0}

    pipeline_steps: dict[str, dict] = {}
    for step_key in ("calibration", "clip", "avg3", "tissue_mask", "spectral_reduction", "patching"):
        step = pipe.get(step_key)
        if isinstance(step, dict):
            pipeline_steps[step_key] = {"enabled": bool(step.get("enabled", True))}

    metadata = {
        "num_patches": len(patches),
        "cube_shape": list(reduced_cube.shape),
        "input_path": str(input_path),
        "spectral_reducer": reducer,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "min_tissue_ratio_patch": min_tissue,
        "tissue_mask": tissue_mask_meta,
        "patch_stats": patch_stats,
        "pipeline_config_relative": pipeline_config_relative,
        "pipeline_steps": pipeline_steps,
    }

    return patches, coords, reduced_cube, metadata
