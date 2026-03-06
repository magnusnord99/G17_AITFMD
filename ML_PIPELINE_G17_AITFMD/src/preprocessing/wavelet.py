"""Wavelet-based spectral reduction utilities for HSI cubes."""

from __future__ import annotations

import numpy as np
import pywt


def _validate_cube(cube: np.ndarray) -> tuple[int, int, int]:
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (H, W, B), got {cube.shape}")
    h, w, b = cube.shape
    if b < 2:
        raise ValueError(f"Expected at least 2 spectral bands, got {b}")
    return h, w, b


def _validate_level(n_bands: int, wavelet: str, level: int, label: str = "level") -> None:
    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n_bands, wav.dec_len)
    if level <= 0:
        raise ValueError(f"{label} must be > 0, got {level}")
    if level > max_level:
        raise ValueError(f"{label}={level} is too large for n_bands={n_bands}. Max allowed={max_level}.")


def _min_level_for_target(n_bands: int, target_len: int) -> int:
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")
    return int(np.ceil(np.log2(max(1.0, n_bands / float(target_len)))))


def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")
    src_len = int(signal.shape[0])
    if src_len == target_len:
        return signal.astype(np.float32, copy=False)
    if src_len == 1:
        return np.full((target_len,), float(signal[0]), dtype=np.float32)
    x_src = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(x_dst, x_src, signal.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def choose_wavelet_level_for_target(
    n_bands: int,
    wavelet: str,
    mode: str,
    target_bands: int,
) -> int:
    """Choose DWT level that keeps approximation length closest to target."""
    if target_bands <= 0:
        raise ValueError(f"target_bands must be > 0, got {target_bands}")

    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n_bands, wav.dec_len)
    if max_level <= 0:
        return 1

    best_level = 1
    best_delta = float("inf")
    current_len = n_bands
    for level in range(1, max_level + 1):
        current_len = pywt.dwt_coeff_len(current_len, wav.dec_len, mode)
        delta = abs(current_len - target_bands)
        if delta < best_delta:
            best_delta = delta
            best_level = level
    return best_level


def reduce_cube_wavelet_1d(
    cube: np.ndarray,
    target_bands: int = 32,
    wavelet: str = "db2",
    level: int | None = None,
    mode: str = "symmetric",
) -> np.ndarray:
    """
    Reduce spectral dimension using 1D wavelet approximation per pixel.

    Process:
    1) For each pixel spectrum (length B), run wavelet decomposition.
    2) Keep only approximation coefficients (low-frequency spectral content).
    3) Resample approximation vector to target_bands for fixed output shape.

    Returns:
      Reduced cube with shape (H, W, target_bands), dtype float32.
    """
    h, w, b = _validate_cube(cube)
    if target_bands <= 0:
        raise ValueError(f"target_bands must be > 0, got {target_bands}")

    if level is None:
        effective_level = choose_wavelet_level_for_target(
            n_bands=b,
            wavelet=wavelet,
            mode=mode,
            target_bands=target_bands,
        )
    else:
        effective_level = int(level)
        _validate_level(n_bands=b, wavelet=wavelet, level=effective_level)

    flat = cube.reshape(-1, b).astype(np.float32, copy=False)
    out = np.empty((flat.shape[0], target_bands), dtype=np.float32)

    for i in range(flat.shape[0]):
        coeffs = pywt.wavedec(flat[i], wavelet=wavelet, level=effective_level, mode=mode)
        approx = np.asarray(coeffs[0], dtype=np.float32)
        out[i] = _resample_1d(approx, target_bands)

    return out.reshape(h, w, target_bands).astype(np.float32, copy=False)


def reduce_cube_wavelet_approx_detail_padded(
    cube: np.ndarray,
    target_bands: int = 32,
    wavelet: str = "db2",
    level: int | None = None,
    mode: str = "periodization",
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Build fixed-size spectral features without interpolation:
      - first half bands = approximation coeffs cA_L
      - second half bands = detail coeffs cD_L

    Strategy:
      1) Pad spectral length B to n_pad = half * 2^L (>= B)
      2) Run 1D wavedec along spectral axis
      3) Concatenate cA_L and cD_L -> exact target_bands
    """
    h, w, b = _validate_cube(cube)
    if target_bands <= 0 or target_bands % 2 != 0:
        raise ValueError(f"target_bands must be a positive even integer, got {target_bands}")

    half = target_bands // 2
    min_level = _min_level_for_target(b, half)

    if level is None:
        effective_level = max(1, min_level)
    else:
        effective_level = int(level)
        if effective_level < min_level:
            raise ValueError(
                f"level={effective_level} is too small for B={b} and target_bands={target_bands}. "
                f"Need level >= {min_level}."
            )

    padded_bands = half * (2 ** effective_level)
    if padded_bands < b:
        raise RuntimeError("Internal error: padded length is smaller than input bands.")

    flat = cube.reshape(-1, b).astype(np.float32, copy=False)
    if padded_bands > b:
        pad_width = ((0, 0), (0, padded_bands - b))
        flat = np.pad(flat, pad_width=pad_width, mode=pad_mode)

    _validate_level(n_bands=padded_bands, wavelet=wavelet, level=effective_level, label="level")
    coeffs = pywt.wavedec(flat, wavelet=wavelet, level=effective_level, mode=mode, axis=1)
    approx = np.asarray(coeffs[0], dtype=np.float32)
    detail = np.asarray(coeffs[1], dtype=np.float32)  # cD_L

    if approx.shape[1] != half or detail.shape[1] != half:
        raise RuntimeError(
            "Unexpected coeff length after padding. "
            f"Expected {half}, got cA={approx.shape[1]}, cD={detail.shape[1]}."
        )

    out = np.concatenate([approx, detail], axis=1).astype(np.float32, copy=False)
    return out.reshape(h, w, target_bands).astype(np.float32, copy=False)


def reduce_cube_wavelet_approx_padded(
    cube: np.ndarray,
    target_bands: int = 32,
    wavelet: str = "db2",
    level: int | None = None,
    mode: str = "periodization",
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Build approx-only wavelet features without interpolation.

    Strategy:
      1) Pad spectral length B to n_pad = target_bands * 2^L (>= B)
      2) Run 1D wavedec along spectral axis
      3) Keep cA_L only, whose length is exactly target_bands
    """
    h, w, b = _validate_cube(cube)
    if target_bands <= 0:
        raise ValueError(f"target_bands must be > 0, got {target_bands}")

    min_level = _min_level_for_target(b, target_bands)
    if level is None:
        effective_level = max(1, min_level)
    else:
        effective_level = int(level)
        if effective_level < min_level:
            raise ValueError(
                f"level={effective_level} is too small for B={b} and target_bands={target_bands}. "
                f"Need level >= {min_level}."
            )

    padded_bands = target_bands * (2 ** effective_level)
    if padded_bands < b:
        raise RuntimeError("Internal error: padded length is smaller than input bands.")

    flat = cube.reshape(-1, b).astype(np.float32, copy=False)
    if padded_bands > b:
        pad_width = ((0, 0), (0, padded_bands - b))
        flat = np.pad(flat, pad_width=pad_width, mode=pad_mode)

    _validate_level(n_bands=padded_bands, wavelet=wavelet, level=effective_level, label="level")
    coeffs = pywt.wavedec(flat, wavelet=wavelet, level=effective_level, mode=mode, axis=1)
    approx = np.asarray(coeffs[0], dtype=np.float32)

    if approx.shape[1] != target_bands:
        raise RuntimeError(
            "Unexpected cA length after padding. "
            f"Expected {target_bands}, got cA={approx.shape[1]}."
        )

    return approx.reshape(h, w, target_bands).astype(np.float32, copy=False)
