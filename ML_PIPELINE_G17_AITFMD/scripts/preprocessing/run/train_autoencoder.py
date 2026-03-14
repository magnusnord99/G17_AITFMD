"""Train convolutional autoencoder on avg3 cubes for spectral compression."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.autoencoder import ConvAutoencoder


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / raw_path).resolve()


class HsiPatchDataset(Dataset):
    """Dataset that yields random (C, H, W) patches from HSI cubes.
    Caches recently loaded cubes to avoid re-reading large files from disk.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        input_root: Path,
        patch_size: int,
        patches_per_cube: int,
        seed: int = 42,
        max_cached_cubes: int = 10,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.input_root = Path(input_root)
        self.patch_size = patch_size
        self.patches_per_cube = patches_per_cube
        self.rng = np.random.default_rng(seed)
        self.max_cached_cubes = max_cached_cubes
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []

    def _load_cube(self, row_idx: int) -> np.ndarray:
        if row_idx in self._cache:
            self._cache_order.remove(row_idx)
            self._cache_order.append(row_idx)
            return self._cache[row_idx]

        row = self.rows.iloc[row_idx]
        cube_path = self.input_root / str(row["patient_id"]) / f"{row['roi_name']}.npy"
        cube = np.load(cube_path).astype(np.float32)

        while len(self._cache) >= self.max_cached_cubes and self._cache_order:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        self._cache[row_idx] = cube
        self._cache_order.append(row_idx)
        return cube

    def __len__(self) -> int:
        return len(self.rows) * self.patches_per_cube

    def __getitem__(self, idx: int) -> torch.Tensor:
        row_idx = idx // self.patches_per_cube
        cube = self._load_cube(row_idx)
        h, w, c = cube.shape

        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            cube = np.pad(cube, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            h, w, _ = cube.shape

        y = self.rng.integers(0, h - self.patch_size + 1) if h > self.patch_size else 0
        x = self.rng.integers(0, w - self.patch_size + 1) if w > self.patch_size else 0
        patch = cube[y : y + self.patch_size, x : x + self.patch_size, :]

        return torch.from_numpy(patch).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoencoder on avg3 train split.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/autoencoder.yaml")
    parser.add_argument("--max-cubes", type=int, default=None, help="Limit train cubes (smoke test).")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        config_path = PROJECT_ROOT / args.config
    cfg = load_yaml(config_path)

    paths = cfg["paths"]
    input_root = resolve_path(config_path, paths["input_root"])
    split_csv = resolve_path(config_path, paths["split_csv"])
    model_dir = resolve_path(config_path, paths["model_dir"])
    model_filename = str(paths["model_filename"])

    ae_cfg = cfg["autoencoder"]
    train_cfg = cfg["training"]
    seed = int(cfg.get("seed", 42))
    device_str = cfg.get("runtime", {}).get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str in ("cuda", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (M1/M2/M3/M4)
    else:
        device = torch.device("cpu")
    verbose = cfg.get("runtime", {}).get("verbose", True)

    in_channels = int(ae_cfg["in_channels"])
    latent_channels = int(ae_cfg["latent_channels"])
    patch_size = int(ae_cfg["patch_size"])
    patches_per_cube = int(ae_cfg.get("patches_per_cube", 8))
    max_cached_cubes = int(ae_cfg.get("max_cached_cubes", 10))
    max_train_cubes = ae_cfg.get("max_train_cubes")
    if max_train_cubes is not None:
        max_train_cubes = int(max_train_cubes)

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    lr = float(train_cfg["learning_rate"])
    val_fraction = float(train_cfg.get("val_fraction", 0.15))

    split_df = pd.read_csv(split_csv)
    split_df["input_path"] = split_df.apply(
        lambda r: str(input_root / str(r["patient_id"]) / f"{r['roi_name']}.npy"),
        axis=1,
    )
    split_df["exists"] = split_df["input_path"].map(lambda p: Path(p).exists())

    train_rows = split_df[(split_df["split"] == "train") & split_df["exists"]].reset_index(drop=True)
    if train_rows.empty:
        raise FileNotFoundError("No train cubes found.")

    if max_train_cubes is not None:
        train_rows = train_rows.head(max_train_cubes)
    if args.max_cubes is not None:
        train_rows = train_rows.head(args.max_cubes)

    train_rows = train_rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    indices = np.arange(len(train_rows))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = max(1, int(len(train_rows) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_sub = train_rows.iloc[train_idx].reset_index(drop=True)
    val_sub = train_rows.iloc[val_idx].reset_index(drop=True)

    train_ds = HsiPatchDataset(
        rows=train_sub,
        input_root=input_root,
        patch_size=patch_size,
        patches_per_cube=patches_per_cube,
        seed=seed,
        max_cached_cubes=max_cached_cubes,
    )
    val_ds = HsiPatchDataset(
        rows=val_sub,
        input_root=input_root,
        patch_size=patch_size,
        patches_per_cube=patches_per_cube,
        seed=seed + 1,
        max_cached_cubes=max_cached_cubes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ConvAutoencoder(in_channels=in_channels, latent_channels=latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    checkpoint_path = model_dir / (model_filename.replace(".pt", "_checkpoint.pt"))
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[ae] resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    training_state: dict = {"epoch": start_epoch}

    def _save_checkpoint(epoch: int, interrupted: bool = False) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "in_channels": in_channels,
                "latent_channels": latent_channels,
            },
            checkpoint_path,
        )
        if interrupted:
            print(f"\n[ae] paused. checkpoint saved: {checkpoint_path}", flush=True)
            print("[ae] run with --resume to continue.", flush=True)

    def _on_interrupt(*args) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        _save_checkpoint(training_state["epoch"], interrupted=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_interrupt)

    print(f"[ae] config: {config_path}")
    print(f"[ae] input_root: {input_root}")
    print(f"[ae] train cubes: {len(train_sub)}, val cubes: {len(val_sub)}")
    print(f"[ae] in_channels={in_channels}, latent_channels={latent_channels}")
    print(f"[ae] patch_size={patch_size}, epochs={epochs}, batch_size={batch_size}")
    print(f"[ae] device: {device}")
    print(f"[ae] cube cache: {max_cached_cubes} cubes (~{max_cached_cubes * 300 / 1024:.1f} GB RAM)")

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_filename

    print("[ae] starting training... (Ctrl+C to pause and save)", flush=True)
    for epoch in range(start_epoch, epochs):
        training_state["epoch"] = epoch
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=not verbose)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                val_loss += criterion(out, batch).item() * batch.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "latent_channels": latent_channels,
                    "epoch": epoch,
                },
                model_path,
            )

        _save_checkpoint(epoch)

        if verbose:
            marker = " *" if val_loss == best_val_loss else ""
            print(f"[ae] epoch {epoch + 1}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  best={best_val_loss:.6f}{marker}", flush=True)

    print(f"[ae] training done. best val_loss={best_val_loss:.6f}")
    print(f"[ae] saved model: {model_path}")


if __name__ == "__main__":
    main()
