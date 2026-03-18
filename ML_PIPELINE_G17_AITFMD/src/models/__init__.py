"""Model package – 3D CNN variants og registry."""

from src.models.registry import build_model, build_model_from_config, list_models

__all__ = ["build_model", "build_model_from_config", "list_models"]
