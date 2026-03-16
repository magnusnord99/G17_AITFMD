"""Inference backends: dummy (for GUI testing), pytorch, onnx."""

from src.inference.backend.dummy_backend import predict_dummy

__all__ = ["predict_dummy"]
