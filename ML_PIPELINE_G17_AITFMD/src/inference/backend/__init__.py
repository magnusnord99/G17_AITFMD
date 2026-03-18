"""Inference backends: dummy (for GUI testing), pytorch, onnx."""

from src.inference.backend.dummy_backend import predict_dummy
from src.inference.backend.pytorch_backend import predict_pytorch

__all__ = ["predict_dummy", "predict_pytorch"]
