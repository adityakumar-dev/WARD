"""
WARD — Inference Provider Abstract Base Class
=============================================
Both LocalInferenceProvider and HFInferenceProvider implement this
interface.  The rest of the application only ever calls predict(image).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from models.fusion import FusionResult


class InferenceProvider(ABC):
    """Abstract provider — predict a PIL image, return FusionResult."""

    @abstractmethod
    def predict(self, image: Image.Image) -> FusionResult:
        """Run dual-model inference and return a fused result."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend description for the UI."""
        ...


class InferenceUnavailableError(RuntimeError):
    """Raised when an inference backend cannot fulfil a request."""
