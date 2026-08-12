"""
WARD — Local Inference Provider (DEV mode)
==========================================
Uses locally loaded PyTorch models for both WARD models.
Fusion logic is entirely delegated to models/fusion.py.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from inference.provider import InferenceProvider, InferenceUnavailableError
from models.fusion import FusionResult, fuse, predict_single
from models.loader import ModelPair

logger = logging.getLogger(__name__)


class LocalInferenceProvider(InferenceProvider):
    """DEV-mode inference: runs both models on local GPU/CPU."""

    def __init__(self, pair: ModelPair, cfg) -> None:
        self._pair = pair
        self._cfg = cfg

    @property
    def backend_name(self) -> str:
        device = self._pair.device.upper()
        return f"LOCAL GPU ({device})" if "cuda" in self._pair.device else f"LOCAL CPU"

    def predict(self, image: Image.Image) -> FusionResult:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            original_result = predict_single(
                model=self._pair.original_model,
                processor=self._pair.original_proc,
                image=image,
                model_name="original",
                device=self._pair.device,
            )
            fine_result = predict_single(
                model=self._pair.fine_model,
                processor=self._pair.fine_proc,
                image=image,
                model_name="fine_tuned",
                device=self._pair.device,
            )
            return fuse(
                original_result,
                fine_result,
                original_weight=self._cfg.original_weight,
                fine_weight=self._cfg.fine_weight,
                ambiguity_threshold=self._cfg.ambiguity_threshold,
            )
        except Exception as exc:
            logger.exception("LocalInferenceProvider.predict failed")
            raise InferenceUnavailableError(str(exc)) from exc
