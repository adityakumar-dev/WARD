"""
WARD — Model Loader
====================
Loads both WARD models and their processors from Hugging Face.
Models are cached via @st.cache_resource so they are never reloaded
across Streamlit reruns.

Usage
-----
    from models.loader import load_models
    pair = load_models(cfg)
    pair.original_model  # AutoModelForImageClassification
    pair.original_proc   # AutoImageProcessor
    pair.fine_model
    pair.fine_proc
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ModelPair:
    original_model: object
    original_proc: object
    fine_model: object
    fine_proc: object
    device: str


def _load_single(model_id: str, device: str):
    """Load one model and its processor from Hugging Face (or local cache)."""
    import torch  # lazy import
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    logger.info("Loading model: %s → %s", model_id, device)
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    logger.info("Loaded: %s  labels=%s", model_id, model.config.id2label)
    return model, proc


def load_models(cfg) -> ModelPair:
    """
    Load both WARD models.  Called once and cached by the caller
    (Streamlit @st.cache_resource wraps this call in app.py).
    """
    import torch  # lazy import
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Inference device: %s", device)

    orig_model, orig_proc = _load_single(cfg.original_model_id, device)
    fine_model, fine_proc = _load_single(cfg.fine_model_id, device)

    return ModelPair(
        original_model=orig_model,
        original_proc=orig_proc,
        fine_model=fine_model,
        fine_proc=fine_proc,
        device=device,
    )
