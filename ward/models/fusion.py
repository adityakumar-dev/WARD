"""
WARD — Fusion Engine
=====================
Defines PredictionResult, FusionResult, and all fusion logic.
This is the SINGLE source of truth for how models are combined.
Both LocalInferenceProvider and HFInferenceProvider call fuse() here.

Fusion rules
------------
dry_score   = 0.30 * original_dry   + 0.70 * fine_dry
wet_score   = 0.30 * original_wet   + 0.70 * fine_wet
damp_score  = fine_damp                        (fine only)
drying_score = fine_drying                     (fine only)

Snow class from original model is silently excluded.
Missing classes default to 0.0 — never invented.
Scores are normalised to sum to 1.0 after fusion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from models.labels import WARD_CLASSES, ORIGINAL_FUSION_CLASSES, build_id_map


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Output of a single model inference call."""

    model_name: str
    scores: dict[str, float]          # keys: subset of WARD_CLASSES
    top_label: str
    top_confidence: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    # Convenience: get a score for any WARD class (0.0 if not supported)
    def get_score(self, label: str) -> float:
        return self.scores.get(label, 0.0)


@dataclass
class FusionResult:
    """Combined output of the two WARD models."""

    original: PredictionResult
    fine: PredictionResult
    fused_scores: dict[str, float]   # all four WARD classes, normalised
    top_label: str
    top_confidence: float
    agreement: bool                  # do both models agree on top label?
    ambiguous: bool                  # top1 - top2 < threshold
    original_label: str
    fine_label: str
    timestamp: float = field(default_factory=time.time)

    @property
    def margin(self) -> float:
        """Difference between the top two fused scores."""
        sorted_scores = sorted(self.fused_scores.values(), reverse=True)
        if len(sorted_scores) < 2:
            return 1.0
        return sorted_scores[0] - sorted_scores[1]


# ── Single-model inference ──────────────────────────────────────────────────

def predict_single(
    model,
    processor,
    image: Image.Image,
    model_name: str,
    device: str,
) -> PredictionResult:
    """
    Run inference with one model on a PIL RGB image.
    Returns PredictionResult with normalised scores for WARD classes only.
    """
    import torch  # lazy import so module is importable without torch installed

    t0 = time.perf_counter()

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    id2label = build_id_map(model)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

    # Scalar guard (single-class edge case)
    if isinstance(probs, float):
        probs = [probs]

    # Build label → prob map, keeping only WARD classes (drop snow, etc.)
    raw_scores: dict[str, float] = {}
    for idx, prob in enumerate(probs):
        label = id2label.get(idx, "").lower().strip()
        if label in WARD_CLASSES:
            raw_scores[label] = float(prob)

    # Renormalise over supported classes
    total = sum(raw_scores.values())
    if total > 0:
        scores = {k: v / total for k, v in raw_scores.items()}
    else:
        scores = {}

    top_label = max(scores, key=scores.get) if scores else "unknown"
    top_confidence = scores.get(top_label, 0.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictionResult(
        model_name=model_name,
        scores=scores,
        top_label=top_label,
        top_confidence=top_confidence,
        latency_ms=latency_ms,
    )


# ── Fusion ──────────────────────────────────────────────────────────────────

def fuse(
    original: PredictionResult,
    fine: PredictionResult,
    original_weight: float = 0.30,
    fine_weight: float = 0.70,
    ambiguity_threshold: float = 0.10,
) -> FusionResult:
    """
    Combine two PredictionResults into a FusionResult.

    Rules
    -----
    dry   = w_orig * orig_dry  + w_fine * fine_dry
    wet   = w_orig * orig_wet  + w_fine * fine_wet
    damp  = fine_damp
    drying = fine_drying

    Missing original classes use 0.0 — no invented values.
    Result is normalised so scores sum to 1.0.
    """
    raw: dict[str, float] = {}

    # dry and wet — both models contribute
    for cls in ("dry", "wet"):
        raw[cls] = (
            original_weight * original.get_score(cls)
            + fine_weight * fine.get_score(cls)
        )

    # damp and drying — fine-tuned model only
    for cls in ("damp", "drying"):
        raw[cls] = fine.get_score(cls)

    # Normalise
    total = sum(raw.values())
    if total > 0:
        fused_scores = {k: v / total for k, v in raw.items()}
    else:
        fused_scores = {k: 0.25 for k in WARD_CLASSES}  # uniform fallback

    sorted_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    top_label, top_confidence = sorted_scores[0]

    margin = (
        sorted_scores[0][1] - sorted_scores[1][1]
        if len(sorted_scores) >= 2
        else 1.0
    )
    ambiguous = margin < ambiguity_threshold

    agreement = original.top_label == fine.top_label

    return FusionResult(
        original=original,
        fine=fine,
        fused_scores=fused_scores,
        top_label=top_label,
        top_confidence=top_confidence,
        agreement=agreement,
        ambiguous=ambiguous,
        original_label=original.top_label,
        fine_label=fine.top_label,
    )
