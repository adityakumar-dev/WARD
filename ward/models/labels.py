"""
WARD — Dynamic Label Handling
==============================
Reads id2label / label2id from each model's config at load-time.
Normalises all labels to lowercase.
"""

from __future__ import annotations

# The four canonical WARD output classes (in display order)
WARD_CLASSES: list[str] = ["dry", "damp", "wet", "drying"]

# Classes that feed into fusion; original model contributes only these two
ORIGINAL_FUSION_CLASSES: set[str] = {"dry", "wet"}


def build_label_map(model) -> dict[str, int]:
    """
    Return a normalised {label: index} map from the model config.
    Labels are lowercased.  Only labels recognised by the model are included.
    """
    raw: dict = getattr(model.config, "label2id", {}) or {}
    return {str(k).lower().strip(): int(v) for k, v in raw.items()}


def build_id_map(model) -> dict[int, str]:
    """
    Return a normalised {index: label} map from the model config.
    """
    raw: dict = getattr(model.config, "id2label", {}) or {}
    return {int(k): str(v).lower().strip() for k, v in raw.items()}


def supported_classes(model) -> list[str]:
    """
    Return the subset of WARD_CLASSES that the model can predict.
    Snow and any other non-WARD labels are excluded.
    """
    known = set(build_label_map(model).keys())
    return [c for c in WARD_CLASSES if c in known]
