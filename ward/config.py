"""
WARD — Centralised Configuration
=================================
All tuneable parameters live here.  Nothing else should import os.getenv
for WARD settings directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if it exists (silently ignored in PROD/container environments)
load_dotenv(Path(__file__).parent / ".env", override=False)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # ── Mode ──────────────────────────────────────────────────────────────────
    ward_mode: str = field(default_factory=lambda: _env("WARD_MODE", "dev").lower())

    # ── Model IDs ─────────────────────────────────────────────────────────────
    original_model_id: str = field(
        default_factory=lambda: _env(
            "ORIGINAL_MODEL_ID", "adityakumarxdev/ward-siglip"
        )
    )
    fine_model_id: str = field(
        default_factory=lambda: _env(
            "FINE_MODEL_ID", "adityakumarxdev/ward-siglip-lora-2"
        )
    )

    # ── Fusion Weights ────────────────────────────────────────────────────────
    original_weight: float = field(
        default_factory=lambda: _env_float("ORIGINAL_WEIGHT", 0.30)
    )
    fine_weight: float = field(
        default_factory=lambda: _env_float("FINE_WEIGHT", 0.70)
    )

    # ── Frame Processing ──────────────────────────────────────────────────────
    inference_fps: int = field(
        default_factory=lambda: _env_int("INFERENCE_FPS", 5)
    )
    frame_queue_size: int = field(
        default_factory=lambda: _env_int("FRAME_QUEUE_SIZE", 2)
    )

    # ── Temporal Engine ───────────────────────────────────────────────────────
    prediction_history_size: int = field(
        default_factory=lambda: _env_int("PREDICTION_HISTORY_SIZE", 20)
    )
    ema_alpha: float = field(
        default_factory=lambda: _env_float("EMA_ALPHA", 0.40)
    )
    switch_margin: float = field(
        default_factory=lambda: _env_float("SWITCH_MARGIN", 0.10)
    )
    min_candidate_streak: int = field(
        default_factory=lambda: _env_int("MIN_CANDIDATE_STREAK", 3)
    )
    min_state_dwell_seconds: float = field(
        default_factory=lambda: _env_float("MIN_STATE_DWELL_SECONDS", 2.0)
    )
    strong_override_threshold: float = field(
        default_factory=lambda: _env_float("STRONG_OVERRIDE_THRESHOLD", 0.90)
    )
    ambiguity_threshold: float = field(
        default_factory=lambda: _env_float("AMBIGUITY_THRESHOLD", 0.10)
    )

    # ── Weather ───────────────────────────────────────────────────────────────
    weather_refresh_seconds: int = field(
        default_factory=lambda: _env_int("WEATHER_REFRESH_SECONDS", 300)
    )

    # ── HF Production Endpoints ───────────────────────────────────────────────
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))
    hf_original_endpoint: str = field(
        default_factory=lambda: _env("HF_ORIGINAL_ENDPOINT", "")
    )
    hf_fine_endpoint: str = field(
        default_factory=lambda: _env("HF_FINE_ENDPOINT", "")
    )

    # ── Computed helpers ──────────────────────────────────────────────────────
    @property
    def is_dev(self) -> bool:
        return self.ward_mode == "dev"

    @property
    def is_prod(self) -> bool:
        return self.ward_mode == "prod"

    # Drying requires one extra candidate frame for stability
    @property
    def drying_candidate_streak(self) -> int:
        return self.min_candidate_streak + 1


# Module-level singleton so callers can just `from config import cfg`
cfg = Config()
