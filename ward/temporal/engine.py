"""
WARD — Temporal State Machine
==============================
Prevents frame-to-frame flicker by requiring candidate persistence
before committing a stable state change.

Rules
-----
Normal switch:
    candidate_streak >= MIN_CANDIDATE_STREAK
    AND fused top_confidence - stable confidence >= SWITCH_MARGIN
    AND time since last switch >= MIN_STATE_DWELL_SECONDS

Fast-path override (strong evidence):
    top_confidence >= STRONG_OVERRIDE_THRESHOLD
    AND candidate_streak >= 2
    → switches immediately, bypassing dwell timer

Drying special treatment:
    requires drying_candidate_streak (= MIN + 1) before promotion

One-frame spike protection:
    A single outlier frame resets candidate_streak to 1 (or 0 if
    the previous candidate was different), never triggering a switch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from models.fusion import FusionResult
from models.labels import WARD_CLASSES
from temporal.history import PredictionHistory
from temporal.smoothing import EMASmoother


@dataclass
class TemporalState:
    label: str                         # current stable label
    confidence: float                  # smoothed top-class score
    stability: float                   # fraction of history agreeing
    smoothed_scores: dict[str, float]  # all WARD classes, EMA-smoothed
    candidate_label: Optional[str]     # pending challenger label
    candidate_streak: int              # consecutive frames for candidate
    ambiguous: bool                    # top1 - top2 < threshold
    last_switch_time: float = field(default_factory=time.time)


_DRYING_EXTRA_STREAK = 1   # drying requires streak + this many extra frames


class TemporalEngine:
    """
    Maintains stable road-condition state across a sequence of frames.

    Parameters
    ----------
    cfg : Config object (uses ema_alpha, switch_margin, min_candidate_streak,
          min_state_dwell_seconds, strong_override_threshold, ambiguity_threshold,
          prediction_history_size)
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._history = PredictionHistory(maxlen=cfg.prediction_history_size)
        self._smoother = EMASmoother(alpha=cfg.ema_alpha)
        self._state: Optional[TemporalState] = None
        self._candidate_label: Optional[str] = None
        self._candidate_streak: int = 0
        self._last_switch_time: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._history.clear()
        self._smoother.reset()
        self._state = None
        self._candidate_label = None
        self._candidate_streak = 0
        self._last_switch_time = 0.0

    @property
    def state(self) -> Optional[TemporalState]:
        return self._state

    def update(self, fusion: FusionResult) -> TemporalState:
        """
        Feed a new FusionResult into the engine and return the updated
        TemporalState.
        """
        self._history.push(fusion)
        smoothed = self._smoother.update(fusion.fused_scores)

        # Determine current top from smoothed scores
        sorted_smoothed = sorted(smoothed.items(), key=lambda x: x[1], reverse=True)
        top_label, top_conf = sorted_smoothed[0]
        second_conf = sorted_smoothed[1][1] if len(sorted_smoothed) >= 2 else 0.0
        ambiguous = (top_conf - second_conf) < self._cfg.ambiguity_threshold

        # ── Initialise on first call ────────────────────────────────────────
        if self._state is None:
            self._state = TemporalState(
                label=top_label,
                confidence=top_conf,
                stability=1.0,
                smoothed_scores=smoothed,
                candidate_label=None,
                candidate_streak=0,
                ambiguous=ambiguous,
                last_switch_time=time.time(),
            )
            self._last_switch_time = time.time()
            return self._state

        stable_label = self._state.label
        now = time.time()

        # ── Candidate tracking ──────────────────────────────────────────────
        if top_label == stable_label:
            # Prediction agrees with stable — reset candidate
            self._candidate_label = None
            self._candidate_streak = 0
        else:
            if top_label == self._candidate_label:
                self._candidate_streak += 1
            else:
                # New challenger — start fresh streak
                self._candidate_label = top_label
                self._candidate_streak = 1

        # ── Switch decision ─────────────────────────────────────────────────
        switched = False

        if self._candidate_label is not None:
            streak_needed = self._cfg.min_candidate_streak
            if self._candidate_label == "drying":
                streak_needed += _DRYING_EXTRA_STREAK

            stable_conf = smoothed.get(stable_label, 0.0)
            margin_ok = (top_conf - stable_conf) >= self._cfg.switch_margin
            dwell_ok = (now - self._last_switch_time) >= self._cfg.min_state_dwell_seconds
            streak_ok = self._candidate_streak >= streak_needed

            # Fast-path override: strong evidence bypasses dwell timer
            fast_override = (
                top_conf >= self._cfg.strong_override_threshold
                and self._candidate_streak >= 2
            )

            if fast_override or (streak_ok and margin_ok and dwell_ok):
                stable_label = self._candidate_label
                self._candidate_label = None
                self._candidate_streak = 0
                self._last_switch_time = now
                switched = True

        # ── Stability metric ────────────────────────────────────────────────
        stability = self._history.stability(stable_label, n=min(10, len(self._history)))

        self._state = TemporalState(
            label=stable_label,
            confidence=smoothed.get(stable_label, top_conf),
            stability=stability,
            smoothed_scores=smoothed,
            candidate_label=self._candidate_label,
            candidate_streak=self._candidate_streak,
            ambiguous=ambiguous,
            last_switch_time=self._last_switch_time,
        )
        return self._state
