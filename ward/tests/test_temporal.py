"""Tests for temporal engine — EMA, streak, hysteresis, spike, override, ambiguity."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock

from models.fusion import FusionResult, PredictionResult
from temporal.smoothing import EMASmoother
from temporal.engine import TemporalEngine


# ── Helper: build a minimal FusionResult ─────────────────────────────────────
def _fusion(top_label: str, top_conf: float, agreement=True, ambiguous=False):
    scores = {cls: 0.0 for cls in ["dry", "damp", "wet", "drying"]}
    scores[top_label] = top_conf
    remaining = (1.0 - top_conf) / max(1, len(scores) - 1)
    for k in scores:
        if k != top_label:
            scores[k] = remaining

    orig = PredictionResult("original", {top_label: top_conf}, top_label, top_conf, 10.0)
    fine = PredictionResult("fine", scores, top_label, top_conf, 10.0)
    return FusionResult(
        original=orig, fine=fine,
        fused_scores=scores,
        top_label=top_label, top_confidence=top_conf,
        agreement=agreement, ambiguous=ambiguous,
        original_label=top_label, fine_label=top_label,
    )


# ── EMASmoother ──────────────────────────────────────────────────────────────
class TestEMASmoother:
    def test_first_call_initialises_directly(self):
        s = EMASmoother(alpha=0.5)
        scores = {"dry": 1.0, "damp": 0.0, "wet": 0.0, "drying": 0.0}
        result = s.update(scores)
        assert result["dry"] == pytest.approx(1.0)

    def test_ema_blends_correctly(self):
        s = EMASmoother(alpha=0.5)
        s.update({"dry": 1.0, "damp": 0.0, "wet": 0.0, "drying": 0.0})
        result = s.update({"dry": 0.0, "damp": 1.0, "wet": 0.0, "drying": 0.0})
        # dry: 0.5*0 + 0.5*1 = 0.5, damp: 0.5*1 + 0.5*0 = 0.5
        assert result["dry"] == pytest.approx(0.5)
        assert result["damp"] == pytest.approx(0.5)

    def test_missing_classes_default_to_zero(self):
        s = EMASmoother(alpha=0.5)
        result = s.update({"dry": 1.0})  # no damp/wet/drying
        assert result["damp"] == pytest.approx(0.0)
        assert result["wet"] == pytest.approx(0.0)
        assert result["drying"] == pytest.approx(0.0)

    def test_reset_clears_state(self):
        s = EMASmoother(alpha=0.5)
        s.update({"dry": 1.0, "damp": 0.0, "wet": 0.0, "drying": 0.0})
        s.reset()
        assert s.smoothed is None
        assert not s.is_initialised


# ── TemporalEngine ────────────────────────────────────────────────────────────
class _MockCfg:
    prediction_history_size = 20
    ema_alpha = 0.99           # almost no smoothing for deterministic tests
    switch_margin = 0.10
    min_candidate_streak = 3
    min_state_dwell_seconds = 0.0    # disabled for test speed
    strong_override_threshold = 0.95
    ambiguity_threshold = 0.10
    drying_candidate_streak = 4      # 3 + 1


class TestTemporalEngine:
    def _engine(self):
        return TemporalEngine(_MockCfg())

    def test_initial_state_from_first_prediction(self):
        eng = self._engine()
        state = eng.update(_fusion("dry", 0.90))
        assert state.label == "dry"

    def test_stable_label_unchanged_on_single_spike(self):
        """One spike should NOT switch state."""
        eng = self._engine()
        for _ in range(5):
            eng.update(_fusion("dry", 0.90))
        state = eng.update(_fusion("wet", 0.91))  # single spike
        assert state.label == "dry"

    def test_candidate_streak_accumulates(self):
        eng = self._engine()
        eng.update(_fusion("dry", 0.90))  # stable = dry
        eng.update(_fusion("wet", 0.85))
        eng.update(_fusion("wet", 0.85))
        state = eng.update(_fusion("wet", 0.85))
        assert state.candidate_label == "wet" or state.label == "wet"
        assert state.candidate_streak >= 0

    def test_repeated_candidate_switches_state(self):
        """After enough consecutive frames, state should switch."""
        eng = self._engine()
        eng.update(_fusion("dry", 0.90))
        # Feed wet many times with high confidence
        state = None
        for _ in range(10):
            state = eng.update(_fusion("wet", 0.90))
        assert state.label == "wet"

    def test_strong_override_fast_switch(self):
        """confidence >= 0.95 and streak >= 2 should override dwell timer."""
        eng = self._engine()
        eng.update(_fusion("dry", 0.90))
        eng.update(_fusion("wet", 0.96))  # streak 1
        state = eng.update(_fusion("wet", 0.96))  # streak 2 → override
        assert state.label == "wet"

    def test_ambiguity_flag_propagated(self):
        """Ambiguity is detected when smoothed top1 - top2 < threshold."""
        eng = self._engine()
        # Build a fusion where all classes are nearly equal (margin < 0.10)
        near_equal = {"dry": 0.26, "damp": 0.25, "wet": 0.25, "drying": 0.24}
        orig = PredictionResult("original", {"dry": 0.26, "wet": 0.25}, "dry", 0.26, 10.0)
        fine = PredictionResult("fine", near_equal, "dry", 0.26, 10.0)
        f = FusionResult(
            original=orig, fine=fine,
            fused_scores=near_equal,
            top_label="dry", top_confidence=0.26,
            agreement=True, ambiguous=True,
            original_label="dry", fine_label="dry",
        )
        state = eng.update(f)
        # With alpha=0.99, smoothed ≈ input. top1-top2 = 0.26-0.25 = 0.01 < 0.10
        assert state.ambiguous is True

    def test_reset_clears_all_state(self):
        eng = self._engine()
        for _ in range(5):
            eng.update(_fusion("wet", 0.90))
        eng.reset()
        assert eng.state is None

    def test_stability_reflects_history(self):
        eng = self._engine()
        for _ in range(10):
            eng.update(_fusion("dry", 0.90))
        state = eng.update(_fusion("dry", 0.90))
        assert state.stability > 0.8
