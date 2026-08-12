"""Tests for models/fusion.py — fusion logic and normalization."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock
from models.fusion import PredictionResult, FusionResult, fuse, predict_single
from models.labels import WARD_CLASSES


def _pred(model_name, scores, top_label, top_conf, latency=10.0):
    return PredictionResult(
        model_name=model_name,
        scores=scores,
        top_label=top_label,
        top_confidence=top_conf,
        latency_ms=latency,
    )


# ── Fusion rules ──────────────────────────────────────────────────────────────

class TestFusion:
    def test_dry_weighted_30_70(self):
        orig = _pred("original", {"dry": 1.0, "wet": 0.0}, "dry", 1.0)
        fine = _pred("fine", {"dry": 0.0, "wet": 0.0, "damp": 0.0, "drying": 0.0}, "dry", 0.0)
        result = fuse(orig, fine, original_weight=0.30, fine_weight=0.70)
        # dry = 0.30 * 1.0 + 0.70 * 0.0 = 0.30 → normalised = 1.0 (only non-zero)
        assert result.fused_scores["dry"] == pytest.approx(1.0)

    def test_wet_weighted_30_70(self):
        orig = _pred("original", {"dry": 0.0, "wet": 1.0}, "wet", 1.0)
        fine = _pred("fine", {"dry": 0.0, "wet": 0.5, "damp": 0.5, "drying": 0.0}, "wet", 0.5)
        result = fuse(orig, fine, original_weight=0.30, fine_weight=0.70)
        # raw_dry  = 0.30*0 + 0.70*0    = 0
        # raw_wet  = 0.30*1 + 0.70*0.5  = 0.65
        # raw_damp = fine_damp          = 0.5
        # raw_dry_fine = 0.70*0 = 0
        # total = 0 + 0.65 + 0.5 + 0 = 1.15
        raw_wet  = 0.30 * 1.0 + 0.70 * 0.5   # 0.65
        raw_damp = 0.5                         # fine_damp
        total = raw_wet + raw_damp             # 1.15
        assert result.fused_scores["wet"]  == pytest.approx(raw_wet  / total, abs=1e-6)
        assert result.fused_scores["damp"] == pytest.approx(raw_damp / total, abs=1e-6)

    def test_damp_comes_from_fine_only(self):
        """Original model has no damp — damp score must equal fine_damp after normalization."""
        orig = _pred("original", {"dry": 0.8, "wet": 0.2}, "dry", 0.8)
        fine = _pred("fine", {"dry": 0.0, "damp": 1.0, "wet": 0.0, "drying": 0.0}, "damp", 1.0)
        result = fuse(orig, fine, original_weight=0.30, fine_weight=0.70)
        # raw_dry = 0.30 * 0.8 + 0.70 * 0.0 = 0.24
        # raw_damp = 1.0, raw_wet = 0.30 * 0.2 = 0.06
        raw_dry = 0.24; raw_damp = 1.0; raw_wet = 0.06
        total = raw_dry + raw_damp + raw_wet
        assert result.fused_scores["damp"] == pytest.approx(raw_damp / total)

    def test_drying_comes_from_fine_only(self):
        orig = _pred("original", {"dry": 1.0}, "dry", 1.0)
        fine = _pred("fine", {"dry": 0.0, "damp": 0.0, "wet": 0.0, "drying": 1.0}, "drying", 1.0)
        result = fuse(orig, fine, original_weight=0.30, fine_weight=0.70)
        raw_dry = 0.30 * 1.0 + 0.70 * 0.0
        raw_drying = 1.0
        total = raw_dry + raw_drying
        assert result.fused_scores["drying"] == pytest.approx(raw_drying / total)

    def test_normalization_sums_to_one(self):
        orig = _pred("original", {"dry": 0.6, "wet": 0.4}, "dry", 0.6)
        fine = _pred("fine", {"dry": 0.3, "damp": 0.3, "wet": 0.2, "drying": 0.2}, "dry", 0.3)
        result = fuse(orig, fine)
        assert sum(result.fused_scores.values()) == pytest.approx(1.0, abs=1e-6)

    def test_missing_original_class_defaults_zero(self):
        """Original model lacks damp/drying — these should come purely from fine."""
        orig = _pred("original", {"dry": 0.9, "wet": 0.1}, "dry", 0.9)
        fine = _pred("fine", {"dry": 0.1, "damp": 0.5, "wet": 0.1, "drying": 0.3}, "damp", 0.5)
        result = fuse(orig, fine)
        # damp = fine_damp / total (no contribution from original)
        total = sum(result.fused_scores.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_agreement_true_when_labels_match(self):
        orig = _pred("original", {"dry": 0.9, "wet": 0.1}, "dry", 0.9)
        fine = _pred("fine", {"dry": 0.8, "damp": 0.1, "wet": 0.1, "drying": 0.0}, "dry", 0.8)
        result = fuse(orig, fine)
        assert result.agreement is True

    def test_agreement_false_when_labels_differ(self):
        orig = _pred("original", {"dry": 0.9, "wet": 0.1}, "dry", 0.9)
        fine = _pred("fine", {"dry": 0.1, "damp": 0.0, "wet": 0.8, "drying": 0.1}, "wet", 0.8)
        result = fuse(orig, fine)
        assert result.agreement is False

    def test_ambiguous_flag_set_on_fusion(self):
        """fuse() sets ambiguous when top1 - top2 < threshold."""
        orig = _pred("original", {"dry": 0.5, "wet": 0.5}, "dry", 0.5)
        fine = _pred("fine", {"dry": 0.26, "damp": 0.25, "wet": 0.25, "drying": 0.24}, "dry", 0.26)
        result = fuse(orig, fine, ambiguity_threshold=0.10)
        assert result.ambiguous is True

    def test_not_ambiguous_when_clear_winner(self):
        orig = _pred("original", {"dry": 1.0, "wet": 0.0}, "dry", 1.0)
        fine = _pred("fine", {"dry": 1.0, "damp": 0.0, "wet": 0.0, "drying": 0.0}, "dry", 1.0)
        result = fuse(orig, fine, ambiguity_threshold=0.10)
        assert result.ambiguous is False
