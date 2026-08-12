"""Tests for inference providers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image

from inference.provider import InferenceUnavailableError
from models.fusion import FusionResult, PredictionResult


def _rgb_image():
    return Image.new("RGB", (32, 32), color=(100, 120, 140))


# ── LocalInferenceProvider ────────────────────────────────────────────────────
class TestLocalProvider:
    def test_returns_fusion_result(self):
        """LocalInferenceProvider.predict must return a FusionResult."""
        from inference.local_provider import LocalInferenceProvider
        from models.loader import ModelPair

        # Mock model pair
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "dry", 1: "wet"}
        mock_model.config.label2id = {"dry": 0, "wet": 1}
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": MagicMock()}

        import torch
        mock_model.return_value = MagicMock(logits=torch.tensor([[0.8, 0.2]]))

        pair = ModelPair(
            original_model=mock_model,
            original_proc=mock_proc,
            fine_model=mock_model,
            fine_proc=mock_proc,
            device="cpu",
        )

        cfg = MagicMock()
        cfg.original_weight = 0.30
        cfg.fine_weight = 0.70
        cfg.ambiguity_threshold = 0.10

        provider = LocalInferenceProvider(pair, cfg)
        result = provider.predict(_rgb_image())
        assert isinstance(result, FusionResult)
        assert result.top_label in ["dry", "wet"]

    def test_raises_on_model_exception(self):
        from inference.local_provider import LocalInferenceProvider
        from models.loader import ModelPair

        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("GPU OOM")
        mock_proc = MagicMock()

        pair = ModelPair(
            original_model=mock_model,
            original_proc=mock_proc,
            fine_model=mock_model,
            fine_proc=mock_proc,
            device="cpu",
        )
        cfg = MagicMock()
        cfg.original_weight = 0.30
        cfg.fine_weight = 0.70
        cfg.ambiguity_threshold = 0.10

        provider = LocalInferenceProvider(pair, cfg)
        with pytest.raises(InferenceUnavailableError):
            provider.predict(_rgb_image())


# ── HFInferenceProvider ───────────────────────────────────────────────────────
class TestHFProvider:
    def _cfg(self):
        cfg = MagicMock()
        cfg.hf_token = "test_token"
        cfg.hf_original_endpoint = "http://orig.example.com"
        cfg.hf_fine_endpoint = "http://fine.example.com"
        cfg.original_weight = 0.30
        cfg.fine_weight = 0.70
        cfg.ambiguity_threshold = 0.10
        return cfg

    def test_raises_without_endpoints(self):
        from inference.hf_provider import HFInferenceProvider
        cfg = MagicMock()
        cfg.hf_original_endpoint = ""
        cfg.hf_fine_endpoint = ""
        with pytest.raises(ValueError):
            HFInferenceProvider(cfg)

    def test_returns_fusion_on_success(self):
        from inference.hf_provider import HFInferenceProvider
        import requests

        orig_response = [{"label": "dry", "score": 0.85}, {"label": "wet", "score": 0.15}]
        fine_response = [{"label": "dry", "score": 0.70}, {"label": "damp", "score": 0.15},
                         {"label": "wet", "score": 0.10}, {"label": "drying", "score": 0.05}]

        def mock_post(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            if "orig" in url:
                mock_resp.json.return_value = orig_response
            else:
                mock_resp.json.return_value = fine_response
            return mock_resp

        with patch("inference.hf_provider.requests.post", side_effect=mock_post):
            provider = HFInferenceProvider(self._cfg())
            result = provider.predict(_rgb_image())

        assert isinstance(result, FusionResult)
        assert result.top_label in ["dry", "damp", "wet", "drying"]

    def test_timeout_raises_unavailable(self):
        from inference.hf_provider import HFInferenceProvider
        import requests

        with patch("inference.hf_provider.requests.post",
                   side_effect=requests.exceptions.Timeout("timed out")):
            provider = HFInferenceProvider(self._cfg())
            with pytest.raises(InferenceUnavailableError):
                provider.predict(_rgb_image())

    def test_http_error_raises_unavailable(self):
        from inference.hf_provider import HFInferenceProvider
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401")
        mock_resp.status_code = 401

        with patch("inference.hf_provider.requests.post", return_value=mock_resp):
            provider = HFInferenceProvider(self._cfg())
            with pytest.raises(InferenceUnavailableError):
                provider.predict(_rgb_image())
