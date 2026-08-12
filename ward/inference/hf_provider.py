"""
WARD — HF Inference Endpoint Provider (PROD mode)
==================================================
Sends images to two Hugging Face Inference Endpoints and applies the
same fusion logic from models/fusion.py.

Expected HF endpoint response format
-------------------------------------
[
    {"label": "wet", "score": 0.82},
    {"label": "dry", "score": 0.12},
    ...
]
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import requests
from PIL import Image

from inference.provider import InferenceProvider, InferenceUnavailableError
from models.fusion import FusionResult, PredictionResult, fuse
from models.labels import WARD_CLASSES

logger = logging.getLogger(__name__)

_HF_TIMEOUT_SECONDS = 10
_HF_RETRY_BACKOFF = 0.5


class HFInferenceProvider(InferenceProvider):
    """PROD-mode inference: calls two HF Inference Endpoints."""

    def __init__(self, cfg) -> None:
        if not cfg.hf_original_endpoint or not cfg.hf_fine_endpoint:
            raise ValueError(
                "HF_ORIGINAL_ENDPOINT and HF_FINE_ENDPOINT must be set in PROD mode."
            )
        self._original_url = cfg.hf_original_endpoint
        self._fine_url = cfg.hf_fine_endpoint
        self._token = cfg.hf_token
        self._cfg = cfg

    @property
    def backend_name(self) -> str:
        return "HF INFERENCE ENDPOINT"

    # ── Internal helpers ────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "image/jpeg"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _call_endpoint(self, url: str, image_bytes: bytes, model_name: str) -> list[dict]:
        """POST to one HF endpoint with one retry on transient failure."""
        for attempt in range(2):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    data=image_bytes,
                    timeout=_HF_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                data = resp.json()
                if not isinstance(data, list):
                    raise ValueError(f"Unexpected HF response format: {type(data)}")
                return data, latency_ms
            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning("%s endpoint timeout — retrying", model_name)
                    time.sleep(_HF_RETRY_BACKOFF)
                    continue
                raise InferenceUnavailableError(
                    f"{model_name} endpoint timed out after {_HF_TIMEOUT_SECONDS}s"
                )
            except requests.exceptions.ConnectionError as exc:
                raise InferenceUnavailableError(
                    f"{model_name} endpoint connection failed: {exc}"
                ) from exc
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                raise InferenceUnavailableError(
                    f"{model_name} endpoint returned HTTP {status}"
                ) from exc
            except Exception as exc:
                raise InferenceUnavailableError(
                    f"{model_name} endpoint error: {exc}"
                ) from exc
        raise InferenceUnavailableError(f"{model_name} endpoint unavailable")

    def _parse_response(
        self, data: list[dict], latency_ms: float, model_name: str
    ) -> PredictionResult:
        """Convert HF response list into a PredictionResult."""
        scores: dict[str, float] = {}
        for item in data:
            label = str(item.get("label", "")).lower().strip()
            score = float(item.get("score", 0.0))
            if label in WARD_CLASSES:
                scores[label] = score

        # Renormalise
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        top_label = max(scores, key=scores.get) if scores else "unknown"
        top_confidence = scores.get(top_label, 0.0)

        return PredictionResult(
            model_name=model_name,
            scores=scores,
            top_label=top_label,
            top_confidence=top_confidence,
            latency_ms=latency_ms,
        )

    # ── Public interface ────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> FusionResult:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_bytes = self._image_to_bytes(image)

            orig_data, orig_latency = self._call_endpoint(
                self._original_url, image_bytes, "original"
            )
            fine_data, fine_latency = self._call_endpoint(
                self._fine_url, image_bytes, "fine_tuned"
            )

            original_result = self._parse_response(orig_data, orig_latency, "original")
            fine_result = self._parse_response(fine_data, fine_latency, "fine_tuned")

            return fuse(
                original_result,
                fine_result,
                original_weight=self._cfg.original_weight,
                fine_weight=self._cfg.fine_weight,
                ambiguity_threshold=self._cfg.ambiguity_threshold,
            )
        except InferenceUnavailableError:
            raise
        except Exception as exc:
            logger.exception("HFInferenceProvider.predict failed")
            raise InferenceUnavailableError(str(exc)) from exc
