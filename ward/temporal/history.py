"""
WARD — Temporal Prediction History
====================================
A bounded deque of FusionResult objects.
Used by the smoothing and engine modules to access recent predictions.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from models.fusion import FusionResult
from models.labels import WARD_CLASSES


class PredictionHistory:
    """Bounded rolling window of FusionResult objects."""

    def __init__(self, maxlen: int = 20) -> None:
        self._deque: Deque[FusionResult] = deque(maxlen=maxlen)

    def push(self, result: FusionResult) -> None:
        self._deque.append(result)

    def clear(self) -> None:
        self._deque.clear()

    def __len__(self) -> int:
        return len(self._deque)

    def is_empty(self) -> bool:
        return len(self._deque) == 0

    def recent(self, n: Optional[int] = None) -> List[FusionResult]:
        """Return the last n results (all if n is None)."""
        items = list(self._deque)
        if n is None:
            return items
        return items[-n:]

    def recent_scores(self, n: Optional[int] = None) -> List[dict[str, float]]:
        """Return a list of fused_scores dicts for the last n results."""
        return [r.fused_scores for r in self.recent(n)]

    def stability(self, label: str, n: int = 10) -> float:
        """
        Fraction of the last n predictions whose top label is `label`.
        Returns 0.0 if history is empty.
        """
        recent = self.recent(n)
        if not recent:
            return 0.0
        matches = sum(1 for r in recent if r.top_label == label)
        return matches / len(recent)
