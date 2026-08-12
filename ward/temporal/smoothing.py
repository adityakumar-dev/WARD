"""
WARD — EMA Probability Smoother
=================================
Applies exponential moving average smoothing to per-class scores.
Reduces frame-to-frame prediction noise.

smoothed[cls] = alpha * current[cls] + (1 - alpha) * previous[cls]
"""

from __future__ import annotations

from typing import Dict, Optional

from models.labels import WARD_CLASSES


class EMASmoother:
    """
    Per-class EMA over WARD_CLASSES probability scores.

    Parameters
    ----------
    alpha : smoothing factor  (0.0 = no update, 1.0 = no smoothing)
    """

    def __init__(self, alpha: float = 0.40) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._smoothed: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        self._smoothed = None

    @property
    def is_initialised(self) -> bool:
        return self._smoothed is not None

    def update(self, current: Dict[str, float]) -> Dict[str, float]:
        """
        Blend current scores with previous smoothed scores.
        First call initialises from current directly (no blending).

        Parameters
        ----------
        current : dict of {class: probability}, may be partial

        Returns
        -------
        Smoothed scores dict (all WARD_CLASSES present).
        """
        # Fill missing classes with 0.0
        full_current: Dict[str, float] = {
            cls: current.get(cls, 0.0) for cls in WARD_CLASSES
        }

        if self._smoothed is None:
            self._smoothed = dict(full_current)
        else:
            self._smoothed = {
                cls: self.alpha * full_current[cls] + (1.0 - self.alpha) * self._smoothed[cls]
                for cls in WARD_CLASSES
            }

        return dict(self._smoothed)

    @property
    def smoothed(self) -> Optional[Dict[str, float]]:
        return dict(self._smoothed) if self._smoothed else None
