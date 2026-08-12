"""
WARD — In-Memory Event Log
===========================
Stores notable events during a session.
All events are cleared on session reset.
This is NOT a database.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class EventType(str, Enum):
    CONDITION_CHANGED = "CONDITION_CHANGED"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    PREDICTION_STABILIZED = "PREDICTION_STABILIZED"
    AMBIGUOUS = "AMBIGUOUS"
    ERROR = "ERROR"
    INFO = "INFO"


@dataclass
class Event:
    timestamp: float
    event_type: EventType
    message: str
    details: dict = field(default_factory=dict)

    @property
    def time_str(self) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")


class EventLog:
    """In-memory event log, cleared on session reset."""

    _HIGH_CONFIDENCE_THRESHOLD = 0.90

    def __init__(self, max_events: int = 200) -> None:
        self._events: List[Event] = []
        self._max = max_events
        self._last_stable_label: Optional[str] = None
        self._last_disagreement_logged: float = 0.0
        self._last_ambiguous_logged: float = 0.0

    def clear(self) -> None:
        self._events.clear()
        self._last_stable_label = None
        self._last_ambiguous_logged = 0.0

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def add(self, event_type: EventType, message: str, details: dict = None) -> None:
        evt = Event(
            timestamp=time.time(),
            event_type=event_type,
            message=message,
            details=details or {},
        )
        self._events.append(evt)
        if len(self._events) > self._max:
            self._events.pop(0)

    # ── Convenience helpers called from the main loop ───────────────────────

    def process_fusion(self, fusion, temporal_label: str) -> None:
        """Automatically generate events from a FusionResult + stable label."""
        from models.fusion import FusionResult

        # Condition changed
        if (
            self._last_stable_label is not None
            and temporal_label != self._last_stable_label
        ):
            self.add(
                EventType.CONDITION_CHANGED,
                f"Condition changed: {self._last_stable_label.upper()} → {temporal_label.upper()}",
                {
                    "from": self._last_stable_label,
                    "to": temporal_label,
                },
            )

        self._last_stable_label = temporal_label

        # High confidence wet
        if fusion.top_label == "wet" and fusion.top_confidence >= self._HIGH_CONFIDENCE_THRESHOLD:
            self.add(
                EventType.HIGH_CONFIDENCE,
                f"High confidence WET detected: {fusion.top_confidence:.0%}",
                {"confidence": fusion.top_confidence},
            )

        # Model disagreement (rate-limited to once per 5 s)
        now = time.time()
        if not fusion.agreement and (now - self._last_disagreement_logged) > 5.0:
            self.add(
                EventType.MODEL_DISAGREEMENT,
                f"Models disagree: Original={fusion.original_label.upper()}, Fine={fusion.fine_label.upper()}",
                {
                    "original": fusion.original_label,
                    "fine": fusion.fine_label,
                    "fused": fusion.top_label,
                },
            )
            self._last_disagreement_logged = now

        # Ambiguous prediction (rate-limited to once per 5 s)
        if fusion.ambiguous and (now - self._last_ambiguous_logged) > 5.0:
            self.add(
                EventType.AMBIGUOUS,
                f"Ambiguous prediction — top margin < threshold",
                {"scores": fusion.fused_scores},
            )
            self._last_ambiguous_logged = now
