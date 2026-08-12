"""
WARD — Session State
=====================
Manages per-session analysis state.
Tracks counters, lifecycle, and per-frame history for CSV export and
the session summary.

Lifecycle
---------
IDLE → READY → ANALYZING → (PAUSED → ANALYZING) → COMPLETE
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from models.fusion import FusionResult
from temporal.engine import TemporalState


class SessionStatus(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    ANALYZING = "ANALYZING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"


@dataclass
class FrameRecord:
    """One row of the CSV export."""
    frame_index: int
    timestamp_s: float
    filename: Optional[str]           # None for video frames
    original_label: str
    original_confidence: float
    original_scores: dict[str, float]
    fine_label: str
    fine_confidence: float
    fine_scores: dict[str, float]
    fused_label: str
    fused_confidence: float
    fused_scores: dict[str, float]
    stable_label: str
    stable_confidence: float
    stability: float
    agreement: bool
    ambiguous: bool
    original_latency_ms: float
    fine_latency_ms: float


class SessionState:
    """
    Tracks the current analysis session.
    One instance lives in st.session_state['ward_session'].
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.status = SessionStatus.IDLE
        self.input_name: Optional[str] = None
        self.input_type: Optional[str] = None   # "image", "video", "frames"

        # Frame counters
        self.frames_read: int = 0
        self.frames_processed: int = 0
        self.frames_dropped: int = 0

        # Timing
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None

        # Current frame data
        self.current_fusion: Optional[FusionResult] = None
        self.current_temporal: Optional[TemporalState] = None

        # History for export and timeline
        self.records: List[FrameRecord] = []

        # Latency tracking
        self._latency_samples: List[float] = []

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def start(self, input_name: str, input_type: str) -> None:
        self.reset()
        self.input_name = input_name
        self.input_type = input_type
        self.status = SessionStatus.ANALYZING
        self.started_at = time.time()

    def pause(self) -> None:
        if self.status == SessionStatus.ANALYZING:
            self.status = SessionStatus.PAUSED

    def resume(self) -> None:
        if self.status == SessionStatus.PAUSED:
            self.status = SessionStatus.ANALYZING

    def complete(self) -> None:
        self.status = SessionStatus.COMPLETE
        self.completed_at = time.time()

    def stop(self) -> None:
        self.complete()

    # ── Frame recording ─────────────────────────────────────────────────────

    def record_frame(
        self,
        fusion: FusionResult,
        temporal: TemporalState,
        timestamp_s: float = 0.0,
        filename: Optional[str] = None,
    ) -> None:
        self.current_fusion = fusion
        self.current_temporal = temporal
        self.frames_processed += 1

        orig_latency = fusion.original.latency_ms
        fine_latency = fusion.fine.latency_ms
        total_latency = orig_latency + fine_latency
        self._latency_samples.append(total_latency)

        record = FrameRecord(
            frame_index=self.frames_processed - 1,
            timestamp_s=timestamp_s,
            filename=filename,
            original_label=fusion.original_label,
            original_confidence=fusion.original.top_confidence,
            original_scores=dict(fusion.original.scores),
            fine_label=fusion.fine_label,
            fine_confidence=fusion.fine.top_confidence,
            fine_scores=dict(fusion.fine.scores),
            fused_label=fusion.top_label,
            fused_confidence=fusion.top_confidence,
            fused_scores=dict(fusion.fused_scores),
            stable_label=temporal.label,
            stable_confidence=temporal.confidence,
            stability=temporal.stability,
            agreement=fusion.agreement,
            ambiguous=fusion.ambiguous,
            original_latency_ms=orig_latency,
            fine_latency_ms=fine_latency,
        )
        self.records.append(record)

    # ── Summary metrics ─────────────────────────────────────────────────────

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    @property
    def avg_confidence(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.fused_confidence for r in self.records) / len(self.records)

    @property
    def avg_stability(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.stability for r in self.records) / len(self.records)

    @property
    def model_agreement_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.agreement) / len(self.records)

    @property
    def dominant_condition(self) -> Optional[str]:
        if not self.records:
            return None
        from collections import Counter
        counts = Counter(r.stable_label for r in self.records)
        return counts.most_common(1)[0][0]

    @property
    def peak_confidence(self) -> float:
        if not self.records:
            return 0.0
        return max(r.fused_confidence for r in self.records)

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_csv_lines(self) -> List[str]:
        """Return CSV string lines for the session records."""
        header = (
            "frame_index,timestamp_s,filename,"
            "original_label,original_confidence,"
            "fine_label,fine_confidence,"
            "fused_label,fused_confidence,"
            "stable_label,stable_confidence,stability,"
            "agreement,ambiguous,"
            "original_latency_ms,fine_latency_ms"
        )
        lines = [header]
        for r in self.records:
            lines.append(
                f"{r.frame_index},{r.timestamp_s:.3f},{r.filename or ''},"
                f"{r.original_label},{r.original_confidence:.4f},"
                f"{r.fine_label},{r.fine_confidence:.4f},"
                f"{r.fused_label},{r.fused_confidence:.4f},"
                f"{r.stable_label},{r.stable_confidence:.4f},{r.stability:.4f},"
                f"{int(r.agreement)},{int(r.ambiguous)},"
                f"{r.original_latency_ms:.1f},{r.fine_latency_ms:.1f}"
            )
        return lines
