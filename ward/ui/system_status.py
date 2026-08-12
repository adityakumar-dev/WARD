"""
WARD — System Status Panel
============================
Renders technical metrics: mode, backend, GPU, FPS, latency, queue,
frame counters, model status.

All HTML uses inline styles only — no CSS class dependencies.
This ensures correct rendering inside st.dialog (isolated React portal scope).
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from session.state import SessionState, SessionStatus

# ── Shared inline style fragments ──────────────────────────────────────────
_MONO   = "font-family:'JetBrains Mono',monospace"
_SEC    = (
    f"{_MONO};font-size:0.62rem;font-weight:600;color:#475569;"
    "letter-spacing:0.22em;text-transform:uppercase;"
    "border-bottom:1px solid #1e2433;padding-bottom:0.4rem;margin-bottom:0.75rem;"
)
_CARD   = (
    "background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:0.9rem 1rem;"
)
_ROW    = (
    "display:flex;justify-content:space-between;align-items:center;"
    "padding:0.3rem 0;border-bottom:1px solid #0d1117;"
)


def render_system_status(
    session: SessionState,
    mode: str,
    backend_name: str,
    inference_fps_measured: float = 0.0,
) -> None:
    """Render the system performance and status panel (inline styles only)."""
    import torch  # lazy import

    st.markdown(f'<div style="{_SEC}">SYSTEM</div>', unsafe_allow_html=True)

    if torch.cuda.is_available():
        gpu_txt   = torch.cuda.get_device_name(0)
        gpu_color = "#22c55e"
    else:
        gpu_txt   = "CPU ONLY"
        gpu_color = "#f59e0b"

    status_color_map = {
        SessionStatus.IDLE:      "#475569",
        SessionStatus.READY:     "#60a5fa",
        SessionStatus.ANALYZING: "#22c55e",
        SessionStatus.PAUSED:    "#f59e0b",
        SessionStatus.COMPLETE:  "#8b5cf6",
    }
    sess_color = status_color_map.get(session.status, "#475569")

    avg_lat = session.avg_latency_ms
    fps_txt = f"{inference_fps_measured:.1f}" if inference_fps_measured > 0 else "—"
    lat_txt = f"{avg_lat:.0f} ms" if avg_lat > 0 else "—"

    rows = [
        ("MODE",             mode.upper(),                  "#22c55e" if mode == "dev" else "#60a5fa"),
        ("BACKEND",          backend_name,                   "#e2e8f0"),
        ("COMPUTE",          gpu_txt,                        gpu_color),
        ("SESSION",          session.status.value,           sess_color),
        ("INFERENCE FPS",    fps_txt,                        "#e2e8f0"),
        ("AVG LATENCY",      lat_txt,                        "#e2e8f0"),
        ("FRAMES READ",      str(session.frames_read),       "#e2e8f0"),
        ("FRAMES PROCESSED", str(session.frames_processed),  "#e2e8f0"),
        ("FRAMES DROPPED",   str(session.frames_dropped),
                             "#ef4444" if session.frames_dropped > 0 else "#e2e8f0"),
    ]

    rows_html = "".join(
        f'<div style="{_ROW}">'
        f'<span style="font-size:0.65rem;color:#475569;letter-spacing:0.1em;">{label}</span>'
        f'<span style="{_MONO};font-size:0.7rem;color:{color};">{value}</span>'
        f'</div>'
        for label, value, color in rows
    )

    model_html = (
        '<div style="margin-top:0.75rem;">'
        '<div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;'
        'text-transform:uppercase;margin-bottom:0.45rem;">MODEL STATUS</div>'
        f'<div style="font-size:0.7rem;color:#22c55e;{_MONO};margin-bottom:0.2rem;">'
        '● ORIGINAL WARD — READY</div>'
        f'<div style="font-size:0.7rem;color:#22c55e;{_MONO};">'
        '● FINE-TUNED WARD — READY</div>'
        '</div>'
    )

    st.markdown(
        f'<div style="{_CARD}">{rows_html}{model_html}</div>',
        unsafe_allow_html=True,
    )


def render_event_log(events) -> None:
    """Render the in-session event log (inline styles only)."""
    st.markdown(f'<div style="{_SEC}">EVENTS</div>', unsafe_allow_html=True)

    if not events:
        st.markdown(
            '<div style="color:#475569;font-size:0.75rem;padding:0.5rem 0;">No events yet.</div>',
            unsafe_allow_html=True,
        )
        return

    border_colors = {
        "CONDITION_CHANGED":      "#22c55e",
        "HIGH_CONFIDENCE":        "#ef4444",
        "MODEL_DISAGREEMENT":     "#f59e0b",
        "PREDICTION_STABILIZED":  "#60a5fa",
        "AMBIGUOUS":              "#eab308",
        "ERROR":                  "#ef4444",
        "INFO":                   "#475569",
    }

    items_html = ""
    for evt in reversed(events[-30:]):
        bcolor = border_colors.get(evt.event_type.value, "#475569")
        items_html += (
            f'<div style="padding:0.45rem 0.75rem;border-left:2px solid {bcolor};'
            f'margin-bottom:0.35rem;background:#0d1117;border-radius:0 6px 6px 0;">'
            f'<div style="{_MONO};color:#475569;font-size:0.6rem;">{evt.time_str}</div>'
            f'<div style="color:#e2e8f0;font-size:0.72rem;margin-top:0.1rem;">{evt.message}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="max-height:320px;overflow-y:auto;padding-right:0.2rem;">'
        f'{items_html}</div>',
        unsafe_allow_html=True,
    )


def render_session_summary(session: SessionState) -> None:
    """Render the end-of-session summary card."""
    if session.status != SessionStatus.COMPLETE:
        return

    st.markdown(
        f'<div style="{_SEC}">ANALYSIS COMPLETE</div>',
        unsafe_allow_html=True,
    )

    dominant  = session.dominant_condition or "—"
    dom_color = {
        "dry": "#22c55e", "damp": "#f59e0b",
        "wet": "#ef4444", "drying": "#f97316",
    }.get(dominant, "#6b7280")

    drop_color = "#ef4444" if session.frames_dropped > 0 else "#22c55e"

    grid = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;">'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Input</div>'
        f'<div style="font-size:0.75rem;color:#94a3b8;word-break:break-all;">{session.input_name or "—"}</div></div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Type</div>'
        f'<div style="{_MONO};font-size:0.75rem;color:#94a3b8;">{(session.input_type or "—").upper()}</div></div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Frames Read</div>'
        f'<div style="{_MONO};font-size:1.1rem;font-weight:600;color:#e2e8f0;">{session.frames_read:,}</div></div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Frames Processed</div>'
        f'<div style="{_MONO};font-size:1.1rem;font-weight:600;color:#e2e8f0;">{session.frames_processed:,}</div></div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Frames Dropped</div>'
        f'<div style="{_MONO};font-size:1.1rem;font-weight:600;color:{drop_color};">{session.frames_dropped:,}</div></div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Duration</div>'
        f'<div style="{_MONO};font-size:1.1rem;font-weight:600;color:#e2e8f0;">{session.elapsed_seconds:.0f}s</div></div>'

        f'<div style="grid-column:span 2;">'
        f'<div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.3rem;">Dominant Condition</div>'
        f'<div style="{_MONO};font-size:1.8rem;font-weight:700;color:{dom_color};text-shadow:0 0 20px {dom_color}60;">{dominant.upper()}</div>'
        f'</div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Avg Confidence</div>'
        f'<div style="{_MONO};font-size:1.1rem;font-weight:600;color:#e2e8f0;">{session.avg_confidence:.0%}</div></div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Model Agreement</div>'
        f'<div style="{_MONO};font-size:1.1rem;font-weight:600;color:#e2e8f0;">{session.model_agreement_rate:.0%}</div></div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Peak Confidence</div>'
        f'<div style="{_MONO};font-size:1.1rem;font-weight:600;color:#e2e8f0;">{session.peak_confidence:.0%}</div></div>'

        f'<div><div style="font-size:0.58rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.2rem;">Avg Latency</div>'
        f'<div style="{_MONO};font-size:1.1rem;font-weight:600;color:#e2e8f0;">{session.avg_latency_ms:.0f} ms</div></div>'

        f'</div>'
    )

    st.markdown(
        f'<div style="{_CARD}background:linear-gradient(135deg,#0d1117,#161b22);border-color:#22c55e20;">'
        f'<div style="{_MONO};font-size:0.95rem;font-weight:700;color:#22c55e;margin-bottom:1rem;letter-spacing:0.2em;">'
        f'ANALYSIS COMPLETE ✓</div>{grid}</div>',
        unsafe_allow_html=True,
    )
