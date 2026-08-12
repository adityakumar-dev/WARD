"""
WARD — Temporal Timeline Panel
================================
Renders a scrollable table timeline of frame predictions and a
horizontal condition-strip chart showing condition transitions.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from session.state import FrameRecord
from ui.dashboard import CONDITION_COLORS


def render_timeline(records: List[FrameRecord]) -> None:
    """Render the temporal prediction timeline."""
    st.markdown('<div class="section-header">TEMPORAL TIMELINE</div>', unsafe_allow_html=True)

    if not records:
        st.markdown(
            '<div style="color:#475569;font-size:0.8rem;padding:1rem;">No frames processed yet.</div>',
            unsafe_allow_html=True,
        )
        return

    # Show last 50 records
    recent = records[-50:]

    rows_html = ""
    for r in reversed(recent):
        ts = f"{r.timestamp_s:.1f}s" if r.timestamp_s else f"#{r.frame_index}"
        orig_color = CONDITION_COLORS.get(r.original_label, "#6b7280")
        fine_color = CONDITION_COLORS.get(r.fine_label, "#6b7280")
        fused_color = CONDITION_COLORS.get(r.fused_label, "#6b7280")
        stable_color = CONDITION_COLORS.get(r.stable_label, "#6b7280")
        agree_txt = "✓" if r.agreement else "✗"
        agree_color = "#22c55e" if r.agreement else "#ef4444"
        ambig_txt = "⚠" if r.ambiguous else ""

        rows_html += f"""
        <tr>
            <td style="color:#475569;">{ts}</td>
            <td style="color:#475569;">{r.frame_index}</td>
            <td style="color:{orig_color};">{r.original_label.upper()}</td>
            <td style="color:{fine_color};">{r.fine_label.upper()}</td>
            <td style="color:{fused_color};font-weight:600;">{r.fused_label.upper()}</td>
            <td style="color:{stable_color};font-weight:700;">{r.stable_label.upper()}</td>
            <td style="color:#94a3b8;">{r.fused_confidence:.0%}</td>
            <td style="color:{agree_color};">{agree_txt}</td>
            <td style="color:#f59e0b;">{ambig_txt}</td>
        </tr>
        """

    st.markdown(
        f"""
        <div class="scroll-box">
        <table class="fusion-table" style="width:100%;">
            <thead>
                <tr>
                    <th style="text-align:left;">TIME</th>
                    <th style="text-align:left;">#</th>
                    <th style="text-align:left;">ORIG</th>
                    <th style="text-align:left;">FINE</th>
                    <th style="text-align:left;">FUSED</th>
                    <th style="text-align:left;">STABLE</th>
                    <th>CONF</th>
                    <th>AGR</th>
                    <th>AMB</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_condition_strip(records: List[FrameRecord]) -> None:
    """
    Render a horizontal strip showing condition transitions over time.
    Each segment is coloured by stable_label.
    """
    if not records:
        return

    # Build transition segments
    segments: list[tuple[str, int]] = []
    for r in records:
        if segments and segments[-1][0] == r.stable_label:
            segments[-1] = (segments[-1][0], segments[-1][1] + 1)
        else:
            segments.append((r.stable_label, 1))

    total = sum(c for _, c in segments)
    if total == 0:
        return

    segments_html = ""
    for label, count in segments:
        pct = count / total * 100
        color = CONDITION_COLORS.get(label, "#6b7280")
        segments_html += f'<div style="flex:{pct};background:{color};min-width:2px;" title="{label.upper()} ({count} frames)"></div>'

    st.markdown(
        f"""
        <div style="margin:0.5rem 0;">
            <div style="font-size:0.62rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.3rem;">CONDITION TIMELINE</div>
            <div style="display:flex;height:12px;border-radius:6px;overflow:hidden;border:1px solid #21262d;">
                {segments_html}
            </div>
            <div style="display:flex;gap:1rem;margin-top:0.4rem;flex-wrap:wrap;">
                {"".join(f'<div style="display:flex;align-items:center;gap:4px;font-size:0.62rem;color:#94a3b8;"><div style="width:8px;height:8px;background:{CONDITION_COLORS.get(lbl, "#6b7280")};border-radius:50%;"></div>{lbl.upper()}</div>' for lbl in dict.fromkeys(lbl for lbl, _ in segments))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
