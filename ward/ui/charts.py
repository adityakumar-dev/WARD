"""
WARD — Live Plotly Charts (compact grid + expandable modal)
============================================================
Compact 2×3 chart grid for the unified dashboard.
Each chart has an ⤢ expand button that opens a JS full-screen modal.

Charts:
 1. Class probabilities over time
 2. Fused confidence over time
 3. Temporal stability over time
 4. Model agreement timeline
 5. Inference latency
"""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go
import streamlit as st

from models.labels import WARD_CLASSES
from session.state import FrameRecord
from ui.dashboard import CONDITION_COLORS

_BG     = "#0d1117"
_PAPER  = "#0d1117"
_GRID   = "#1e2433"
_FONT   = "#94a3b8"
_MONO   = "JetBrains Mono, monospace"


def _hex_to_rgba(hex_color: str, alpha: float = 0.09) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(title: str, y_range=None, height: int = 180) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=10, color=_FONT, family=_MONO)),
        plot_bgcolor=_BG,
        paper_bgcolor=_PAPER,
        font=dict(family=_MONO, size=9, color=_FONT),
        margin=dict(l=32, r=12, t=28, b=22),
        xaxis=dict(showgrid=True, gridcolor=_GRID, gridwidth=1, zeroline=False,
                   tickfont=dict(size=8), showline=False),
        yaxis=dict(showgrid=True, gridcolor=_GRID, gridwidth=1, zeroline=False,
                   tickfont=dict(size=8), range=y_range),
        legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)",
                    bordercolor=_GRID, borderwidth=1, orientation="h",
                    yanchor="top", y=-0.18, xanchor="left", x=0),
        height=height,
    )


def _chart_card_header(chart_key: str, title: str) -> None:
    """Render a section header with expand button for a chart."""
    st.markdown(
        f"""
        <div class="chart-card-title">
            <span>{title}</span>
            <button class="expand-btn" onclick="wardExpandChart('{chart_key}', '{title}')">
                ⤢ Expand
            </button>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _wrap_chart(chart_key: str) -> str:
    """Return opening div with data-chart-id attribute."""
    return f'<div data-chart-id="{chart_key}" style="margin:-0.25rem -0.1rem;">'


def render_charts(records: List[FrameRecord]) -> None:
    """Render compact 2×3 chart grid — each chart expandable via popup."""
    st.markdown('<div class="section-header">LIVE CHARTS</div>', unsafe_allow_html=True)

    if not records:
        st.markdown(
            '<div style="color:#475569;font-size:0.78rem;padding:0.5rem 0;">'
            'Charts appear after first inference.</div>',
            unsafe_allow_html=True,
        )
        return

    xs = [r.frame_index for r in records]

    col1, col2 = st.columns(2)

    # ── Chart 1: Class probabilities ──────────────────────────────────────
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        _chart_card_header("chart-probs", "CLASS PROBABILITIES")
        st.markdown(_wrap_chart("chart-probs"), unsafe_allow_html=True)
        fig = go.Figure()
        for cls in WARD_CLASSES:
            ys = [r.fused_scores.get(cls, 0.0) for r in records]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, name=cls.upper(), mode="lines",
                line=dict(color=CONDITION_COLORS.get(cls, "#6b7280"), width=1.5),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(CONDITION_COLORS.get(cls, "#6b7280")),
            ))
        fig.update_layout(**_base_layout("", y_range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Chart 2: Fused confidence ─────────────────────────────────────────
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        _chart_card_header("chart-conf", "FUSED CONFIDENCE")
        st.markdown(_wrap_chart("chart-conf"), unsafe_allow_html=True)
        ys_conf = [r.fused_confidence for r in records]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=xs, y=ys_conf, name="CONFIDENCE", mode="lines",
            line=dict(color="#22c55e", width=2),
            fill="tozeroy", fillcolor=_hex_to_rgba("#22c55e"),
        ))
        fig2.update_layout(**_base_layout("", y_range=[0, 1]))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div></div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    # ── Chart 3: Temporal stability ───────────────────────────────────────
    with col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        _chart_card_header("chart-stab", "TEMPORAL STABILITY")
        st.markdown(_wrap_chart("chart-stab"), unsafe_allow_html=True)
        ys_stab = [r.stability for r in records]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=xs, y=ys_stab, name="STABILITY", mode="lines",
            line=dict(color="#60a5fa", width=2),
            fill="tozeroy", fillcolor=_hex_to_rgba("#60a5fa"),
        ))
        fig3.update_layout(**_base_layout("", y_range=[0, 1]))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Chart 4: Model agreement ──────────────────────────────────────────
    with col4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        _chart_card_header("chart-agree", "MODEL AGREEMENT")
        st.markdown(_wrap_chart("chart-agree"), unsafe_allow_html=True)
        agree_colors = ["#22c55e" if r.agreement else "#ef4444" for r in records]
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=xs, y=[1] * len(records),
            marker_color=agree_colors,
            name="AGREEMENT", showlegend=False,
        ))
        fig4.update_layout(**_base_layout(""))
        fig4.update_yaxes(showticklabels=False)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Chart 5: Inference latency (full width) ───────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    _chart_card_header("chart-lat", "INFERENCE LATENCY (ms)")
    st.markdown(_wrap_chart("chart-lat"), unsafe_allow_html=True)
    ys_lat_orig = [r.original_latency_ms for r in records]
    ys_lat_fine = [r.fine_latency_ms for r in records]
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=xs, y=ys_lat_orig, name="ORIGINAL", mode="lines",
        line=dict(color="#60a5fa", width=1.5),
    ))
    fig5.add_trace(go.Scatter(
        x=xs, y=ys_lat_fine, name="FINE-TUNED", mode="lines",
        line=dict(color="#22c55e", width=1.5),
    ))
    fig5.update_layout(**_base_layout("", height=160))
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div></div>', unsafe_allow_html=True)
