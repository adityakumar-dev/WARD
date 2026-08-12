"""
WARD — Model Fusion Panel
==========================
Renders the MODEL FUSION section showing:
- Per-model predictions and confidence
- Fused result
- Contribution weight bars
- Agreement badge
- Model comparison table
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from models.fusion import FusionResult
from models.labels import WARD_CLASSES
from ui.dashboard import CONDITION_COLORS, _condition_color


def render_fusion_panel(fusion: Optional[FusionResult]) -> None:
    """Render the full model fusion section."""
    st.markdown('<div class="section-header">MODEL FUSION</div>', unsafe_allow_html=True)

    if fusion is None:
        st.markdown(
            '<div style="color:#475569;font-size:0.8rem;padding:1rem;">No prediction yet.</div>',
            unsafe_allow_html=True,
        )
        return

    orig = fusion.original
    fine = fusion.fine
    fused_scores = fusion.fused_scores

    orig_color = _condition_color(orig.top_label)
    fine_color = _condition_color(fine.top_label)
    fused_color = _condition_color(fusion.top_label)

    agree_cls = "agree-yes" if fusion.agreement else "agree-no"
    agree_txt = "YES" if fusion.agreement else "NO"

    # ── Fusion summary card ─────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:1.25rem;">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;align-items:center;">

                <div style="text-align:center;">
                    <div style="font-size:0.62rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.3rem;">ORIGINAL WARD</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:{orig_color};">{orig.top_label.upper()}</div>
                    <div style="font-size:0.75rem;color:#94a3b8;">{orig.top_confidence:.0%}</div>
                    <div style="font-size:0.6rem;color:#475569;margin-top:0.2rem;">{orig.latency_ms:.0f} ms</div>
                </div>

                <div style="text-align:center;border-left:1px solid #21262d;border-right:1px solid #21262d;padding:0 1rem;">
                    <div style="font-size:0.62rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.3rem;">FINE-TUNED WARD</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:{fine_color};">{fine.top_label.upper()}</div>
                    <div style="font-size:0.75rem;color:#94a3b8;">{fine.top_confidence:.0%}</div>
                    <div style="font-size:0.6rem;color:#475569;margin-top:0.2rem;">{fine.latency_ms:.0f} ms</div>
                </div>

                <div style="text-align:center;">
                    <div style="font-size:0.62rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.3rem;">FUSED RESULT</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:{fused_color};text-shadow:0 0 15px {fused_color}80;">{fusion.top_label.upper()}</div>
                    <div style="font-size:0.75rem;color:#94a3b8;">{fusion.top_confidence:.0%}</div>
                    <div style="font-size:0.7rem;margin-top:0.3rem;">Agreement: <span class="{agree_cls}">{agree_txt}</span></div>
                </div>

            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Contribution bars ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Contribution Weights</div>
                <div style="margin-top:0.5rem;">
                    <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94a3b8;margin-bottom:0.2rem;">
                        <span>Original WARD</span><span>30%</span>
                    </div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:30%;background:#60a5fa;"></div></div>
                    <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94a3b8;margin-top:0.5rem;margin-bottom:0.2rem;">
                        <span>Fine-tuned WARD</span><span>70%</span>
                    </div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:70%;background:#22c55e;"></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        ambig_txt = "⚠️ YES" if fusion.ambiguous else "✅ NO"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Prediction Quality</div>
                <div style="margin-top:0.5rem;">
                    <div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #1e2433;">
                        <span style="font-size:0.72rem;color:#94a3b8;">Model Agreement</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;" class="{agree_cls}">{agree_txt}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #1e2433;">
                        <span style="font-size:0.72rem;color:#94a3b8;">Ambiguous</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;">{ambig_txt}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:0.3rem 0;">
                        <span style="font-size:0.72rem;color:#94a3b8;">Margin</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#e2e8f0;">{fusion.margin:.2%}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Comparison table ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">CLASS COMPARISON</div>', unsafe_allow_html=True)

    rows_html = ""
    for cls in WARD_CLASSES:
        orig_s = orig.scores.get(cls)
        fine_s = fine.scores.get(cls)
        fused_s = fused_scores.get(cls, 0.0)

        orig_txt = f"{orig_s:.0%}" if orig_s is not None else "—"
        fine_txt = f"{fine_s:.0%}" if fine_s is not None else "—"
        fused_txt = f"{fused_s:.0%}"
        color = _condition_color(cls)
        bold = "font-weight:700;" if cls == fusion.top_label else ""

        rows_html += f"""
        <tr>
            <td style="color:{color};{bold}">{cls.upper()}</td>
            <td style="{bold}">{orig_txt}</td>
            <td style="{bold}">{fine_txt}</td>
            <td style="color:{color};{bold}">{fused_txt}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="fusion-table">
            <thead>
                <tr>
                    <th>CLASS</th>
                    <th>ORIGINAL</th>
                    <th>FINE-TUNED</th>
                    <th>FUSED</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
