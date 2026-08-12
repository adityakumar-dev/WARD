"""
WARD — Main Streamlit Dashboard
=================================
Unified single-frame layout.
Everything lives on one scrollable screen:
  - Condition hero  (left)
  - Input + live preview (right)
  - Fusion panel (left)
  - Mini event log + weather (right)
  - 2×3 chart grid (full width, each expandable)
"""

from __future__ import annotations

from typing import Optional

import streamlit as st
import textwrap

from models.fusion import FusionResult
from models.labels import WARD_CLASSES
from session.state import SessionState, SessionStatus
from temporal.engine import TemporalState

# ── Color palette ──────────────────────────────────────────────────────────
CONDITION_COLORS = {
    "dry":    "#22c55e",
    "damp":   "#f59e0b",
    "wet":    "#ef4444",
    "drying": "#f97316",
    "unknown": "#6b7280",
}

CONDITION_EMOJIS = {
    "dry":    "☀️",
    "damp":   "🌫️",
    "wet":    "🌧️",
    "drying": "🌤️",
    "unknown": "❓",
}


def _condition_color(label: str) -> str:
    return CONDITION_COLORS.get(label.lower(), "#6b7280")


def _stability_label(stability: float) -> str:
    if stability >= 0.80:
        return "HIGH"
    if stability >= 0.50:
        return "MED"
    return "LOW"


def _stability_color(stability: float) -> str:
    if stability >= 0.80:
        return "#22c55e"
    if stability >= 0.50:
        return "#f59e0b"
    return "#ef4444"


def _hex_rgba(hex_color: str, alpha: float = 0.10) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════

def render_css() -> None:
    """Inject all global CSS including modals, popup, expand buttons."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* ── Base ─────────────────────────────────────────── */
        .stApp {
            background: #060608;
            background-image:
                radial-gradient(ellipse at 15% 0%,  rgba(34,197,94,0.05)  0%, transparent 55%),
                radial-gradient(ellipse at 85% 100%, rgba(96,165,250,0.04) 0%, transparent 55%);
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1600px;
        }
        #MainMenu, footer, header { visibility: hidden; }

        /* ── WARD Header ───────────────────────────────────── */
        .ward-header {
            background: linear-gradient(135deg, rgba(13,17,23,0.96) 0%, rgba(20,25,32,0.96) 100%);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 1.25rem 2rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }
        .ward-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(34,197,94,0.5) 30%,
                rgba(34,197,94,0.9) 50%,
                rgba(34,197,94,0.5) 70%,
                transparent 100%);
        }
        .ward-header::after {
            content: '';
            position: absolute;
            bottom: -100px; right: -60px;
            width: 280px; height: 280px;
            background: radial-gradient(circle, rgba(34,197,94,0.06) 0%, transparent 70%);
            pointer-events: none;
        }
        .ward-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: #22c55e;
            letter-spacing: 0.38em;
            text-shadow:
                0 0 20px rgba(34,197,94,0.7),
                0 0 60px rgba(34,197,94,0.3),
                0 0 120px rgba(34,197,94,0.1);
            margin: 0;
        }
        .ward-subtitle {
            color: #374151;
            font-size: 0.68rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            margin-top: 0.3rem;
        }
        .mode-badge {
            display: inline-block;
            padding: 0.22rem 0.75rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .mode-dev  { background: rgba(34,197,94,0.1);  color: #22c55e; border: 1px solid rgba(34,197,94,0.22); }
        .mode-prod { background: rgba(59,130,246,0.1);  color: #60a5fa; border: 1px solid rgba(59,130,246,0.22); }

        /* ── Condition hero card ───────────────────────────── */
        .condition-card {
            border-radius: 18px;
            padding: 2rem 1.75rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
            background: rgba(10,13,18,0.85);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            position: relative;
            overflow: hidden;
            transition: border-color 0.5s ease, box-shadow 0.5s ease;
        }
        /* ambient glow layer — filled by inline style */
        .condition-card .cond-glow {
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 55%;
            pointer-events: none;
            border-radius: 0 0 18px 18px;
        }
        /* animated scan line for analyzing state */
        .condition-card.analyzing::after {
            content: '';
            position: absolute;
            left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #22c55e, transparent);
            animation: scan 2.5s linear infinite;
            z-index: 1;
        }
        @keyframes scan {
            0%   { top: -2px; opacity: 0; }
            4%   { opacity: 1; }
            96%  { opacity: 1; }
            100% { top: 102%; opacity: 0; }
        }
        .condition-label-huge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 3.4rem;
            font-weight: 700;
            letter-spacing: 0.2em;
            margin: 0 0 0.1rem;
            line-height: 1.05;
            animation: glow-breathe 4s ease-in-out infinite;
        }
        @keyframes glow-breathe {
            0%, 100% { filter: brightness(1)   drop-shadow(0 0 8px  currentColor); }
            50%       { filter: brightness(1.12) drop-shadow(0 0 22px currentColor); }
        }
        .condition-sublabel {
            font-size: 0.6rem;
            color: #374151;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        /* ── Metric rings (conic gradient) ─────────────────── */
        .ring-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.45rem;
        }
        .ring-label {
            font-size: 0.55rem;
            color: #374151;
            letter-spacing: 0.2em;
            text-transform: uppercase;
        }

        /* ── Class Cards ───────────────────────────────────── */
        .class-card {
            background: rgba(10,13,18,0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.4rem;
            transition: border-color 0.35s, box-shadow 0.35s, background 0.35s;
            position: relative;
            overflow: hidden;
        }
        .class-card::before {
            content: '';
            position: absolute;
            inset: 0;
            opacity: 0;
            transition: opacity 0.35s;
            background: linear-gradient(135deg, var(--class-color, #22c55e) 0%, transparent 55%);
        }
        .class-card.active {
            border-color: var(--class-color);
            box-shadow:
                0 0 22px color-mix(in srgb, var(--class-color) 18%, transparent),
                inset 0 0 18px color-mix(in srgb, var(--class-color) 6%, transparent);
        }
        .class-card.active::before { opacity: 0.08; }
        .class-name {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }
        .class-prob {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.08rem;
            font-weight: 700;
        }
        .prob-bar-bg {
            background: rgba(255,255,255,0.04);
            border-radius: 3px;
            height: 3px;
            margin-top: 0.45rem;
            overflow: hidden;
        }
        .prob-bar-fill {
            border-radius: 3px;
            height: 3px;
            transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        .prob-bar-fill::after {
            content: '';
            position: absolute;
            right: 0; top: 0; bottom: 0;
            width: 8px;
            background: rgba(255,255,255,0.6);
            border-radius: 3px;
            filter: blur(1px);
        }

        /* ── Metric Cards ──────────────────────────────────── */
        .metric-card {
            background: rgba(10,13,18,0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }
        .metric-label {
            font-size: 0.58rem;
            color: #374151;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.15rem;
            font-weight: 600;
            color: #e2e8f0;
        }
        .metric-sub {
            font-size: 0.62rem;
            color: #475569;
            margin-top: 0.2rem;
        }

        /* ── Section Headers ───────────────────────────────── */
        .section-header {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            font-weight: 600;
            color: #374151;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            padding-bottom: 0.4rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .section-header::before {
            content: '';
            display: inline-block;
            width: 2px;
            height: 9px;
            background: linear-gradient(180deg, #22c55e, #16a34a);
            border-radius: 2px;
            flex-shrink: 0;
        }

        /* ── Status Pills ──────────────────────────────────── */
        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            backdrop-filter: blur(4px);
        }
        .pill-stable    { background: rgba(34,197,94,0.1);  color: #22c55e;  border: 1px solid rgba(34,197,94,0.22); }
        .pill-uncertain { background: rgba(245,158,11,0.1); color: #f59e0b;  border: 1px solid rgba(245,158,11,0.22); }
        .pill-ambiguous { background: rgba(234,179,8,0.1);  color: #eab308;  border: 1px solid rgba(234,179,8,0.22); }

        /* ── Agreement ─────────────────────────────────────── */
        .agree-yes { color: #22c55e; font-weight: 700; }
        .agree-no  { color: #ef4444; font-weight: 700; }

        /* ── Event Log ─────────────────────────────────────── */
        .event-item {
            padding: 0.42rem 0.75rem;
            border-left: 2px solid #22c55e;
            margin-bottom: 0.3rem;
            background: rgba(10,13,18,0.6);
            border-radius: 0 8px 8px 0;
            font-size: 0.72rem;
            transition: background 0.2s;
        }
        .event-item:hover { background: rgba(22,27,34,0.7); }
        .event-time {
            font-family: 'JetBrains Mono', monospace;
            color: #374151;
            font-size: 0.6rem;
        }

        /* ── Tables ────────────────────────────────────────── */
        .fusion-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
        }
        .fusion-table th {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            color: #374151;
            text-transform: uppercase;
            padding: 0.35rem 0.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            text-align: right;
        }
        .fusion-table th:first-child { text-align: left; }
        .fusion-table td {
            font-family: 'JetBrains Mono', monospace;
            padding: 0.35rem 0.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            text-align: right;
        }
        .fusion-table td:first-child { text-align: left; color: #4b5563; }
        .fusion-table tr:hover td { background: rgba(22,27,34,0.5); }

        /* ── Scroll box ─────────────────────────────────────── */
        .scroll-box {
            max-height: 260px;
            overflow-y: auto;
            padding-right: 0.2rem;
        }
        .scroll-box::-webkit-scrollbar { width: 2px; }
        .scroll-box::-webkit-scrollbar-track { background: transparent; }
        .scroll-box::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }

        /* ── Live preview frame ─────────────────────────────── */
        .preview-frame {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.06);
            background: #060608;
            overflow: hidden;
            position: relative;
        }
        .preview-label-overlay {
            position: absolute;
            bottom: 0; left: 0; right: 0;
            padding: 0.5rem 0.75rem;
            background: linear-gradient(0deg, rgba(6,6,8,0.95) 0%, transparent 100%);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.12em;
        }

        /* ── Chart cards ─────────────────────────────────────── */
        .chart-card {
            background: rgba(10,13,18,0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 14px;
            padding: 0.65rem 0.85rem 0.45rem;
            position: relative;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        .chart-card:hover {
            border-color: rgba(255,255,255,0.09);
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        }
        .chart-card-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.56rem;
            color: #374151;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .expand-btn {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            padding: 2px 7px;
            border-radius: 4px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            color: #374151;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.54rem;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
        }
        .expand-btn:hover {
            background: rgba(34,197,94,0.08);
            color: #22c55e;
            border-color: rgba(34,197,94,0.22);
        }

        /* ── PROCESSING POPUP ────────────────────────────────── */
        #ward-proc-overlay {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 99998;
            background: rgba(6,6,8,0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            animation: fadeInBg 0.3s ease;
        }
        @keyframes fadeInBg { from { opacity: 0; } to { opacity: 1; } }
        #ward-proc-overlay.active { display: flex; align-items: center; justify-content: center; }

        #ward-proc-card {
            background: rgba(13,17,23,0.97);
            backdrop-filter: blur(24px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1.75rem 2rem;
            width: min(520px, 92vw);
            box-shadow: 0 30px 100px rgba(0,0,0,0.7);
            animation: slideUp 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
            position: relative;
            overflow: hidden;
        }
        #ward-proc-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, #22c55e, #f97316, transparent);
            animation: shimmer 2.5s linear infinite;
            background-size: 200% 100%;
        }
        @keyframes shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(28px) scale(0.96); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }

        /* pulse ring */
        .proc-pulse-ring {
            position: relative;
            width: 12px; height: 12px;
            display: inline-block;
            margin-right: 8px;
        }
        .proc-pulse-ring::before,
        .proc-pulse-ring::after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background: #22c55e;
        }
        .proc-pulse-ring::before { animation: ring-pulse 1.5s ease-out infinite; }
        .proc-pulse-ring::after  { opacity: 0.5; }
        @keyframes ring-pulse {
            0%   { transform: scale(1); opacity: 1; }
            100% { transform: scale(2.8); opacity: 0; }
        }
        .proc-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 600;
            color: #22c55e;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .proc-frame-wrap {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 1rem;
            min-height: 80px;
            background: #060608;
            position: relative;
        }
        .proc-label-badge {
            position: absolute;
            bottom: 8px; left: 8px;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            backdrop-filter: blur(4px);
        }
        .proc-progress-bar-bg {
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            height: 6px;
            margin-bottom: 0.5rem;
            overflow: hidden;
        }
        .proc-progress-bar-fill {
            height: 6px;
            border-radius: 6px;
            background: linear-gradient(90deg, #22c55e, #16a34a);
            transition: width 0.3s ease;
            position: relative;
        }
        .proc-progress-bar-fill::after {
            content: '';
            position: absolute;
            right: 0; top: 0; bottom: 0;
            width: 20px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3));
            animation: shine 1s ease infinite;
        }
        @keyframes shine {
            0%, 100% { opacity: 0; }
            50%       { opacity: 1; }
        }
        .proc-stats {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .proc-stat {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px;
            padding: 0.5rem 0.6rem;
            text-align: center;
        }
        .proc-stat-label {
            font-size: 0.55rem;
            color: #475569;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }
        .proc-stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-top: 0.2rem;
        }

        /* ── CHART MODAL (expand popup) ──────────────────────── */
        #ward-chart-overlay {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 99999;
            background: rgba(6,6,8,0.88);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }
        #ward-chart-overlay.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #ward-chart-modal {
            background: rgba(13,17,23,0.96);
            backdrop-filter: blur(24px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            width: min(900px, 95vw);
            max-height: 80vh;
            overflow: auto;
            padding: 1.5rem;
            box-shadow: 0 40px 120px rgba(0,0,0,0.8);
            animation: slideUp 0.3s ease;
            position: relative;
        }
        #ward-chart-modal::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(34,197,94,0.6), rgba(96,165,250,0.6), transparent);
        }
        .chart-modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .chart-modal-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: #4b5563;
            letter-spacing: 0.22em;
            text-transform: uppercase;
        }
        .modal-close-btn {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            color: #4b5563;
            border-radius: 8px;
            padding: 0.3rem 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .modal-close-btn:hover {
            background: rgba(239,68,68,0.1);
            color: #ef4444;
            border-color: rgba(239,68,68,0.22);
        }

        /* ── Info popup button ──────────────────────────────── */
        .popup-view-btn {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 0.25rem 0.75rem;
            background: rgba(96,165,250,0.08);
            border: 1px solid rgba(96,165,250,0.2);
            color: #60a5fa;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.08em;
            cursor: pointer;
            transition: background 0.2s;
            text-decoration: none;
        }
        .popup-view-btn:hover { background: rgba(96,165,250,0.14); }

        /* ── Divider ────────────────────────────────────────── */
        .ward-divider {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.04);
            margin: 0.75rem 0;
        }

        /* ── Awaiting state ─────────────────────────────────── */
        .awaiting-card {
            background: rgba(10,13,18,0.5);
            backdrop-filter: blur(12px);
            border: 1px dashed rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 3.5rem 2rem;
            text-align: center;
        }
        .awaiting-pulse {
            width: 52px; height: 52px;
            border-radius: 50%;
            border: 1px solid rgba(34,197,94,0.2);
            margin: 0 auto 1.25rem;
            position: relative;
            animation: breath 3.5s ease-in-out infinite;
        }
        .awaiting-pulse::after {
            content: '🛣️';
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }
        @keyframes breath {
            0%, 100% { transform: scale(1);    box-shadow: 0 0 12px rgba(34,197,94,0.04); }
            50%       { transform: scale(1.06); box-shadow: 0 0 28px rgba(34,197,94,0.14); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # JS for chart expand modal
    st.markdown(
        """
        <script>
        function wardExpandChart(chartId, chartTitle) {
            var overlay = document.getElementById('ward-chart-overlay');
            var title   = document.getElementById('ward-chart-modal-title');
            var body    = document.getElementById('ward-chart-modal-body');
            if (!overlay) return;
            title.textContent = chartTitle;
            body.innerHTML = '';
            var src = document.querySelector('[data-chart-id="' + chartId + '"] .js-plotly-plot');
            if (src) {
                var clone = src.cloneNode(true);
                clone.style.height = '65vh';
                body.appendChild(clone);
                if (window.Plotly) {
                    Plotly.relayout(clone, { height: window.innerHeight * 0.65, width: null, autosize: true });
                }
            }
            overlay.classList.add('active');
        }
        function wardCloseChartModal() {
            var overlay = document.getElementById('ward-chart-overlay');
            if (overlay) overlay.classList.remove('active');
        }
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') { wardCloseChartModal(); }
        });
        </script>

        <!-- Chart expand modal DOM -->
        <div id="ward-chart-overlay" onclick="if(event.target===this)wardCloseChartModal()">
            <div id="ward-chart-modal">
                <div class="chart-modal-header">
                    <div class="chart-modal-title" id="ward-chart-modal-title">CHART</div>
                    <button class="modal-close-btn" onclick="wardCloseChartModal()">✕ Close</button>
                </div>
                <div id="ward-chart-modal-body"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

def render_header(mode: str, backend_name: str) -> None:
    badge_cls = "mode-dev" if mode == "dev" else "mode-prod"
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="ward-header">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="ward-title">W▸A▸R▸D</div>
                        <div class="ward-subtitle">AI Road Surface Condition Intelligence</div>
                    </div>
                    <div style="text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:0.4rem;">
                        <div class="mode-badge {badge_cls}">{mode.upper()} MODE</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.56rem;color:#374151;">{backend_name}</div>
                    </div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Condition hero + class cards
# ═══════════════════════════════════════════════════════════════════════════

def render_main_condition(
    fusion: Optional[FusionResult],
    temporal: Optional[TemporalState],
    is_analyzing: bool = False,
) -> None:
    """Condition hero card + 4 class score cards."""
    if fusion is None or temporal is None:
        st.markdown(
            textwrap.dedent(
                """
                <div class="awaiting-card">
                    <div class="awaiting-pulse"></div>
                    <div style="color:#374151;font-family:'JetBrains Mono',monospace;
                                letter-spacing:0.22em;font-size:0.78rem;">AWAITING INPUT</div>
                    <div style="color:#1f2937;font-size:0.68rem;margin-top:0.5rem;">
                        Upload an image, video or frame folder to begin
                    </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
        return

    label       = temporal.label
    color       = _condition_color(label)
    emoji       = CONDITION_EMOJIS.get(label, "")
    conf_pct    = int(temporal.confidence * 100)
    stab_pct    = int(temporal.stability * 100)
    stab_label  = _stability_label(temporal.stability)
    stab_color  = _stability_color(temporal.stability)
    glow_color  = _hex_rgba(color, 0.12)
    analyzing_cls = "analyzing" if is_analyzing else ""

    if temporal.ambiguous:
        pill = '<span class="status-pill pill-ambiguous">AMBIGUOUS</span>'
    elif temporal.stability >= 0.70:
        pill = '<span class="status-pill pill-stable">STABLE</span>'
    else:
        pill = '<span class="status-pill pill-uncertain">UNCERTAIN</span>'

    # Conic gradient rings for confidence & stability
    conf_bg = f"conic-gradient({color} {conf_pct}%, rgba(255,255,255,0.04) 0)"
    stab_bg = f"conic-gradient({stab_color} {stab_pct}%, rgba(255,255,255,0.04) 0)"

    ring  = "width:80px;height:80px;border-radius:50%;display:flex;align-items:center;justify-content:center;position:relative;margin:0 auto;"
    inner = ("position:absolute;width:60px;height:60px;border-radius:50%;background:#060608;"
             "display:flex;align-items:center;justify-content:center;flex-direction:column;")

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="condition-card {analyzing_cls}" style="border-color:{color}22;box-shadow:0 0 40px {_hex_rgba(color,0.06)},inset 0 0 40px {_hex_rgba(color,0.02)};">
                <!-- ambient glow -->
                <div class="cond-glow" style="background:radial-gradient(ellipse at 50% 100%, {glow_color} 0%, transparent 70%);"></div>
                <div class="condition-sublabel">WARD CONDITION</div>
                <div class="condition-label-huge" style="color:{color};">{emoji} {label.upper()}</div>
                <!-- metric rings row -->
                <div style="margin-top:1.75rem;display:flex;justify-content:center;gap:2.5rem;align-items:flex-start;position:relative;z-index:1;">
                    <!-- Confidence ring -->
                    <div class="ring-wrap">
                        <div style="{ring}background:{conf_bg};">
                            <div style="{inner}">
                                <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:{color};line-height:1;">{conf_pct}%</span>
                            </div>
                        </div>
                        <div class="ring-label">CONFIDENCE</div>
                    </div>
                    <!-- Stability ring -->
                    <div class="ring-wrap">
                        <div style="{ring}background:{stab_bg};">
                            <div style="{inner}">
                                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700;color:{stab_color};line-height:1;">{stab_label}</span>
                                <span style="font-size:0.5rem;color:#374151;letter-spacing:0.1em;margin-top:1px;">{stab_pct}%</span>
                            </div>
                        </div>
                        <div class="ring-label">STABILITY</div>
                    </div>
                    <!-- Status -->
                    <div class="ring-wrap" style="padding-top:22px;">
                        {pill}
                        <div class="ring-label" style="margin-top:0.55rem;">STATUS</div>
                    </div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    # 4 class score cards
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    smoothed = temporal.smoothed_scores if temporal else {}
    for i, cls in enumerate(WARD_CLASSES):
        score = smoothed.get(cls, 0.0)
        pct   = int(score * 100)
        ccol  = _condition_color(cls)
        is_active   = cls == label
        active_cls  = "active" if is_active else ""
        candidate_txt = ""
        if temporal.candidate_label == cls:
            candidate_txt = f"<div style='font-size:0.5rem;color:#f59e0b;margin-top:0.25rem;letter-spacing:0.08em;'>▸ CANDIDATE ({temporal.candidate_streak})</div>"

        with cols[i]:
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="class-card {active_cls}" style="--class-color:{ccol};">
                        <div style="display:flex;justify-content:space-between;align-items:center;position:relative;z-index:1;">
                            <div class="class-name" style="color:{ccol};">{cls.upper()}</div>
                            <div class="class-prob" style="color:{ccol if is_active else '#e2e8f0'};">{pct}%</div>
                        </div>
                        <div class="prob-bar-bg" style="position:relative;z-index:1;">
                            <div class="prob-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{ccol}cc,{ccol});"></div>
                        </div>
                        {candidate_txt}
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
