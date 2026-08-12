"""
WARD — Streamlit Application Entry Point (Unified Dashboard)
=============================================================
Run with:
    WARD_MODE=dev  streamlit run app.py
    WARD_MODE=prod streamlit run app.py

Single-page layout. No tabs.
Processing display uses native st.status + st.progress + st.image + st.metric.
Popups (Fusion / Timeline / System / Weather) use st.dialog.
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import cfg

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WARD — Road Surface Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Model loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading WARD models…")
def get_provider():
    if cfg.is_dev:
        from models.loader import load_models
        from inference.local_provider import LocalInferenceProvider
        pair = load_models(cfg)
        return LocalInferenceProvider(pair, cfg)
    else:
        from inference.hf_provider import HFInferenceProvider
        return HFInferenceProvider(cfg)


provider = get_provider()


# ── Session init ───────────────────────────────────────────────────────────
def _init_state():
    if "ward_session" not in st.session_state:
        from session.state import SessionState
        st.session_state["ward_session"] = SessionState()
    if "ward_events" not in st.session_state:
        from session.events import EventLog
        st.session_state["ward_events"] = EventLog()
    if "ward_temporal" not in st.session_state:
        from temporal.engine import TemporalEngine
        st.session_state["ward_temporal"] = TemporalEngine(cfg)
    if "weather_cache" not in st.session_state:
        from weather.cache import WeatherCache
        st.session_state["weather_cache"] = WeatherCache(cfg.weather_refresh_seconds)
    if "inference_fps_measured" not in st.session_state:
        st.session_state["inference_fps_measured"] = 0.0
    if "show_fusion_popup" not in st.session_state:
        st.session_state["show_fusion_popup"] = False
    if "show_timeline_popup" not in st.session_state:
        st.session_state["show_timeline_popup"] = False
    if "show_system_popup" not in st.session_state:
        st.session_state["show_system_popup"] = False
    if "show_weather_popup" not in st.session_state:
        st.session_state["show_weather_popup"] = False
    if "ward_stop_processing" not in st.session_state:
        st.session_state["ward_stop_processing"] = False


_init_state()

session         = st.session_state["ward_session"]
events          = st.session_state["ward_events"]
temporal_engine = st.session_state["ward_temporal"]
weather_cache   = st.session_state["weather_cache"]

# ── Imports ────────────────────────────────────────────────────────────────
from ui.dashboard import (
    render_css, render_header, render_main_condition,
    CONDITION_COLORS, _condition_color,
)
from ui.fusion_panel import render_fusion_panel
from ui.timeline import render_timeline, render_condition_strip
from ui.charts import render_charts
from ui.weather_panel import render_weather_panel
from ui.system_status import (
    render_system_status, render_event_log, render_session_summary,
)

render_css()
render_header(cfg.ward_mode, provider.backend_name)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: run inference on one PIL image
# ═══════════════════════════════════════════════════════════════════════════
def process_one_image(image, timestamp_s: float = 0.0, filename: Optional[str] = None):
    from inference.provider import InferenceUnavailableError
    try:
        t0 = time.perf_counter()
        fusion  = provider.predict(image)
        temporal = temporal_engine.update(fusion)
        elapsed = time.perf_counter() - t0

        old_fps = st.session_state["inference_fps_measured"]
        new_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        st.session_state["inference_fps_measured"] = 0.7 * old_fps + 0.3 * new_fps

        session.record_frame(fusion, temporal, timestamp_s=timestamp_s, filename=filename)
        events.process_fusion(fusion, temporal.label)
        return fusion, temporal

    except InferenceUnavailableError as exc:
        st.error(f"⚠ Inference error: {exc}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Processing display — Inline card (renders inside the active layout column)
# ═══════════════════════════════════════════════════════════════════════════
class _ProcessingPopup:
    """
    Inline live analysis status card.
    Renders directly within the input/preview container so the live frame feed,
    progress bar, and real-time inference statistics are cleanly visible without
    blocking the rest of the dashboard or overlapping controls.
    """

    _LABEL_COLORS = {
        "wet": "#ef4444", "dry": "#22c55e",
        "damp": "#f59e0b", "drying": "#f97316",
    }

    def __init__(self, source_name: str):
        self._source = source_name
        self._slot   = st.empty()
        self._render(None, "—", 0.0, 0, 1, 0, 0.0)

    @staticmethod
    def _b64(pil_image) -> str:
        from io import BytesIO
        import base64
        buf  = BytesIO()
        thumb = pil_image.copy()
        thumb.thumbnail((640, 360))
        thumb.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()

    def _render(
        self,
        frame_b64: "str | None",
        label: str,
        confidence: float,
        frame_idx: int,
        total_frames: int,
        frames_dropped: int,
        elapsed: float,
    ) -> None:
        prog     = int(min(frame_idx / total_frames, 1.0) * 100) if total_frames > 0 else 0
        eta      = (elapsed / frame_idx * (total_frames - frame_idx)) if frame_idx > 0 else 0
        eta_s    = f"{eta:.0f}s" if eta > 0 else "—"
        lcolor   = self._LABEL_COLORS.get(label.lower(), "#94a3b8")
        drop_col = "#ef4444" if frames_dropped > 0 else "#6b7280"

        img_html = (
            f'<div style="border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);'
            f'background:#000;margin-bottom:0.75rem;">'
            f'<img src="data:image/jpeg;base64,{frame_b64}" '
            f'style="width:100%;max-height:340px;object-fit:contain;display:block;" />'
            f'</div>'
            if frame_b64 else
            '<div style="height:180px;background:rgba(255,255,255,0.02);'
            'border:1px solid rgba(255,255,255,0.05);border-radius:12px;display:flex;'
            'align-items:center;justify-content:center;color:#374151;'
            'font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;'
            'letter-spacing:0.15em;margin-bottom:0.75rem;">INITIALISING LIVE FEED…</div>'
        )

        self._slot.html(
            f"""<div style="background:rgba(10,13,18,0.92);border:1px solid rgba(34,197,94,0.3);
            border-radius:16px;padding:1.2rem;margin-bottom:1rem;
            box-shadow:0 12px 40px rgba(0,0,0,0.6);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
    <div style="display:flex;align-items:center;gap:0.5rem;overflow:hidden;">
      <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;
                  box-shadow:0 0 10px #22c55e;flex-shrink:0;"></div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:600;
                  color:#22c55e;letter-spacing:0.16em;text-transform:uppercase;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
        ANALYSING — {self._source}
      </div>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#475569;flex-shrink:0;">
      Frame {frame_idx} / {total_frames} ({prog}%)
    </div>
  </div>
  {img_html}
  <div style="height:6px;background:rgba(255,255,255,0.05);border-radius:6px;overflow:hidden;margin-bottom:0.85rem;">
    <div style="height:6px;width:{prog}%;background:linear-gradient(90deg,#22c55e,#16a34a);border-radius:6px;transition:width 0.2s;"></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.4rem;text-align:center;
              background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
              border-radius:10px;padding:0.6rem 0.4rem;">
    <div>
      <div style="font-size:0.52rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;">CONDITION</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:700;color:{lcolor};">{label.upper()}</div>
    </div>
    <div>
      <div style="font-size:0.52rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;">CONFIDENCE</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;color:#e2e8f0;">{int(confidence*100)}%</div>
    </div>
    <div>
      <div style="font-size:0.52rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;">DROPPED</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;color:{drop_col};">{frames_dropped}</div>
    </div>
    <div>
      <div style="font-size:0.52rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;">ETA</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;color:#e2e8f0;">{eta_s}</div>
    </div>
  </div>
</div>"""
        )

    def update(
        self,
        frame_image,
        label: str,
        confidence: float,
        frame_idx: int,
        total_frames: int,
        frames_dropped: int,
        elapsed: float,
    ) -> None:
        self._render(
            self._b64(frame_image),
            label, confidence,
            frame_idx, total_frames,
            frames_dropped, elapsed,
        )

    def complete(self, frames_processed: int, frames_dropped: int) -> None:
        self._slot.empty()

    def error(self, msg: str) -> None:
        self._slot.empty()
        st.error(msg)




# ═══════════════════════════════════════════════════════════════════════════
# st.dialog popups (Fusion / Timeline / System / Weather)
# ═══════════════════════════════════════════════════════════════════════════
@st.dialog("🔀 Model Fusion Detail", width="large")
def _popup_fusion():
    render_fusion_panel(session.current_fusion)


@st.dialog("⏱ Temporal Timeline", width="large")
def _popup_timeline():
    render_timeline(session.records)
    if len(session.records) > 1:
        render_condition_strip(session.records)


@st.dialog("⚙ System Status", width="large")
def _popup_system():
    col_a, col_b = st.columns(2)
    with col_a:
        render_system_status(
            session, cfg.ward_mode, provider.backend_name,
            inference_fps_measured=st.session_state["inference_fps_measured"],
        )
    with col_b:
        render_event_log(events.events)
        if events.events:
            if st.button("Clear Events", key="dlg_clear_events"):
                events.clear()
                st.rerun()


@st.dialog("🌤 Weather & Environment", width="large")
def _popup_weather():
    render_weather_panel(weather_cache)
    w = weather_cache.last_data
    if w and session.current_fusion:
        f = session.current_fusion
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:1rem;">
                <div class="section-header">ENVIRONMENT CONTEXT</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem;">
                    <div><div class="metric-label">Location</div>
                         <div style="font-size:0.8rem;color:#e2e8f0;">{w.location_name}</div></div>
                    <div><div class="metric-label">Temperature</div>
                         <div class="metric-value">{f"{w.temperature_c:.1f}°C" if w.temperature_c else "—"}</div></div>
                    <div><div class="metric-label">Humidity</div>
                         <div class="metric-value">{f"{int(w.humidity_pct)}%" if w.humidity_pct else "—"}</div></div>
                    <div><div class="metric-label">Rain</div>
                         <div class="metric-value">{f"{w.rain_mm:.1f} mm" if w.rain_mm else "0 mm"}</div></div>
                    <div><div class="metric-label">ML Condition</div>
                         <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                                     font-weight:700;color:#22c55e;">{f.top_label.upper()}</div></div>
                    <div><div class="metric-label">ML Confidence</div>
                         <div class="metric-value">{f.top_confidence:.0%}</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Open dialogs when buttons were clicked last render cycle
if st.session_state.get("show_fusion_popup"):
    st.session_state["show_fusion_popup"] = False
    _popup_fusion()

if st.session_state.get("show_timeline_popup"):
    st.session_state["show_timeline_popup"] = False
    _popup_timeline()

if st.session_state.get("show_system_popup"):
    st.session_state["show_system_popup"] = False
    _popup_system()

if st.session_state.get("show_weather_popup"):
    st.session_state["show_weather_popup"] = False
    _popup_weather()


# ═══════════════════════════════════════════════════════════════════════════
# Mini helpers (inline compact renderers for the unified dashboard)
# ═══════════════════════════════════════════════════════════════════════════
def _render_mini_event_log():
    border_colors = {
        "CONDITION_CHANGED":      "#22c55e",
        "HIGH_CONFIDENCE":        "#ef4444",
        "MODEL_DISAGREEMENT":     "#f59e0b",
        "PREDICTION_STABILIZED":  "#60a5fa",
        "AMBIGUOUS":              "#eab308",
        "ERROR":                  "#ef4444",
        "INFO":                   "#475569",
    }
    evts = events.events[-5:] if events.events else []
    if not evts:
        st.caption("No events yet.")
        return
    items = ""
    for evt in reversed(evts):
        bc = border_colors.get(evt.event_type.value, "#475569")
        items += (
            f'<div class="event-item" style="border-left-color:{bc};">'
            f'<div class="event-time">{evt.time_str}</div>'
            f'<div style="color:#e2e8f0;font-size:0.72rem;">{evt.message}</div>'
            f'</div>'
        )
    st.markdown(f'<div>{items}</div>', unsafe_allow_html=True)


def _render_mini_fusion(fusion):
    if fusion is None:
        st.caption("No prediction yet.")
        return

    orig    = fusion.original
    fine    = fusion.fine
    oc      = _condition_color(orig.top_label)
    fc      = _condition_color(fine.top_label)
    fused_c = _condition_color(fusion.top_label)
    agree_cls = "agree-yes" if fusion.agreement else "agree-no"
    agree_txt = "YES" if fusion.agreement else "NO"

    st.markdown(
        f"""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:0.9rem;">
            <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0.5rem;align-items:center;">
                <div style="text-align:center;">
                    <div style="font-size:0.55rem;color:#475569;letter-spacing:0.12em;
                                text-transform:uppercase;">ORIGINAL</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                                font-weight:700;color:{oc};">{orig.top_label.upper()}</div>
                    <div style="font-size:0.68rem;color:#94a3b8;">{orig.top_confidence:.0%}</div>
                </div>
                <div style="color:#30363d;font-size:1rem;">→</div>
                <div style="text-align:center;">
                    <div style="font-size:0.55rem;color:#475569;letter-spacing:0.12em;
                                text-transform:uppercase;">FINE-TUNED</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                                font-weight:700;color:{fc};">{fine.top_label.upper()}</div>
                    <div style="font-size:0.68rem;color:#94a3b8;">{fine.top_confidence:.0%}</div>
                </div>
            </div>
            <div style="margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid #1e2433;
                        display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:0.55rem;color:#475569;letter-spacing:0.12em;
                                text-transform:uppercase;margin-bottom:0.15rem;">FUSED</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                                font-weight:700;color:{fused_c};
                                text-shadow:0 0 12px {fused_c}60;">{fusion.top_label.upper()}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.62rem;color:#475569;">Agreement</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;"
                         class="{agree_cls}">{agree_txt}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_mini_weather():
    w = weather_cache.get()
    if w is None:
        st.caption("Enter a location in the Weather popup.")
        return
    temp     = f"{w.temperature_c:.1f}°C"   if w.temperature_c  is not None else "—"
    humidity = f"{int(w.humidity_pct)}%"     if w.humidity_pct   is not None else "—"
    wind     = f"{w.wind_speed_kmh:.0f} km/h" if w.wind_speed_kmh is not None else "—"
    st.markdown(
        f"""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:0.75rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                        color:#e2e8f0;margin-bottom:0.5rem;">📍 {w.location_name}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.4rem;">
                <div>
                    <div style="font-size:0.55rem;color:#475569;letter-spacing:0.12em;
                                text-transform:uppercase;">TEMP</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                font-weight:600;color:#60a5fa;">{temp}</div>
                </div>
                <div>
                    <div style="font-size:0.55rem;color:#475569;letter-spacing:0.12em;
                                text-transform:uppercase;">HUMIDITY</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                font-weight:600;color:#94a3b8;">{humidity}</div>
                </div>
                <div>
                    <div style="font-size:0.55rem;color:#475569;letter-spacing:0.12em;
                                text-transform:uppercase;">WIND</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                font-weight:600;color:#94a3b8;">{wind}</div>
                </div>
            </div>
            <div style="margin-top:0.4rem;font-size:0.6rem;color:#475569;font-style:italic;">
                {w.condition_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ██  MAIN UNIFIED LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
is_analyzing = (session.status.value == "ANALYZING")

left_col, right_col = st.columns([5, 7], gap="medium")

# ╔══════════════════════════════╗
# ║  LEFT COLUMN                 ║
# ╚══════════════════════════════╝
with left_col:

    # Condition hero card
    render_main_condition(session.current_fusion, session.current_temporal, is_analyzing)

    st.write("")  # spacer

    # ── Quick View popup buttons ────────────────────────────────────────
    st.markdown('<div class="section-header">QUICK VIEW</div>', unsafe_allow_html=True)
    qb1, qb2, qb3, qb4 = st.columns(4)
    with qb1:
        if st.button("🔀 Fusion", key="btn_fusion_popup", use_container_width=True):
            st.session_state["show_fusion_popup"] = True
            st.rerun()
    with qb2:
        if st.button("⏱ Timeline", key="btn_timeline_popup", use_container_width=True):
            st.session_state["show_timeline_popup"] = True
            st.rerun()
    with qb3:
        if st.button("⚙ System", key="btn_system_popup", use_container_width=True):
            st.session_state["show_system_popup"] = True
            st.rerun()
    with qb4:
        if st.button("🌤 Weather", key="btn_weather_popup", use_container_width=True):
            st.session_state["show_weather_popup"] = True
            st.rerun()

    st.write("")  # spacer

    # ── Fusion mini ──────────────────────────────────────────────────────
    fus_h, fus_b = st.columns([3, 1])
    with fus_h:
        st.markdown('<div class="section-header">MODEL FUSION</div>', unsafe_allow_html=True)
    with fus_b:
        if session.current_fusion:
            if st.button("Detail", key="btn_fus_detail", use_container_width=True):
                st.session_state["show_fusion_popup"] = True
                st.rerun()
    _render_mini_fusion(session.current_fusion)

    st.write("")  # spacer

    # ── Condition timeline strip ─────────────────────────────────────────
    if session.records and len(session.records) > 1:
        render_condition_strip(session.records)

    st.write("")  # spacer

    # ── Event log mini ───────────────────────────────────────────────────
    ev_h, ev_b = st.columns([3, 1])
    with ev_h:
        st.markdown('<div class="section-header">EVENTS</div>', unsafe_allow_html=True)
    with ev_b:
        if events.events:
            if st.button("View All", key="btn_evt_all", use_container_width=True):
                st.session_state["show_system_popup"] = True
                st.rerun()
    _render_mini_event_log()


# ╔══════════════════════════════╗
# ║  RIGHT COLUMN                ║
# ╚══════════════════════════════╝
with right_col:

    # ── Input controls ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">INPUT SOURCE</div>', unsafe_allow_html=True)

    from session.state import SessionStatus

    input_type = st.radio(
        "Input",
        ["Single Image", "Video File", "Frames Folder"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_type_radio",
    )

    st.markdown("---")

    # ── SINGLE IMAGE ─────────────────────────────────────────────────────
    if input_type == "Single Image":
        uploaded = st.file_uploader(
            "Upload a road image",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            from PIL import Image
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption=uploaded.name, use_container_width=True)

            btn_col, _ = st.columns([1, 2])
            with btn_col:
                if st.button("▶ Analyse Image", type="primary", use_container_width=True):
                    temporal_engine.reset()
                    session.reset()
                    events.clear()
                    session.start(uploaded.name, "image")
                    session.frames_read = 1

                    popup = _ProcessingPopup(uploaded.name)
                    popup.update(img, "analysing", 0.5, 0, 1, 0, 0.0)

                    fusion, temporal = process_one_image(img, filename=uploaded.name)
                    session.complete()

                    if fusion:
                        popup.complete(session.frames_processed, session.frames_dropped)
                    st.rerun()

    # ── VIDEO FILE ───────────────────────────────────────────────────────
    elif input_type == "Video File":
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="video_upload",
            label_visibility="collapsed",
        )

        if uploaded_video:
            st.video(uploaded_video)

            ctrl_start, ctrl_stop, ctrl_reset = st.columns([2, 1, 1])
            start_btn = ctrl_start.button(
                "▶ Start Analysis", type="primary", key="vid_start", use_container_width=True
            )
            stop_btn  = ctrl_stop.button("■ Stop",  key="vid_stop",  use_container_width=True)
            reset_btn = ctrl_reset.button("↺ Reset", key="vid_reset", use_container_width=True)

            if reset_btn:
                temporal_engine.reset()
                session.reset()
                events.clear()
                st.session_state["ward_stop_processing"] = False
                st.rerun()

            if stop_btn:
                st.session_state["ward_stop_processing"] = True

            if start_btn:
                from input.video_reader import VideoReader, VideoReadError

                temporal_engine.reset()
                session.reset()
                events.clear()
                st.session_state["ward_stop_processing"] = False
                session.start(uploaded_video.name, "video")

                frame_q: queue.Queue = queue.Queue(maxsize=cfg.frame_queue_size)
                dropped   = 0
                video_data = uploaded_video.getvalue()
                reader    = VideoReader(source=video_data, target_fps=cfg.inference_fps)

                popup = _ProcessingPopup(uploaded_video.name)
                t_start = time.perf_counter()

                try:
                    for vf in reader.frames():
                        if st.session_state.get("ward_stop_processing"):
                            break

                        session.frames_read += 1

                        if frame_q.full():
                            try:
                                frame_q.get_nowait()
                                dropped += 1
                                session.frames_dropped = dropped
                            except queue.Empty:
                                pass
                        frame_q.put_nowait(vf)

                        try:
                            frame = frame_q.get_nowait()
                        except queue.Empty:
                            continue

                        fusion, temporal = process_one_image(
                            frame.image, timestamp_s=frame.timestamp_s,
                        )
                        if fusion is None:
                            continue

                        elapsed = time.perf_counter() - t_start
                        popup.update(
                            frame_image    = frame.image,
                            label          = fusion.top_label,
                            confidence     = fusion.top_confidence,
                            frame_idx      = session.frames_read,
                            total_frames   = frame.total_frames or session.frames_read,
                            frames_dropped = session.frames_dropped,
                            elapsed        = elapsed,
                        )

                except VideoReadError as exc:
                    popup.error(f"Video error: {exc}")

                session.complete()
                popup.complete(session.frames_processed, session.frames_dropped)
                st.rerun()

    # ── FRAMES FOLDER ────────────────────────────────────────────────────
    elif input_type == "Frames Folder":
        st.info("Select multiple frame images to process as a sequence.")
        uploaded_frames = st.file_uploader(
            "Upload frame images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="frames_upload",
            label_visibility="collapsed",
        )

        if uploaded_frames:
            st.write(f"**{len(uploaded_frames)}** frames selected")

            ctrl2_start, ctrl2_reset = st.columns([2, 1])
            start_btn2 = ctrl2_start.button(
                "▶ Process Frames", type="primary", key="frames_start", use_container_width=True
            )
            reset_btn2 = ctrl2_reset.button("↺ Reset", key="frames_reset", use_container_width=True)

            if reset_btn2:
                temporal_engine.reset()
                session.reset()
                events.clear()
                st.rerun()

            if start_btn2:
                from natsort import natsorted
                from PIL import Image, UnidentifiedImageError

                temporal_engine.reset()
                session.reset()
                events.clear()

                sorted_frames = natsorted(uploaded_frames, key=lambda f: f.name)
                session.start(f"{len(sorted_frames)} frames", "frames")
                session.frames_read = len(sorted_frames)

                popup2  = _ProcessingPopup(f"{len(sorted_frames)} frames")
                t_start2 = time.perf_counter()

                for i, uf in enumerate(sorted_frames):
                    try:
                        img = Image.open(uf).convert("RGB")
                    except Exception as exc:
                        st.warning(f"Skipping {uf.name}: {exc}")
                        continue

                    fusion, temporal = process_one_image(
                        img, filename=uf.name, timestamp_s=float(i),
                    )
                    if fusion is None:
                        continue

                    elapsed2 = time.perf_counter() - t_start2
                    popup2.update(
                        frame_image    = img,
                        label          = fusion.top_label,
                        confidence     = fusion.top_confidence,
                        frame_idx      = i + 1,
                        total_frames   = len(sorted_frames),
                        frames_dropped = 0,
                        elapsed        = elapsed2,
                    )

                session.complete()
                popup2.complete(session.frames_processed, 0)
                st.rerun()

    st.write("")  # spacer

    # ── Weather mini ─────────────────────────────────────────────────────
    wx_h, wx_b = st.columns([3, 1])
    with wx_h:
        st.markdown('<div class="section-header">ENVIRONMENT</div>', unsafe_allow_html=True)
    with wx_b:
        if st.button("Full View", key="btn_wx_full", use_container_width=True):
            st.session_state["show_weather_popup"] = True
            st.rerun()
    _render_mini_weather()

    st.write("")  # spacer

    # ── Session summary + CSV export ─────────────────────────────────────
    if session.status.value == "COMPLETE":
        render_session_summary(session)
        st.write("")
        csv_bytes = "\n".join(session.to_csv_lines()).encode("utf-8")
        st.download_button(
            "⬇ Export CSV",
            data=csv_bytes,
            file_name="ward_session.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# CHART GRID — full width, below both columns
# Each chart has an ⤢ Expand button to open it as a popup
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
render_charts(session.records)
