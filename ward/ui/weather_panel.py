"""
WARD — Weather Context Panel
==============================
Renders the weather context card.
Weather is displayed as environmental information only.
It never overrides the ML prediction.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from weather.client import WeatherData
from weather.cache import WeatherCache


def render_weather_panel(cache: WeatherCache) -> None:
    """Render the weather input + context card."""
    st.markdown('<div class="section-header">ENVIRONMENT</div>', unsafe_allow_html=True)

    location_input = st.text_input(
        "Location",
        value=st.session_state.get("weather_location", ""),
        placeholder="e.g. Dehradun, London, New Delhi",
        key="weather_location_input",
        label_visibility="collapsed",
    )

    if location_input and location_input != st.session_state.get("weather_location", ""):
        st.session_state["weather_location"] = location_input
        cache.set_location(location_input)

    data: Optional[WeatherData] = cache.get()

    if data is None:
        if cache.last_error:
            st.markdown(
                f'<div style="color:#ef4444;font-size:0.75rem;padding:0.5rem;">⚠ {cache.last_error}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color:#475569;font-size:0.75rem;padding:0.5rem;">Enter a location above to see weather context.</div>',
                unsafe_allow_html=True,
            )
        return

    updated_ago = cache.last_updated_seconds_ago
    if updated_ago is not None:
        if updated_ago < 60:
            updated_txt = f"{int(updated_ago)}s ago"
        else:
            updated_txt = f"{int(updated_ago / 60)} min ago"
    else:
        updated_txt = "—"

    temp = f"{data.temperature_c:.1f}°C" if data.temperature_c is not None else "—"
    humidity = f"{int(data.humidity_pct)}%" if data.humidity_pct is not None else "—"
    precip = f"{data.precipitation_mm:.1f} mm" if data.precipitation_mm is not None else "—"
    rain = f"{data.rain_mm:.1f} mm" if data.rain_mm is not None else "—"
    wind = f"{data.wind_speed_kmh:.0f} km/h" if data.wind_speed_kmh is not None else "—"
    cloud = f"{int(data.cloud_cover_pct)}%" if data.cloud_cover_pct is not None else "—"
    snow = f"{data.snowfall_cm:.1f} cm" if (data.snowfall_cm is not None and data.snowfall_cm > 0) else "—"

    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom:0.5rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;color:#e2e8f0;">
                    📍 {data.location_name}
                </div>
                <div style="font-size:0.6rem;color:#475569;">Updated {updated_txt}</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                <div>
                    <div class="metric-label">Temperature</div>
                    <div class="metric-value">{temp}</div>
                </div>
                <div>
                    <div class="metric-label">Humidity</div>
                    <div class="metric-value">{humidity}</div>
                </div>
                <div>
                    <div class="metric-label">Precipitation</div>
                    <div class="metric-value">{precip}</div>
                </div>
                <div>
                    <div class="metric-label">Rain</div>
                    <div class="metric-value">{rain}</div>
                </div>
                <div>
                    <div class="metric-label">Wind</div>
                    <div class="metric-value">{wind}</div>
                </div>
                <div>
                    <div class="metric-label">Cloud Cover</div>
                    <div class="metric-value">{cloud}</div>
                </div>
                <div style="grid-column:span 2;">
                    <div class="metric-label">Condition</div>
                    <div style="font-size:0.8rem;color:#e2e8f0;margin-top:0.15rem;">{data.condition_text}</div>
                </div>
            </div>
            <div style="margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid #1e2433;font-size:0.65rem;color:#475569;font-style:italic;">
                ⚡ Weather is contextual only — ML prediction is authoritative.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
