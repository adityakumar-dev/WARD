"""
WARD — Open-Meteo Weather Client
==================================
Fetches current weather from the free Open-Meteo API (no API key required).
Supports location lookup by city name or lat/lon directly.

Weather is purely contextual — it NEVER overrides ML predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 8  # seconds


class WeatherUnavailableError(RuntimeError):
    """Raised when weather data cannot be retrieved."""


@dataclass
class WeatherData:
    location_name: str
    latitude: float
    longitude: float
    temperature_c: Optional[float]
    humidity_pct: Optional[float]
    precipitation_mm: Optional[float]
    rain_mm: Optional[float]
    showers_mm: Optional[float]
    snowfall_cm: Optional[float]
    weather_code: Optional[int]
    cloud_cover_pct: Optional[float]
    wind_speed_kmh: Optional[float]
    condition_text: str   # human-readable description of weather_code


# WMO Weather Interpretation Codes → human-readable description
_WMO_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ heavy hail",
}


def _wmo_description(code: Optional[int]) -> str:
    if code is None:
        return "Unknown"
    return _WMO_DESCRIPTIONS.get(code, f"Code {code}")


def geocode(location_name: str) -> Tuple[float, float, str]:
    """
    Resolve a city/place name to (latitude, longitude, resolved_name).
    Raises WeatherUnavailableError on failure.
    """
    try:
        resp = requests.get(
            _GEOCODE_URL,
            params={"name": location_name, "count": 1, "language": "en", "format": "json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results")
        if not results:
            raise WeatherUnavailableError(f"Location not found: {location_name!r}")
        r = results[0]
        name = f"{r.get('name', location_name)}, {r.get('country', '')}"
        return float(r["latitude"]), float(r["longitude"]), name.strip(", ")
    except WeatherUnavailableError:
        raise
    except Exception as exc:
        raise WeatherUnavailableError(f"Geocoding failed: {exc}") from exc


def fetch_weather(
    latitude: float,
    longitude: float,
    location_name: str = "Unknown",
) -> WeatherData:
    """
    Fetch current weather for a lat/lon pair.
    Raises WeatherUnavailableError on failure.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
        ]),
        "timezone": "auto",
    }
    try:
        resp = requests.get(_WEATHER_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})

        def _get(key: str) -> Optional[float]:
            v = current.get(key)
            return float(v) if v is not None else None

        wmo = current.get("weather_code")
        return WeatherData(
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            temperature_c=_get("temperature_2m"),
            humidity_pct=_get("relative_humidity_2m"),
            precipitation_mm=_get("precipitation"),
            rain_mm=_get("rain"),
            showers_mm=_get("showers"),
            snowfall_cm=_get("snowfall"),
            weather_code=int(wmo) if wmo is not None else None,
            cloud_cover_pct=_get("cloud_cover"),
            wind_speed_kmh=_get("wind_speed_10m"),
            condition_text=_wmo_description(int(wmo) if wmo is not None else None),
        )
    except WeatherUnavailableError:
        raise
    except Exception as exc:
        raise WeatherUnavailableError(f"Weather fetch failed: {exc}") from exc


def fetch_weather_by_name(location_name: str) -> WeatherData:
    """Convenience: geocode then fetch weather."""
    lat, lon, resolved = geocode(location_name)
    return fetch_weather(lat, lon, resolved)
