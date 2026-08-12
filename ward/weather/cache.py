"""
WARD — Weather Cache
=====================
Wraps the weather client with a timed cache.
- Returns cached data when within refresh interval.
- On failure, returns last successful data (stale fallback).
- If no data has ever been fetched, returns None.
- Weather failure never propagates to inference.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from weather.client import WeatherData, WeatherUnavailableError, fetch_weather_by_name

logger = logging.getLogger(__name__)


class WeatherCache:
    """
    Timed cache for weather data.

    Parameters
    ----------
    refresh_seconds : how often to call the API (default 300 s = 5 min)
    """

    def __init__(self, refresh_seconds: int = 300) -> None:
        self.refresh_seconds = refresh_seconds
        self._last_data: Optional[WeatherData] = None
        self._last_fetch_time: float = 0.0
        self._location: Optional[str] = None
        self._error: Optional[str] = None

    @property
    def last_data(self) -> Optional[WeatherData]:
        return self._last_data

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    @property
    def last_updated_seconds_ago(self) -> Optional[float]:
        if self._last_fetch_time == 0.0:
            return None
        return time.time() - self._last_fetch_time

    def set_location(self, location: str) -> None:
        """Change location — forces refresh on next get()."""
        if location != self._location:
            self._location = location
            self._last_fetch_time = 0.0   # force refresh

    def get(self) -> Optional[WeatherData]:
        """
        Return current weather data, fetching from API if stale.
        Returns None if no data has ever been successfully fetched.
        Never raises.
        """
        if not self._location:
            return None

        age = time.time() - self._last_fetch_time
        if age < self.refresh_seconds and self._last_data is not None:
            return self._last_data

        try:
            data = fetch_weather_by_name(self._location)
            self._last_data = data
            self._last_fetch_time = time.time()
            self._error = None
            logger.debug("Weather refreshed for %s", self._location)
        except WeatherUnavailableError as exc:
            self._error = str(exc)
            logger.warning("Weather fetch failed: %s — using cached data", exc)

        return self._last_data

    def invalidate(self) -> None:
        """Force a refresh on the next get() call."""
        self._last_fetch_time = 0.0
