"""Tests for weather client and cache."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

from weather.client import WeatherData, WeatherUnavailableError, fetch_weather
from weather.cache import WeatherCache


# ── Weather client ────────────────────────────────────────────────────────────
class TestWeatherClient:
    def _mock_response(self):
        return {
            "current": {
                "temperature_2m": 24.5,
                "relative_humidity_2m": 78.0,
                "precipitation": 1.2,
                "rain": 0.8,
                "showers": 0.0,
                "snowfall": 0.0,
                "weather_code": 63,
                "cloud_cover": 80.0,
                "wind_speed_10m": 12.0,
            }
        }

    def test_successful_fetch(self):
        with patch("weather.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = self._mock_response()
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            data = fetch_weather(30.0, 78.0, "Dehradun")
            assert data.temperature_c == pytest.approx(24.5)
            assert data.humidity_pct == pytest.approx(78.0)
            assert data.rain_mm == pytest.approx(0.8)
            assert "rain" in data.condition_text.lower() or data.weather_code == 63

    def test_http_error_raises(self):
        import requests as req
        with patch("weather.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError("404")
            mock_get.return_value = mock_resp

            with pytest.raises(WeatherUnavailableError):
                fetch_weather(30.0, 78.0, "Test")

    def test_connection_error_raises(self):
        import requests as req
        with patch("weather.client.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.ConnectionError("offline")
            with pytest.raises(WeatherUnavailableError):
                fetch_weather(30.0, 78.0, "Test")


# ── Weather cache ─────────────────────────────────────────────────────────────
class TestWeatherCache:
    def _good_data(self):
        return WeatherData(
            location_name="Test", latitude=30.0, longitude=78.0,
            temperature_c=20.0, humidity_pct=60.0, precipitation_mm=0.0,
            rain_mm=0.0, showers_mm=0.0, snowfall_cm=0.0, weather_code=0,
            cloud_cover_pct=10.0, wind_speed_kmh=5.0, condition_text="Clear sky"
        )

    def test_returns_none_without_location(self):
        cache = WeatherCache(refresh_seconds=300)
        assert cache.get() is None

    def test_returns_data_on_success(self):
        cache = WeatherCache(refresh_seconds=300)
        cache.set_location("Dehradun")
        good_data = self._good_data()
        with patch("weather.cache.fetch_weather_by_name", return_value=good_data):
            result = cache.get()
        assert result is not None
        assert result.location_name == "Test"

    def test_returns_stale_on_failure(self):
        cache = WeatherCache(refresh_seconds=300)
        cache.set_location("Dehradun")
        good_data = self._good_data()
        with patch("weather.cache.fetch_weather_by_name", return_value=good_data):
            cache.get()  # populate

        # Force refresh
        cache._last_fetch_time = 0.0
        with patch("weather.cache.fetch_weather_by_name", side_effect=WeatherUnavailableError("offline")):
            result = cache.get()  # should return stale

        assert result is not None
        assert result.location_name == "Test"

    def test_returns_none_if_no_previous_on_failure(self):
        cache = WeatherCache(refresh_seconds=300)
        cache.set_location("Nowhere")
        with patch("weather.cache.fetch_weather_by_name", side_effect=WeatherUnavailableError("fail")):
            result = cache.get()
        assert result is None

    def test_does_not_refresh_within_ttl(self):
        cache = WeatherCache(refresh_seconds=3600)
        cache.set_location("Test")
        good_data = self._good_data()
        with patch("weather.cache.fetch_weather_by_name", return_value=good_data) as mock_fn:
            cache.get()
            cache.get()  # should NOT call API again
            assert mock_fn.call_count == 1
