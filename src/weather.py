"""Population-weighted air temperature for Germany from Open-Meteo (free, no key).

IMPORTANT (leakage caveat): the archive API returns *observed* (reanalysis) temperature. Using
it for the target hour assumes a *perfect* weather forecast. Real day-ahead errors are ~1-2 K, so
live accuracy would be a bit worse. See README -> "Limitations" for the honest-evaluation
alternative (Open-Meteo "historical forecast" API).
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from .config import CITIES, DATA_RAW

log = logging.getLogger(__name__)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_temperature(start: str, end: str, cities: dict = CITIES) -> pd.Series:
    """Hourly 2 m temperature (deg C), population-weighted over ``cities``, indexed in UTC."""
    start_d, end_d = pd.Timestamp(start), pd.Timestamp(end)
    cache = DATA_RAW / f"weather_temp_{start_d:%Y%m%d}_{end_d:%Y%m%d}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True)["temp"]
        s.index = pd.to_datetime(s.index, utc=True)
        return s

    per_city, weights = {}, {}
    for name, (lat, lon, w) in cities.items():
        parts = []
        for year in range(start_d.year, end_d.year + 1):   # one request per year keeps payloads small
            a = max(start_d, pd.Timestamp(f"{year}-01-01"))
            b = min(end_d, pd.Timestamp(f"{year}-12-31"))
            r = requests.get(
                ARCHIVE_URL,
                params={"latitude": lat, "longitude": lon, "start_date": f"{a:%Y-%m-%d}",
                        "end_date": f"{b:%Y-%m-%d}", "hourly": "temperature_2m", "timezone": "UTC"},
                timeout=60,
            )
            r.raise_for_status()
            h = r.json()["hourly"]
            parts.append(pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"], utc=True),
                                   dtype=float))
            time.sleep(0.4)  # be polite to the free API
        per_city[name] = pd.concat(parts)
        weights[name] = w
        log.info("Weather: %s done", name)

    df, wv = pd.DataFrame(per_city), pd.Series(weights)
    temp = df.mul(wv).sum(axis=1) / df.notna().mul(wv).sum(axis=1)   # renormalise if a city is NaN
    temp = temp.rename("temp")
    temp.index.name = "timestamp"
    temp.to_frame().to_csv(cache)
    return temp
