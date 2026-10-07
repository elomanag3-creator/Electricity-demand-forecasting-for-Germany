"""Synthetic load + temperature, used for unit tests and offline dry-runs of the pipeline.

The numbers are NOT real German data - never report results from this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TZ


def make_synthetic(start: str = "2021-01-01", end: str = "2023-12-31", seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC", name="timestamp")
    loc = idx.tz_convert(TZ)
    rng = np.random.default_rng(seed)
    n = len(idx)
    hour, dow, doy = loc.hour.to_numpy(), loc.dayofweek.to_numpy(), loc.dayofyear.to_numpy()

    # temperature: yearly + daily cycle + AR(1) weather noise
    noise = np.zeros(n)
    eps = rng.normal(0, 0.6, n)
    for i in range(1, n):
        noise[i] = 0.985 * noise[i - 1] + eps[i]
    temp = 9.5 - 10 * np.cos(2 * np.pi * (doy - 15) / 365.25) + 3 * np.sin(2 * np.pi * (hour - 9) / 24) + noise

    daily = (9000 * np.exp(-((hour - 11) ** 2) / 18) + 7000 * np.exp(-((hour - 19) ** 2) / 8)
             - 7000 * np.exp(-((hour - 3) ** 2) / 10))
    weekend = np.where(dow == 5, -3500, np.where(dow == 6, -6500, 0))
    heating = 450 * np.clip(15 - temp, 0, None)
    cooling = 150 * np.clip(temp - 23, 0, None)

    month, day = loc.month.to_numpy(), loc.day.to_numpy()
    fixed_hol = ((month == 1) & (day == 1)) | ((month == 5) & (day == 1)) | ((month == 10) & (day == 3)) \
        | ((month == 12) & np.isin(day, [25, 26]))
    xmas = ((month == 12) & (day >= 24)) | ((month == 1) & (day == 1))
    holiday = np.where(fixed_hol, -8000, 0) + np.where(xmas, -3000, 0)

    load = 50000 + daily + weekend + heating + cooling + holiday + rng.normal(0, 0.012, n) * 50000
    load = pd.Series(load, index=idx, name="load_mw")
    # carve out two gaps: one short (gets interpolated) and one long (stays NaN)
    load.iloc[1000:1003] = np.nan
    load.iloc[5000:5030] = np.nan
    return pd.DataFrame({"load_mw": load, "temp": pd.Series(temp, index=idx)})


def _german(v: float) -> str:
    if np.isnan(v):
        return "-"
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def write_smard_style_csv(load_utc: pd.Series, path, quarter_hour: bool = False) -> None:
    """Write ``load_utc`` the way the SMARD download centre does: local wall-clock labels,
    German number format, ';' separator (hence duplicated/missing labels around DST changes)."""
    s = load_utc
    if quarter_hour:
        idx = pd.date_range(s.index[0], s.index[-1] + pd.Timedelta(minutes=45), freq="15min")
        s = (s / 4).reindex(s.index.repeat(4)).set_axis(idx)
        step = pd.Timedelta(minutes=15)
    else:
        step = pd.Timedelta(hours=1)
    fmt = "%d.%m.%Y %H:%M"
    start_lbl = s.index.tz_convert(TZ).strftime(fmt)
    end_lbl = (s.index + step).tz_convert(TZ).strftime(fmt)
    lines = ["Datum von;Datum bis;Netzlast [MWh] Originalauflösungen;Residuallast [MWh] Originalauflösungen"]
    for a, b, v in zip(start_lbl, end_lbl, s.to_numpy()):
        lines.append(f"{a};{b};{_german(v)};{_german(v * 0.6)}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
