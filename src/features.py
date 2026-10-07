"""Feature engineering for the *direct multi-horizon* set-up.

One training row = (forecast origin ``t``, horizon ``h``) -> predict load at ``T = t + h``.

Leakage rules (enforced by tests/test_features.py):
  * load-derived features may only use values at or before the origin ``t``;
  * calendar features of ``T`` are always known in advance;
  * temperature at ``T`` is treated as known (= a perfect weather forecast, see README).

Because the lag "same hour yesterday" (T-24) is only known at the origin if h <= 24, we use the
*latest available same-hour value*: ``lag_day = load[T - 24*ceil(h/24)]``. ``lag_week = load[T-168]``
is known for every h <= 168.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import TZ

# Population (millions) of the 16 Laender - used to weight regional holidays.
STATE_POP = {"BW": 11.2, "BY": 13.4, "BE": 3.7, "BB": 2.6, "HB": 0.68, "HH": 1.9, "HE": 6.4,
             "MV": 1.6, "NI": 8.1, "NW": 18.1, "RP": 4.1, "SL": 1.0, "SN": 4.1, "ST": 2.2,
             "SH": 2.9, "TH": 2.1}


def _holiday_share_by_day(days: pd.DatetimeIndex) -> pd.Series:
    """Fraction of the German population living in a state with a public holiday on that day.
    1.0 = nationwide holiday (e.g. 3 Oct), ~0.3 = regional (e.g. Corpus Christi)."""
    import holidays  # imported lazily so the rest of the package works without it

    years = sorted(set(days.year))
    total = sum(STATE_POP.values())
    share = pd.Series(0.0, index=days)
    for state, pop in STATE_POP.items():
        keys = set(holidays.Germany(subdiv=state, years=years).keys())
        share += pd.Series(days.date).isin(keys).to_numpy() * (pop / total)
    return share


def calendar_features(idx_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features in *local* (Europe/Berlin) time, indexed like ``idx_utc``."""
    loc = idx_utc.tz_convert(TZ)
    day_local = pd.DatetimeIndex(loc.normalize().tz_localize(None))
    all_days = pd.date_range(day_local.min() - pd.Timedelta(days=2),
                             day_local.max() + pd.Timedelta(days=2), freq="D")

    share = _holiday_share_by_day(all_days)
    dow = pd.Series(all_days.dayofweek, index=all_days)
    is_hol = share >= 0.5
    bridge = (dow < 5) & ~is_hol & (((dow == 4) & is_hol.shift(1, fill_value=False))
                                    | ((dow == 0) & is_hol.shift(-1, fill_value=False)))
    xmas = ((all_days.month == 12) & (all_days.day >= 24)) | ((all_days.month == 1) & (all_days.day == 1))
    daily = pd.DataFrame({
        "hol_share": share,
        "hol_prev": share.shift(1, fill_value=0.0),
        "hol_next": share.shift(-1, fill_value=0.0),
        "is_bridge": bridge.astype(float),
        "is_xmas": xmas.astype(float),
    }, index=all_days)
    d = daily.reindex(day_local)
    d.index = idx_utc

    hour, dow_h, doy = loc.hour.to_numpy(), loc.dayofweek.to_numpy(), loc.dayofyear.to_numpy()
    base = pd.DataFrame({
        "hour": hour, "dow": dow_h, "month": loc.month.to_numpy(), "doy": doy,
        "is_weekend": (dow_h >= 5).astype(float),
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=idx_utc)
    return pd.concat([base, d], axis=1)


def build_supervised(
    df: pd.DataFrame, horizons: list[int], origins: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Return a long DataFrame, one row per (origin, horizon).

    Parameters
    ----------
    df : hourly UTC index, columns ``load_mw`` and (optionally) ``temp``.
    horizons : hours ahead to build rows for.
    origins : restrict to these forecast origins (all rolling stats are still computed on the
              full history, so restricting never changes feature values).

    Columns: features..., ``ref`` (seasonal reference), ``horizon``, ``target_time``, ``y``. Index = origin.
    """
    load = df["load_mw"]
    use_temp = "temp" in df and df["temp"].notna().any()
    idx = load.index
    sel = idx if origins is None else idx.intersection(pd.DatetimeIndex(origins))

    roll24 = load.rolling(24, min_periods=20).mean()
    # "normal" load for each hour = median of the same hour in the previous 4 weeks; the *deviation*
    # from that norm is very persistent over the next hours -> key short-horizon signal
    prev4 = pd.concat([load.shift(168 * k) for k in (1, 2, 3, 4)], axis=1)
    norm_t = prev4.median(axis=1).where(prev4.notna().sum(axis=1) >= 3)
    dev = load - norm_t
    origin_feats = pd.DataFrame({
        "dev_t": dev,
        "dev_mean3": dev.rolling(3, min_periods=2).mean(),
        "dev_mean24": dev.rolling(24, min_periods=18).mean(),
        "diff1": load.diff(1),
        "diff3": load.diff(3),
        "load_t_m3": load.shift(3),
        "load_t": load,                                   # most recent observation
        "load_t_m1": load.shift(1),
        "roll24": roll24,                                 # rolling means over the last day / week
        "roll168": load.rolling(168, min_periods=140).mean(),
        "std24": load.rolling(24, min_periods=20).std(),
        "level_ratio": roll24 / roll24.shift(168),        # is this week running above/below last week?
    })

    maxh = max(horizons)
    ext = pd.date_range(idx.min(), idx.max() + pd.Timedelta(hours=maxh), freq="h")
    cal_all = calendar_features(ext)

    if use_temp:
        temp = df["temp"]
        t_roll24 = temp.rolling(24, min_periods=18).mean()
        t_roll72 = temp.rolling(72, min_periods=60).mean()

    parts = []
    for h in horizons:
        f = origin_feats.copy()
        f["horizon"] = h
        l_day = 24 * math.ceil(h / 24)
        f["lag_day"] = load.shift(l_day - h)              # load[T - 24*ceil(h/24)]
        f["lag_week"] = load.shift(168 - h)               # load[T - 168]
        f["lag_2week"] = load.shift(336 - h)              # load[T - 336]
        same_hour = pd.concat([load.shift(168 * k - h) for k in (1, 2, 3, 4)], axis=1)
        f["same_hour_med_4w"] = same_hour.median(axis=1).where(same_hour.notna().sum(axis=1) >= 3)
        f["ref"] = f["same_hour_med_4w"]                  # the model predicts y - ref (not a feature)

        if use_temp:                                      # temperature around the *target* time T
            f["temp"] = temp.shift(-h)
            f["temp_roll24"] = t_roll24.shift(-h)
            f["temp_roll72"] = t_roll72.shift(-h)
            f["temp_delta24"] = temp.shift(-h) - temp.shift(24 - h)
            f["hdd"] = (15.0 - f["temp"]).clip(lower=0)
            f["cdd"] = (f["temp"] - 22.0).clip(lower=0)

        target = idx + pd.Timedelta(hours=h)
        cal = cal_all.reindex(target)
        cal.index = idx
        f = pd.concat([f, cal], axis=1)
        f["target_time"] = target
        f["y"] = load.shift(-h)
        parts.append(f.loc[sel])

    out = pd.concat(parts)
    float_cols = [c for c in out.columns if c not in ("y", "target_time")]  # incl. ref
    out[float_cols] = out[float_cols].astype("float32")
    out.index.name = "origin"
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("y", "y_res", "ref", "target_time")]
