"""Baselines.

* Seasonal naive needs no code here: ``lag_day`` (same hour, latest available day) and
  ``lag_week`` (same hour last week) are already columns produced by ``features.build_supervised``.
* SARIMA: fitted once per fold on the last ``fit_days`` before the test period, then the fitted
  parameters are re-applied to a sliding window ending at each forecast origin (no re-fit per
  origin - that would take hours).
* Prophet: fitted once per fold; its forecast does not depend on the origin (pure extrapolation of
  trend + seasonality + holidays + temperature), so it is identical across horizons by design.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .config import TZ

log = logging.getLogger(__name__)


def sarima_forecasts(
    load: pd.Series,
    train_end: pd.Timestamp,
    origins: pd.DatetimeIndex,
    max_horizon: int,
    fit_days: int = 56,
    window_days: int = 14,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 24),
) -> dict[pd.Timestamp, np.ndarray]:
    """Return {origin: array of ``max_horizon`` forecasts}. Needs ``statsmodels``."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit_y = load.loc[train_end - pd.Timedelta(days=fit_days): train_end - pd.Timedelta(hours=1)]
    y = fit_y.interpolate(limit_direction="both").to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)

    out = {}
    for t in origins:
        w = load.loc[t - pd.Timedelta(days=window_days) + pd.Timedelta(hours=1): t]
        w = w.interpolate(limit_direction="both").to_numpy(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[t] = np.asarray(res.apply(w, refit=False).forecast(max_horizon))
        except Exception as exc:  # noqa: BLE001
            log.warning("SARIMA failed at origin %s: %s", t, exc)
            out[t] = np.full(max_horizon, np.nan)
    return out


def prophet_forecast(
    df: pd.DataFrame, train_end: pd.Timestamp, target_times: pd.DatetimeIndex, train_years: int = 3
) -> np.ndarray:
    """Fit Prophet on data before ``train_end`` and predict ``target_times``. Needs ``prophet``.

    Prophet works on *local wall-clock time* so that the daily profile does not shift by one hour
    when DST changes. The duplicated autumn hour is dropped (first occurrence kept) for fitting; the
    spring gap is simply missing.
    """
    import logging as _logging

    from prophet import Prophet

    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
    _logging.getLogger("prophet").setLevel(_logging.WARNING)

    to_local = lambda ix: ix.tz_convert(TZ).tz_localize(None)  # noqa: E731
    train = df.loc[: train_end - pd.Timedelta(hours=1)]
    train = train.loc[train_end - pd.DateOffset(years=train_years):]
    d = pd.DataFrame({"ds": to_local(train.index), "y": train["load_mw"].to_numpy(),
                      "temp": train["temp"].to_numpy() if "temp" in train else 0.0})
    d = d.dropna().drop_duplicates("ds", keep="first")

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                uncertainty_samples=0)
    m.add_country_holidays("DE")
    m.add_regressor("temp")
    m.fit(d)

    tgt = pd.DataFrame({"ds": to_local(target_times)})
    tgt["temp"] = df["temp"].reindex(target_times).ffill().bfill().to_numpy() if "temp" in df else 0.0
    pred = m.predict(tgt.drop_duplicates("ds"))[["ds", "yhat"]]
    return tgt.merge(pred, on="ds", how="left")["yhat"].to_numpy()
