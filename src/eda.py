"""Exploratory plots: daily / weekly / yearly seasonality and autocorrelation."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGS, TZ  # noqa: E402

log = logging.getLogger(__name__)


def plot_seasonality(load: pd.Series, out_dir: Path = FIGS) -> Path:
    """2x2 panel: daily profile, weekly profile, yearly profile, autocorrelation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gw = (load / 1000.0).dropna()
    loc = gw.index.tz_convert(TZ)            # seasonality is in *local* time
    df = pd.DataFrame({"gw": gw.to_numpy(), "hour": loc.hour, "dow": loc.dayofweek,
                       "month": loc.month, "year": loc.year}, index=gw.index)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (a) daily: mean by hour, split weekday / Saturday / Sunday
    a = ax[0, 0]
    for name, mask in {"Mon-Fri": df.dow < 5, "Saturday": df.dow == 5, "Sunday": df.dow == 6}.items():
        a.plot(df[mask].groupby("hour").gw.mean(), label=name, lw=2)
    a.set(title="Daily seasonality (local time)", xlabel="Hour of day", ylabel="Load [GW]")
    a.legend()

    # (b) weekly: 168-hour profile
    b = ax[0, 1]
    wk = df.groupby(["dow", "hour"]).gw.mean().to_numpy()
    b.plot(wk, lw=1.8, color="tab:purple")
    b.set_xticks(np.arange(12, 168, 24))
    b.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    b.set(title="Weekly seasonality (mean load by hour of week)", ylabel="Load [GW]")
    for x in range(24, 168, 24):
        b.axvline(x, color="grey", lw=0.5, alpha=0.5)

    # (c) yearly: monthly mean for each year
    c = ax[1, 0]
    pivot = df.groupby(["year", "month"]).gw.mean().unstack(0)
    pivot.plot(ax=c, marker="o", lw=1.5)
    c.set(title="Yearly seasonality (monthly mean)", xlabel="Month", ylabel="Load [GW]")
    c.set_xticks(range(1, 13))
    c.legend(title="Year", ncol=2, fontsize=8)

    # (d) ACF -> justifies lags 24 h and 168 h
    d = ax[1, 1]
    lags = np.arange(1, 24 * 14 + 1)
    acf = [gw.autocorr(int(k)) for k in lags]
    d.plot(lags, acf, lw=1.2)
    for k, col in ((24, "tab:red"), (168, "tab:green")):
        d.axvline(k, color=col, ls="--", lw=1, label=f"lag {k} h")
    d.set(title="Autocorrelation of hourly load", xlabel="Lag [hours]", ylabel="ACF")
    d.legend()

    fig.suptitle("German electricity load - seasonality", fontsize=15, y=1.0)
    fig.tight_layout()
    path = out_dir / "seasonality.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path
