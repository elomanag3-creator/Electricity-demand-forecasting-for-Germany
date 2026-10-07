"""Loading + cleaning of German hourly load data.

Three sources are supported (all free, no token):

* ``smard_api``  - the JSON endpoints behind the SMARD.de charts (filter 410 =
                   "Stromverbrauch: Gesamt (Netzlast)"). Timestamps are UTC epoch
                   milliseconds, so there is no DST problem.
* ``smard_csv``  - a CSV exported by hand from https://www.smard.de/en/downloadcenter.
                   Timestamps are *local wall-clock time* -> DST handling is needed.
* ``opsd``       - Open Power System Data, ``time_series_60min_singleindex.csv``
                   (UTC timestamps; coverage ends in 2020).

Everything is converted to a **UTC, strictly hourly** ``pd.Series`` named ``load_mw``.
Local time is only reconstructed later (``tz_convert("Europe/Berlin")``) to build calendar
features. This is the clean way to avoid the daylight-saving trap.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DATA_RAW, TZ

log = logging.getLogger(__name__)

SMARD_CHART = "https://www.smard.de/app/chart_data"
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/latest/"
    "time_series_60min_singleindex.csv"
)
OPSD_COL = "DE_load_actual_entsoe_transparency"


# ---------------------------------------------------------------------------
# SMARD JSON API
# ---------------------------------------------------------------------------
def download_smard_api(
    start: str, end: str, filter_id: int = 410, region: str = "DE", resolution: str = "hour"
) -> pd.Series:
    """Download hourly grid load (MW) from the SMARD chart-data endpoints.

    The API serves one JSON file per week-long block; ``index_<resolution>.json`` lists the
    block start timestamps. Old blocks are cached on disk, recent ones are re-downloaded.
    """
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    to_ms = lambda t: int(t.timestamp() * 1000)  # noqa: E731
    cache_dir = DATA_RAW / "smard_api" / f"{filter_id}_{region}_{resolution}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(f"{SMARD_CHART}/{filter_id}/{region}/index_{resolution}.json", timeout=30)
    r.raise_for_status()
    stamps = sorted(r.json()["timestamps"])
    lo, hi = to_ms(start_ts - pd.Timedelta(days=8)), to_ms(end_ts)
    wanted = [t for t in stamps if lo <= t <= hi]
    log.info("SMARD: fetching %d weekly blocks", len(wanted))

    now = pd.Timestamp.now(tz="UTC")
    rows: list = []
    for t in wanted:
        f = cache_dir / f"{t}.json"
        if f.exists():
            payload = json.loads(f.read_text())
        else:
            url = f"{SMARD_CHART}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{t}.json"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            if pd.Timestamp(t, unit="ms", tz="UTC") < now - pd.Timedelta(days=14):
                f.write_text(json.dumps(payload))  # only cache blocks that are final
        rows.extend(payload["series"])

    df = pd.DataFrame(rows, columns=["ts", "load_mw"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    s = df.drop_duplicates("ts").set_index("ts")["load_mw"].sort_index()
    return s.loc[start_ts:end_ts]


# ---------------------------------------------------------------------------
# SMARD manual CSV export (local time!)
# ---------------------------------------------------------------------------
def dst_audit(naive_local: pd.DatetimeIndex) -> dict:
    """Count the two classic DST artefacts in a *naive local-time* index.

    * autumn: 02:00-03:00 occurs twice  -> duplicated timestamps
    * spring: 02:00-03:00 never occurs  -> missing timestamps
    """
    step = pd.Series(naive_local).diff().mode().iloc[0]
    full = pd.date_range(naive_local.min(), naive_local.max(), freq=step)
    return {
        "duplicated_local_stamps": int(naive_local.duplicated().sum()),
        "missing_local_stamps": int(len(full.difference(naive_local.unique()))),
    }


def read_smard_csv(
    path: str | Path,
    sep: str = ";",
    decimal: str = ",",
    thousands: str = ".",
    fmt: str = "%d.%m.%Y %H:%M",
) -> pd.Series:
    """Read a CSV from the SMARD download centre and return hourly UTC load in MW.

    Handles: German number format ("1.234,56"), "-" placeholders, local-time stamps incl. the
    duplicated autumn hour / missing spring hour, and 15-minute resolution (summed to hourly).
    For English-language exports use ``decimal=".", thousands=","``.
    """
    df = pd.read_csv(path, sep=sep, decimal=decimal, thousands=thousands, na_values=["-", "n/a"],
                     dtype={0: str, 1: str})
    cols = list(df.columns)
    cand = [c for c in cols[2:] if re.search(r"netzlast|grid load", c, re.I)
            and not re.search(r"residual", c, re.I)]
    load_col = cand[0] if cand else cols[2]
    log.info("SMARD CSV: using column %r", load_col)

    naive = pd.DatetimeIndex(pd.to_datetime(df[cols[0]], format=fmt))
    audit = dst_audit(naive)
    log.info("DST audit of local timestamps: %s", audit)

    # First occurrence of a duplicated wall-clock time = summer time (CEST, DST=True),
    # second occurrence = winter time (CET, DST=False). Unambiguous rows ignore the flag.
    is_dst = ~naive.duplicated(keep="first")
    local = naive.tz_localize(TZ, ambiguous=np.asarray(is_dst), nonexistent="shift_forward")

    s = pd.Series(df[load_col].to_numpy(dtype=float), index=local.tz_convert("UTC"), name="load_mw")
    s = s[~s.index.duplicated(keep="first")].sort_index()

    step = s.index.to_series().diff().median()
    if step < pd.Timedelta("1h"):
        n = int(round(pd.Timedelta("1h") / step))
        log.info("Resampling %s data to hourly (MWh per slot are summed -> MWh/h = MW)", step)
        s = s.resample("1h").sum(min_count=n)
    return s


# ---------------------------------------------------------------------------
# Open Power System Data
# ---------------------------------------------------------------------------
def download_opsd(dest: Path | None = None) -> Path:
    dest = dest or (DATA_RAW / "opsd_time_series_60min.csv")
    if dest.exists():
        return dest
    log.info("Downloading OPSD time series (~100+ MB) ...")
    with requests.get(OPSD_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def read_opsd(path: str | Path | None = None) -> pd.Series:
    path = Path(path) if path else download_opsd()
    df = pd.read_csv(path, usecols=["utc_timestamp", OPSD_COL], parse_dates=["utc_timestamp"],
                     index_col="utc_timestamp")
    idx = df.index
    df.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    return df[OPSD_COL].rename("load_mw").dropna().sort_index()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_load(
    s: pd.Series, max_interp_gap: int = 3, min_plausible: float = 10_000.0
) -> tuple[pd.Series, dict]:
    """Make the series strictly hourly, remove impossible values, fill *short* gaps only.

    Gaps longer than ``max_interp_gap`` hours stay NaN (rows with missing targets are simply
    dropped at training time) - inventing days of data would flatter the model.
    """
    s = s.sort_index()
    n_dup = int(s.index.duplicated().sum())
    s = s.groupby(level=0).mean()

    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full)
    s.index.name = "timestamp"

    implausible = s < min_plausible          # zeros / negative / absurdly low values
    n_implausible = int(implausible.sum())
    s = s.mask(implausible)

    is_na = s.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")        # length of the NaN-run each row is in
    fillable = is_na & (run_len <= max_interp_gap)
    interpolated = s.interpolate(limit_area="inside")
    out = s.where(~fillable, interpolated)

    report = {
        "start": str(s.index.min()),
        "end": str(s.index.max()),
        "n_hours": int(len(s)),
        "duplicate_timestamps_merged": n_dup,
        "implausible_values_removed": n_implausible,
        "missing_hours_raw": int(is_na.sum()),
        "longest_gap_hours": int(run_len[is_na].max()) if is_na.any() else 0,
        "hours_interpolated": int((fillable & out.notna()).sum()),
        "missing_hours_after_cleaning": int(out.isna().sum()),
    }
    log.info("Data quality: %s", report)
    return out.rename("load_mw"), report
