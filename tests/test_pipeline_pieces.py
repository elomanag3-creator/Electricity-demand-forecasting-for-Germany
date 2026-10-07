"""Run with:  pytest -q"""
import numpy as np
import pandas as pd

from src.data import clean_load, read_smard_csv
from src.features import build_supervised
from src.synthetic import make_synthetic, write_smard_style_csv


def _series_around_dst():
    # spans the 2022 spring change (27 Mar) and the autumn change (30 Oct)
    idx = pd.date_range("2022-03-20", "2022-11-10", freq="h", tz="UTC")
    return pd.Series(np.arange(len(idx), dtype=float) + 30000, index=idx, name="load_mw")


def test_smard_csv_roundtrip_hourly(tmp_path):
    """Local-time CSV with duplicated autumn hour + missing spring hour must map back to the
    exact original UTC series."""
    s = _series_around_dst()
    p = tmp_path / "smard.csv"
    write_smard_style_csv(s, p)
    back = read_smard_csv(p)
    pd.testing.assert_series_equal(back, s, check_freq=False, check_names=False)


def test_smard_csv_roundtrip_quarter_hour(tmp_path):
    """15-minute exports are summed to hourly MWh (= MW)."""
    s = _series_around_dst()
    p = tmp_path / "smard15.csv"
    write_smard_style_csv(s, p, quarter_hour=True)
    back = read_smard_csv(p)
    pd.testing.assert_series_equal(back, s, check_freq=False, check_names=False, rtol=1e-6)


def test_clean_load_fills_only_short_gaps():
    idx = pd.date_range("2022-01-01", periods=200, freq="h", tz="UTC")
    s = pd.Series(40000.0 + np.arange(200), index=idx)
    s.iloc[10:12] = np.nan      # 2 h  -> interpolated
    s.iloc[50:60] = np.nan      # 10 h -> must stay NaN
    s.iloc[100] = 0.0           # implausible -> removed, then interpolated (1 h gap)
    out, rep = clean_load(s)
    assert out.iloc[10:12].notna().all()
    assert out.iloc[50:60].isna().all()
    assert out.iloc[100] > 0
    assert rep["longest_gap_hours"] == 10


def test_no_look_ahead_in_load_features():
    """Changing load values *after* the origin must not change any feature at that origin."""
    d = make_synthetic("2022-01-01", "2022-04-30")
    df = d.rename(columns={"load_mw": "load_mw"}).copy()
    origin = df.index[24 * 60 + 10]
    base = build_supervised(df, [1, 12, 24, 48], origins=pd.DatetimeIndex([origin]))
    df2 = df.copy()
    df2.loc[df2.index > origin, "load_mw"] += 12345.0
    pert = build_supervised(df2, [1, 12, 24, 48], origins=pd.DatetimeIndex([origin]))
    cols = [c for c in base.columns if c not in ("y",)]
    pd.testing.assert_frame_equal(base[cols], pert[cols])
    assert not np.allclose(base["y"], pert["y"])      # sanity: the target *did* change
