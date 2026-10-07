"""End-to-end pipeline.

    python -m src.run_pipeline --source smard_api --start 2019-01-01 --end 2025-12-31
    python -m src.run_pipeline --source smard_csv --csv data/raw/smard_export.csv --start 2019-01-01 --end 2025-12-31
    python -m src.run_pipeline --source opsd --start 2015-01-01 --end 2020-09-30
    python -m src.run_pipeline --source synthetic          # offline dry run, NOT real data
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from . import config as C
from .data import clean_load, download_smard_api, read_opsd, read_smard_csv
from .eda import plot_seasonality
from .evaluate import (df_to_md, interval_by_horizon, metrics_by_group, metrics_by_horizon, overall_metrics,
                       plot_forecast_band, plot_importance, plot_metric_by_horizon)
from .features import build_supervised, feature_columns
from .model import apply_conformal, conformal_offsets, fit_model_set, predict_set

log = logging.getLogger("pipeline")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["smard_api", "smard_csv", "opsd", "synthetic"], default="smard_api")
    p.add_argument("--csv", help="path for --source smard_csv / opsd")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=(pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                   help="default: today - 10 days (the weather archive lags a few days)")
    p.add_argument("--no-weather", action="store_true", help="skip Open-Meteo / drop temperature features")
    p.add_argument("--skip-sarima", action="store_true")
    p.add_argument("--skip-prophet", action="store_true")
    p.add_argument("--folds", type=int, default=C.N_FOLDS)
    p.add_argument("--test-days", type=int, default=C.TEST_DAYS)
    p.add_argument("--n-estimators", type=int, default=None, help="cap on trees (use a small value for smoke tests)")
    p.add_argument("--skip-final-fit", action="store_true")
    return p.parse_args(argv)


# ------------------------------------------------------------------ data
def get_data(args) -> tuple[pd.Series, pd.Series | None]:
    temp = None
    if args.source == "synthetic":
        from .synthetic import make_synthetic
        d = make_synthetic()
        return d["load_mw"], (None if args.no_weather else d["temp"])
    if args.source == "smard_api":
        load = download_smard_api(args.start, args.end)
    elif args.source == "smard_csv":
        load = read_smard_csv(args.csv).loc[args.start:args.end]
    else:
        load = read_opsd(args.csv).loc[args.start:args.end]
    if not args.no_weather:
        from .weather import fetch_temperature
        temp = fetch_temperature(args.start, args.end)
    return load, temp


def make_folds(index: pd.DatetimeIndex, n_folds: int, test_days: int):
    """Expanding-window, rolling-origin folds that end at the last full day of data."""
    end = index.max().floor("D")
    folds = []
    for k in range(n_folds):
        t1 = end - pd.Timedelta(days=(n_folds - 1 - k) * test_days)
        folds.append((t1 - pd.Timedelta(days=test_days), t1))
    return folds


# ------------------------------------------------------------------ main
ALPHA = C.QUANTILES[0] + (1 - C.QUANTILES[1])   # 0.10 -> 90 % interval


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for d in (C.DATA_PROC, C.FIGS, C.MODELS, C.REPORTS):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data -----------------------------------------------------------------------
    raw_load, temp = get_data(args)
    load, report = clean_load(raw_load)
    (C.REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2))
    df = pd.DataFrame({"load_mw": load})
    if temp is not None:
        df["temp"] = temp.reindex(df.index).interpolate(limit=6, limit_area="inside")
    df.to_csv(C.DATA_PROC / "hourly_load_weather.csv")
    plot_seasonality(load)

    # 2. supervised sets ---------------------------------------------------------------
    train_all = build_supervised(df, C.TRAIN_HORIZONS, origins=df.index[:: C.TRAIN_ORIGIN_STRIDE])
    train_all = train_all.dropna(subset=["y", "ref"])
    train_all["y_res"] = train_all["y"] - train_all["ref"]      # models learn the deviation from the 4-week norm
    feats = feature_columns(train_all)
    log.info("Training pool: %d rows x %d features", len(train_all), len(feats))

    # 3. rolling-origin backtest ---------------------------------------------------------
    folds = make_folds(df.index, args.folds, args.test_days)
    if folds[0][0] - df.index.min() < pd.Timedelta(days=365):
        log.warning("First fold has < 1 year of training data - consider a longer --start/--end span.")

    all_ev, best_iters, last_models, last_offsets = [], [], None, None
    for k, (t0, t1) in enumerate(folds):
        log.info("=== Fold %d/%d  test %s -> %s ===", k + 1, len(folds), t0.date(), t1.date())
        cal_start = t0 - pd.Timedelta(days=C.CAL_DAYS)
        val_start = cal_start - pd.Timedelta(days=C.VAL_DAYS)
        # no row whose *target* lies at/after the next slice's start may be used -> no look-ahead
        tr = train_all[train_all.target_time < val_start]                                   # fit
        va = train_all[(train_all.index >= val_start) & (train_all.target_time < cal_start)]  # early stopping
        ca = train_all[(train_all.index >= cal_start) & (train_all.target_time < t0)]       # conformal calibration
        models = fit_model_set(tr, va, feats, args.n_estimators)
        best_iters.append(int(getattr(models["point"], "best_iteration_", None) or args.n_estimators or 500))

        pc = predict_set(models, ca[feats], ca["ref"].to_numpy())
        offsets = conformal_offsets(ca["y"].to_numpy(), pc["lo"], pc["hi"], ca["horizon"].to_numpy(),
                                    ALPHA, C.CONFORMAL_BINS)
        log.info("  conformal offsets [MW]: %s", {k: round(v) for k, v in offsets.items()})
        last_models, last_offsets = models, offsets

        origins = pd.date_range(t0, t1 - pd.Timedelta(days=2), freq=f"{C.ORIGIN_STEP_HOURS}h", inclusive="left")
        ev = build_supervised(df, C.EVAL_HORIZONS, origins=origins).dropna(subset=["y", "ref"])
        pred = predict_set(models, ev[feats], ev["ref"].to_numpy())
        ev["lgbm"] = pred["point"]
        ev["lgbm_lo_raw"], ev["lgbm_hi_raw"] = pred["lo"], pred["hi"]
        ev["lgbm_lo"], ev["lgbm_hi"] = apply_conformal(pred["lo"], pred["hi"], ev["horizon"].to_numpy(), offsets)
        ev["naive_day"], ev["naive_week"] = ev["lag_day"], ev["lag_week"]

        if not args.skip_sarima:
            try:
                from .baselines import sarima_forecasts
                fc = sarima_forecasts(df["load_mw"], t0, pd.DatetimeIndex(ev.index.unique()), max(C.EVAL_HORIZONS))
                ev["sarima"] = [fc[o][int(h) - 1] for o, h in zip(ev.index, ev.horizon)]
            except ImportError:
                log.warning("statsmodels not installed - SARIMA baseline skipped")
        if not args.skip_prophet:
            try:
                from .baselines import prophet_forecast
                ev["prophet"] = prophet_forecast(df, t0, pd.DatetimeIndex(ev["target_time"]))
            except ImportError:
                log.warning("prophet not installed - Prophet baseline skipped")
            except Exception as exc:  # noqa: BLE001
                log.warning("Prophet failed (%s) - skipped", exc)

        ev["fold"] = k
        all_ev.append(ev)

    bt = pd.concat(all_ev)
    bt.to_csv(C.REPORTS / "backtest_predictions.csv")

    # 4. metrics ---------------------------------------------------------------------------
    names = {"Seasonal naive (same hour, latest day)": "naive_day", "Seasonal naive (same hour last week)": "naive_week",
             "SARIMA(1,0,1)(1,1,1)24": "sarima", "Prophet": "prophet", "LightGBM": "lgbm"}
    models_cols = {n: c for n, c in names.items() if c in bt.columns and bt[c].notna().any()}
    by_h = metrics_by_horizon(bt, models_cols)
    overall = overall_metrics(bt, models_cols)
    by_h.to_csv(C.REPORTS / "metrics_by_horizon.csv", index=False)
    overall.to_csv(C.REPORTS / "metrics_overall.csv", index=False)
    interval = interval_by_horizon(bt)
    interval.to_csv(C.REPORTS / "interval_by_horizon.csv", index=False)
    plot_metric_by_horizon(by_h, "MAE")

    bt["hblock"] = pd.cut(bt["horizon"], [0, 6, 12, 24, 48], labels=["01-06h", "07-12h", "13-24h", "25-48h"])
    bt["daytype"] = np.select(
        [(bt["hol_share"] >= 0.5) | (bt["is_xmas"] == 1) | (bt["is_bridge"] == 1), bt["is_weekend"] == 1],
        ["holiday/bridge/xmas", "weekend"], "normal weekday")
    by_block = metrics_by_group(bt, models_cols, "hblock")
    by_daytype = metrics_by_group(bt, models_cols, "daytype")
    n_day = bt.groupby("daytype").size().rename("rows").reset_index()
    key_h = [1, 6, 12, 24, 36, 48]
    pivot = by_h[by_h.horizon.isin(key_h)].pivot(index="model", columns="horizon", values="MAE")
    pivot.columns = [f"h={c}" for c in pivot.columns]
    (C.REPORTS / "results.md").write_text(
        "### Overall (all folds, all horizons 1-48 h, origins every 6 h)\n\n" + df_to_md(overall, "{:,.2f}")
        + "\n\n### MAE [MW] by horizon block\n\n" + df_to_md(by_block, "{:,.0f}")
        + "\n\n### MAE [MW] by selected horizon\n\n" + df_to_md(pivot.reset_index(), "{:,.0f}")
        + "\n\n### MAE [MW] by day type\n\n" + df_to_md(by_daytype, "{:,.0f}") + "\n\n" + df_to_md(n_day)
        + "\n\n### 90 % prediction interval: raw quantile regression vs. conformalised\n\n"
        + df_to_md(interval[interval.horizon.isin(key_h)], "{:,.1f}")
        + f"\n\nFolds: {len(folds)} x {args.test_days} days, ends {folds[-1][1].date()}.\n"
    )

    # 5. key plot: h = 24 forecast on every hour of the last 14 days of the last fold -----------
    t_end = folds[-1][1]
    mask = (df.index + pd.Timedelta(hours=24) >= t_end - pd.Timedelta(days=14)) & \
           (df.index + pd.Timedelta(hours=24) < t_end)
    plot_df = build_supervised(df, [24], origins=df.index[mask]).dropna(subset=["y", "ref"])
    pp = predict_set(last_models, plot_df[feats], plot_df["ref"].to_numpy())
    plo, phi = apply_conformal(pp["lo"], pp["hi"], plot_df["horizon"].to_numpy(), last_offsets)
    frame = pd.DataFrame({"y": plot_df.y.to_numpy(), "point": pp["point"], "lo": plo, "hi": phi},
                         index=pd.DatetimeIndex(plot_df.target_time))
    plot_forecast_band(
        frame, f"Day-ahead (24 h) load forecast vs. actual, Germany - last 14 days of backtest "
               f"(LightGBM, conformal 90% band, MAE {np.mean(np.abs(frame.y - frame.point)):,.0f} MW)")
    plot_importance(last_models["point"], feats)

    # 6. final model on all data -------------------------------------------------------------
    if not args.skip_final_fit:
        n_final = int(np.mean(best_iters) * 1.1)
        log.info("Final fit on all data with %d trees", n_final)
        final = fit_model_set(train_all, None, feats, n_final)
        for name, m in final.items():
            m.booster_.save_model(str(C.MODELS / f"lgbm_{name}.txt"))
        (C.MODELS / "features.json").write_text(json.dumps(feats))
        (C.MODELS / "conformal_offsets.json").write_text(json.dumps(last_offsets))

    log.info("Done. See reports/results.md and reports/figures/")
    print((C.REPORTS / "results.md").read_text())


if __name__ == "__main__":
    main()
