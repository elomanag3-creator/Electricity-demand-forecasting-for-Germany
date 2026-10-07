# What was wrong in the first run, and how to make the forecast more accurate

## Part 1 - Diagnosis of the first full run

First-run results: LightGBM MAE 1,319 MW (2.5 % MAPE) vs 2,623 MW for "same hour last week". The model was
good overall, but three things in the results were off.

| # | Symptom | Cause | Fix (now in the code) |
|---|---|---|---|
| 1 | MAE zig-zagged across horizons (h=1: 1,679, h=12: 1,058, h=24: 1,716). Even the naive baseline zig-zagged. | One forecast origin per day (10:00 UTC), so each horizon always hit the same hour of day. h=1/24/48 = midday ramp (hard), h=12 = night (easy). The curve measured *time of day*, not forecast skill. | Origins every 6 h (`ORIGIN_STEP_HOURS`), plus MAE by horizon *block* (1-6, 7-12, 13-24, 25-48 h). |
| 2 | SARIMA beat LightGBM at h=1 (594 vs 1,679 MW) and h=6. | Trees predict a *level*; "the next hour is close to the current hour" is hard for them to express, and they cannot extrapolate. The features also had little short-term information. | Target is now the **residual** `y - same_hour_median_4w`; new features `dev_t`, `dev_mean3`, `dev_mean24`, `diff1`, `diff3` describe how far load currently deviates from its norm. |
| 3 | The "90 %" interval covered only 72-82 %. | Raw quantile boosting is typically over-confident: it is fitted on training data and sees new regimes at test time. | **Conformalised quantile regression** on a held-out calibration slice, per horizon group. |

Also worth knowing: the temperature is *observed*, not forecast, so even a perfect pipeline would be slightly
worse live; Prophet and SARIMA are weak by construction (no weekly cycle in SARIMA, no recent data in Prophet).

## Part 2 - How to make it more accurate (ordered by expected payoff / effort)

1. **Re-run and read the breakdowns first** (`reports/results.md`): by horizon block, by day type, by fold. Improve where the
   error actually is. Holidays/bridge days and Christmas usually dominate the worst errors.
2. **Use honest weather.** Replace observed temperature with archived *forecasts* (Open-Meteo historical-forecast API; check
   their docs for coverage) so training matches what you would have in production. Add solar radiation, wind speed
   and cloud cover: PV and heating/cooling change net load, and radiation matters on sunny weekends.
3. **Better holiday modelling.** Add days-to/from the nearest holiday, school-holiday share by state (the `holidays` package
   does not cover school holidays; use a public dataset), and explicit "Christmas week" / "between the years" day indices.
4. **Rooftop PV / structural change.** SMARD "Netzlast" includes the effect of embedded generation. Add a trend/year feature,
   or train on a shorter recent window, and compare. Check whether 2020-2021 (COVID) hurts: try excluding it or flagging it.
5. **Tune the model.** Try `objective="regression_l1"` (directly optimises MAE), smaller `num_leaves` (15-31), larger
   `min_child_samples`, lower learning rate. Use Optuna with the *same* rolling-origin folds - never random CV.
6. **Train separate models per horizon block** (1-6, 7-24, 25-48 h) and compare against the single model with a horizon feature.
7. **Ensemble.** Average LightGBM with a second, different model (e.g. a ridge/elastic-net on the same features, or an NN such as
   N-HiTS). Averaging models with uncorrelated errors typically gives a few percent.
8. **Calibrate intervals by hour of day** (or by day type) in addition to horizon; check conformal coverage per fold.
9. **Check for leakage and overfitting honestly.** `pytest` has a no-look-ahead test; if you add features, extend it.
   Compare against published operator forecasts (ENTSO-E day-ahead load forecast) as an external benchmark.
10. **Ablations to put in your write-up:** without temperature, without holidays, without the residual target, with a single
    vs. multiple origins per day. These show *why* the model works, which is more convincing than one accuracy number.

## Part 3 - Plots and plotly

The project uses **matplotlib only** (no plotly). All figures are PNGs in `reports/figures/`. If you want interactive charts,
`pip install plotly` and export the forecast-vs-actual frame (`reports/backtest_predictions.csv`) with `plotly.express.line`.
