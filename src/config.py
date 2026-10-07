"""Central configuration. Everything tunable lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MODELS = ROOT / "models"

TZ = "Europe/Berlin"

# ---------------------------------------------------------------------------
# Forecast set-up: a day-ahead style forecast.
# Forecasts are issued every ORIGIN_STEP_HOURS (00/06/12/18 UTC) and look 48 hours ahead.
# "Horizon" = hours between the forecast origin and the hour being predicted.
# Using several origins per day matters: with a single daily origin, each horizon always maps to
# the same hour of the day, so "error vs horizon" just shows time-of-day difficulty.
# ---------------------------------------------------------------------------
ORIGIN_STEP_HOURS = 6
EVAL_HORIZONS = list(range(1, 49))
# The model is trained on a subset of horizons (horizon is an input feature,
# so it interpolates to the others) to keep the training set small.
TRAIN_HORIZONS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 42, 48]
TRAIN_ORIGIN_STRIDE = 2          # use every 2nd hourly origin for training

# ---------------------------------------------------------------------------
# Rolling-origin backtest
# ---------------------------------------------------------------------------
N_FOLDS = 4                      # expanding-window folds
TEST_DAYS = 91                   # ~ one quarter per fold -> covers 4 seasons
VAL_DAYS = 45                    # early-stopping slice (before the calibration slice)
CAL_DAYS = 45                    # most recent slice before the test period = conformal calibration set

# 90 % prediction interval
QUANTILES = (0.05, 0.95)
# horizon groups for which the conformal correction is estimated separately
CONFORMAL_BINS = [(1, 6), (7, 12), (13, 24), (25, 36), (37, 48)]

# ---------------------------------------------------------------------------
# Weather: population-weighted temperature from a few large demand centres
# (lat, lon, rough metro-region population in millions as weight).
# ---------------------------------------------------------------------------
CITIES = {
    "Berlin":     (52.52, 13.41, 4.5),
    "Hamburg":    (53.55, 9.99, 3.0),
    "Munich":     (48.14, 11.58, 2.9),
    "Rhine-Ruhr": (51.46, 7.01, 5.0),
    "Cologne":    (50.94, 6.96, 2.0),
    "Frankfurt":  (50.11, 8.68, 2.5),
    "Stuttgart":  (48.78, 9.18, 2.7),
}
