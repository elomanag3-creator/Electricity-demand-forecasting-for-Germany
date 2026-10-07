"""Metrics (MAE / RMSE / MAPE per horizon), interval diagnostics and plots."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGS, QUANTILES, TZ  # noqa: E402

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ metrics
def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mape(y, p): return float(np.mean(np.abs((y - p) / y)) * 100)


def pinball(y, p, q): 
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def metrics_by_horizon(ev: pd.DataFrame, models: dict[str, str]) -> pd.DataFrame:
    """Long table: model, horizon, MAE, RMSE, MAPE, n. Only rows where *all* models have a
    prediction are used, so the comparison is apples-to-apples."""
    ev = ev.dropna(subset=["y", *models.values()])
    rows = []
    for h, g in ev.groupby("horizon"):
        for name, col in models.items():
            rows.append({"model": name, "horizon": int(h), "MAE": mae(g.y, g[col]),
                         "RMSE": rmse(g.y, g[col]), "MAPE_%": mape(g.y, g[col]), "n": len(g)})
    return pd.DataFrame(rows)


def overall_metrics(ev: pd.DataFrame, models: dict[str, str]) -> pd.DataFrame:
    ev = ev.dropna(subset=["y", *models.values()])
    rows = [{"model": n, "MAE": mae(ev.y, ev[c]), "RMSE": rmse(ev.y, ev[c]),
             "MAPE_%": mape(ev.y, ev[c]), "n": len(ev)} for n, c in models.items()]
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def interval_by_horizon(ev: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage / width of the raw quantile interval and of the conformalised one."""
    ev = ev.dropna(subset=["y", "lgbm_lo", "lgbm_hi", "lgbm_lo_raw", "lgbm_hi_raw"])
    cov = lambda g, lo, hi: float(((g.y >= g[lo]) & (g.y <= g[hi])).mean() * 100)  # noqa: E731
    rows = []
    for h, g in ev.groupby("horizon"):
        rows.append({
            "horizon": int(h),
            "coverage_raw_%": cov(g, "lgbm_lo_raw", "lgbm_hi_raw"),
            "coverage_conformal_%": cov(g, "lgbm_lo", "lgbm_hi"),
            "width_raw_MW": float((g.lgbm_hi_raw - g.lgbm_lo_raw).mean()),
            "width_conformal_MW": float((g.lgbm_hi - g.lgbm_lo).mean()),
        })
    return pd.DataFrame(rows)


def metrics_by_group(ev: pd.DataFrame, models: dict[str, str], col: str, metric=mae) -> pd.DataFrame:
    """Pivot: rows = model, columns = groups of ``col`` (e.g. horizon block, day type)."""
    ev = ev.dropna(subset=["y", *models.values()])
    rows = []
    for g, sub in ev.groupby(col, observed=True):
        for name, c in models.items():
            rows.append({"model": name, col: str(g), "v": metric(sub.y, sub[c]), "n": len(sub)})
    d = pd.DataFrame(rows)
    return d.pivot(index="model", columns=col, values="v").reset_index()


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:,.1f}") -> str:
    """Tiny markdown-table writer (avoids the optional ``tabulate`` dependency)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):   # itertuples keeps per-column dtypes (ints stay ints)
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# -------------------------------------------------------------------- plots
def plot_forecast_band(frame: pd.DataFrame, title: str, path: Path = FIGS / "forecast_vs_actual.png") -> Path:
    """THE key plot. ``frame`` is indexed by target time (UTC) with y, point, lo, hi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = frame.copy()
    f.index = f.index.tz_convert(TZ)
    cov = ((f.y >= f.lo) & (f.y <= f.hi)).mean() * 100

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.fill_between(f.index, f.lo / 1000, f.hi / 1000, color="tab:blue", alpha=0.20,
                    label=f"{round((QUANTILES[1]-QUANTILES[0])*100)}% prediction interval "
                          f"(empirical coverage {cov:.0f}%)")
    ax.plot(f.index, f.point / 1000, color="tab:blue", lw=1.6, label="LightGBM forecast")
    ax.plot(f.index, f.y / 1000, color="black", lw=1.3, label="Actual")
    ax.set(ylabel="Grid load [GW]", title=title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)
    return path


def plot_metric_by_horizon(by_h: pd.DataFrame, metric: str = "MAE",
                           path: Path = FIGS / "mae_by_horizon.png") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, g in by_h.groupby("model"):
        lw = 3 if name == "LightGBM" else 1.6
        ax.plot(g.horizon, g[metric] / (1000 if metric != "MAPE_%" else 1), lw=lw, label=name)
    ax.set(xlabel="Forecast horizon [hours]",
           ylabel=f"{metric} [GW]" if metric != "MAPE_%" else "MAPE [%]",
           title=f"{metric} by forecast horizon (rolling-origin backtest)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_importance(model, feats: list[str], path: Path = FIGS / "feature_importance.png", top: int = 20) -> Path:
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=feats)
    imp = (imp / imp.sum()).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax, color="tab:blue")
    ax.set(title="LightGBM feature importance (share of total gain)", xlabel="gain share")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
