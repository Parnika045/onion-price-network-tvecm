"""Shared paths, constants and helper functions for the replication package.

Do Large Price Gaps Close Faster? Spatial Arbitrage, Market Networks and Weather Shocks in India’s Onion Markets.
All scripts import from here so that every constant used in the paper is set in one place.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"
OUT = ROOT / "output"
TAB = OUT / "tables"
FIG = OUT / "figures"
LOG = OUT / "logs"
for p in (DER, TAB, FIG, LOG):
    p.mkdir(parents=True, exist_ok=True)

MASTER_XLSX = RAW / "onion_prices_master_2021_2024_corrected.xlsx"
LPA_CSV = RAW / "lpa_normals_2010_2020.csv"
RESULTS_JSON = OUT / "results.json"

# ---------------------------------------------------------------- constants
SEED = 42
STABILITY_SEEDS = [42, 123, 999]
MIN_REPORT_DAYS = 200          # mandi eligibility
N_MANDIS = 240                 # top-N by cumulative arrivals among eligible
MIN_RETURNS = 60               # minimum valid returns for the correlation matrix
OUTLIER_FACTOR = 5.0           # drop modal prices > 5x or < 1/5 of the same-day national median
TRIM = 0.15                    # threshold grid trimming (each regime >= 15%)
N_GRID = 60                    # threshold grid points
B_BOOT = 999                   # bootstrap replications
VAR_MAXLAGS = 8
NW_LAGS_DAILY = 10             # Newey-West lag truncation for daily regressions
NW_LAGS_MONTHLY = 3
ROLL_WINDOW = 90               # rolling Fiedler window (trading days)
ROLL_STEP = 7
ROLL_N = 60

COMM_ORDER_NOTE = "Communities are relabelled C0..C5 by a deterministic rule (see 02_network.py)."


# ---------------------------------------------------------------- results store
def load_results() -> dict:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text())
    return {}


def save_results(section: str, payload: dict) -> None:
    """Merge one section into output/results.json (file lock on POSIX; Windows runs steps sequentially)."""
    try:
        import fcntl
    except ImportError:          # Windows
        fcntl = None
    with open(OUT / ".results.lock", "w") as lk:
        if fcntl:
            fcntl.flock(lk, fcntl.LOCK_EX)
        res = load_results()
        res[section] = _to_jsonable(payload)
        RESULTS_JSON.write_text(json.dumps(res, indent=2, sort_keys=True))
        if fcntl:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


# ---------------------------------------------------------------- econometric helpers
def half_life(theta: float) -> float:
    """Half-life (trading days) of a deviation with AR coefficient (1+theta)."""
    if not np.isfinite(theta) or theta >= 0:
        return np.inf
    base = 1.0 + theta
    if base <= 0:
        return np.nan  # oscillatory / overshooting: half-life undefined
    return float(np.log(0.5) / np.log(base))


def ols(y: np.ndarray, X: np.ndarray):
    """OLS via lstsq. Returns (beta, resid)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def nw_cov(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (Bartlett) HAC covariance for OLS coefficients. u is (T,) residuals."""
    T, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv * T / (T - k)


def month_dummies(index: pd.DatetimeIndex) -> np.ndarray:
    m = np.asarray(index.month)
    return np.column_stack([(m == k).astype(float) for k in range(2, 13)])


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def write_table(df: pd.DataFrame, name: str, note: str = "") -> None:
    df.to_csv(TAB / f"{name}.csv", index=False)
    with open(TAB / f"{name}.md", "w") as fh:
        fh.write(df.to_markdown(index=False))
        if note:
            fh.write("\n\n" + note + "\n")
