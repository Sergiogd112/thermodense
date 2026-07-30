from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

CONFIDENCE = 0.95


def pearsonr_ci(
    x: np.ndarray, y: np.ndarray, confidence: float = CONFIDENCE
) -> tuple[float, float, float, int]:
    """Pearson r with Fisher z-transform confidence interval.

    Returns (r, r_lo, r_hi, n).  All NaN if insufficient finite pairs
    or zero variance.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(mask))
    if n < 3:
        return np.nan, np.nan, np.nan, n
    x0 = x[mask]
    y0 = y[mask]
    if np.std(x0) == 0 or np.std(y0) == 0:
        return np.nan, np.nan, np.nan, n
    r = float(np.corrcoef(x0, y0)[0, 1])
    if not np.isfinite(r):
        return np.nan, np.nan, np.nan, n
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = float(sp_stats.norm.ppf((1 + confidence) / 2))
    r_lo = float(np.tanh(z - z_crit * se))
    r_hi = float(np.tanh(z + z_crit * se))
    return r, r_lo, r_hi, n


def ols_slope_ci(
    x: np.ndarray, y: np.ndarray, confidence: float = CONFIDENCE
) -> tuple[float, float, float, float, float, float, float, int]:
    """OLS linear fit with slope confidence interval.

    Returns (slope, slope_lo, slope_hi, slope_se, intercept, r, rmse, n).
    All NaN if insufficient data or zero variance.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(mask))
    if n < 3:
        return (np.nan,) * 7 + (n,)
    x0 = x[mask]
    y0 = y[mask]
    if np.std(x0) == 0 or np.std(y0) == 0:
        return (np.nan,) * 7 + (n,)
    slope, intercept = np.polyfit(x0, y0, 1)
    fitted = slope * x0 + intercept
    residuals = y0 - fitted
    sse = float(np.sum(residuals**2))
    sxx = float(np.sum((x0 - np.mean(x0)) ** 2))
    if sxx == 0:
        return (np.nan,) * 7 + (n,)
    mse = sse / (n - 2)
    slope_se = float(np.sqrt(mse / sxx))
    r = float(np.corrcoef(x0, y0)[0, 1])
    rmse = float(np.sqrt(np.mean(residuals**2)))
    t_crit = float(sp_stats.t.ppf((1 + confidence) / 2, df=n - 2))
    slope_lo = float(slope - t_crit * slope_se)
    slope_hi = float(slope + t_crit * slope_se)
    return (
        float(slope),
        slope_lo,
        slope_hi,
        slope_se,
        float(intercept),
        r,
        rmse,
        n,
    )
