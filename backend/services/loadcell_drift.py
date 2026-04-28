"""Loadcell calibration-drift regression (Loadcell-source data).

The user records a separate `Loadcell_<...>.csv` per session in
which a known force is applied by hand to the H-Walker robot's
loadcell while the robot's own reading is logged simultaneously.
Two columns mandatory:

    applied_N        operator-applied reference force (N)
    robot_reported_N OR robot_N OR L_ActForce_N
                     robot's loadcell reading

Drift is the deviation from the identity mapping
`robot = applied`. We fit:

        robot_i = β · applied_i + α + ε_i

and report:

    slope (β)    1.0 = perfect; <1 = robot under-reads; >1 = over
    intercept (α) 0 = no offset; nonzero = bias
    R²            quality of fit
    RMSE          residual standard deviation (N)
    drift_at_50N  (β·50 + α) − 50 = predicted error at typical
                                     working load

All quantities have analytic standard errors; the t-tests against
the null (β=1, α=0) tell us whether the drift is statistically
distinguishable from no drift. Test data needs ≥4 distinct (applied,
robot) pairs (degree-of-freedom requirement for the SE estimate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# Column-name aliases the parser will accept for the two channels.
APPLIED_ALIASES = ("applied_N", "applied", "reference_N", "ref_N",
                   "hand_N", "hand_applied_N")
ROBOT_ALIASES   = ("robot_reported_N", "robot_N", "robot_force_N",
                   "L_ActForce_N", "R_ActForce_N", "loadcell_N")


def _pick(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    for a in aliases:
        if a in df.columns:
            return a
    return None


@dataclass
class DriftResult:
    n: int
    slope: float
    slope_se: float
    slope_t_vs_1: float        # t-stat for H0: β=1
    slope_p_vs_1: float
    intercept: float
    intercept_se: float
    intercept_t_vs_0: float
    intercept_p_vs_0: float
    r_squared: float
    rmse_n: float              # residual RMS, in N
    predicted_at_50N: float
    drift_at_50N_n: float      # (β·50 + α) − 50
    applied_col: str
    robot_col: str

    @property
    def slope_ok(self) -> bool:
        """Slope within 5 % of unity (engineering rule of thumb)."""
        return abs(self.slope - 1.0) < 0.05

    @property
    def intercept_ok(self) -> bool:
        return abs(self.intercept) < 1.0   # within ±1 N at zero load

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "slope": self.slope,
            "slope_se": self.slope_se,
            "slope_t_vs_1": self.slope_t_vs_1,
            "slope_p_vs_1": self.slope_p_vs_1,
            "intercept": self.intercept,
            "intercept_se": self.intercept_se,
            "intercept_t_vs_0": self.intercept_t_vs_0,
            "intercept_p_vs_0": self.intercept_p_vs_0,
            "r_squared": self.r_squared,
            "rmse_n": self.rmse_n,
            "predicted_at_50N": self.predicted_at_50N,
            "drift_at_50N_n": self.drift_at_50N_n,
            "slope_ok": self.slope_ok,
            "intercept_ok": self.intercept_ok,
            "applied_col": self.applied_col,
            "robot_col": self.robot_col,
        }


def fit_drift(df: pd.DataFrame,
              applied_col: Optional[str] = None,
              robot_col: Optional[str] = None) -> DriftResult:
    """Fit `robot = β · applied + α` via OLS and report drift stats.

    Auto-detects column names from the alias lists when not passed.
    Raises ValueError when the data is too small for inference (<4
    points) or when the applied force has no variance.
    """
    a_col = applied_col or _pick(df, APPLIED_ALIASES)
    r_col = robot_col   or _pick(df, ROBOT_ALIASES)
    if a_col is None or r_col is None:
        raise ValueError(
            f"could not locate calibration columns "
            f"(need one of {APPLIED_ALIASES} and one of {ROBOT_ALIASES})"
        )
    a = df[a_col].to_numpy(dtype=np.float64)
    r = df[r_col].to_numpy(dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(r)
    a, r = a[mask], r[mask]
    n = len(a)
    if n < 4:
        raise ValueError(f"need ≥4 calibration points (got {n})")
    if np.var(a) < 1e-12:
        raise ValueError("applied force has no variance — no slope to fit")

    # OLS y = β x + α
    a_mean = float(np.mean(a))
    r_mean = float(np.mean(r))
    sxx = float(np.sum((a - a_mean) ** 2))
    sxy = float(np.sum((a - a_mean) * (r - r_mean)))
    slope = sxy / sxx
    intercept = r_mean - slope * a_mean
    pred = slope * a + intercept
    resid = r - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((r - r_mean) ** 2))
    r_squared = float(1.0 - sse / sst) if sst > 1e-12 else 0.0
    df_resid = n - 2
    s_e2 = sse / df_resid                                        # residual variance
    rmse = float(np.sqrt(s_e2))
    slope_se = float(np.sqrt(s_e2 / sxx))
    intercept_se = float(np.sqrt(s_e2 * (1.0 / n + (a_mean ** 2) / sxx)))

    # t-tests against H0: β=1, α=0
    t_slope = (slope - 1.0) / slope_se if slope_se > 0 else float("inf")
    t_int   = (intercept - 0.0) / intercept_se if intercept_se > 0 else float("inf")

    # Two-sided p-values from t distribution
    from scipy.stats import t as t_dist
    p_slope = 2.0 * (1.0 - t_dist.cdf(abs(t_slope), df_resid))
    p_int   = 2.0 * (1.0 - t_dist.cdf(abs(t_int),   df_resid))

    pred_50 = slope * 50.0 + intercept
    drift_50 = pred_50 - 50.0

    return DriftResult(
        n=n,
        slope=slope, slope_se=slope_se,
        slope_t_vs_1=float(t_slope), slope_p_vs_1=float(p_slope),
        intercept=intercept, intercept_se=intercept_se,
        intercept_t_vs_0=float(t_int), intercept_p_vs_0=float(p_int),
        r_squared=r_squared, rmse_n=rmse,
        predicted_at_50N=float(pred_50), drift_at_50N_n=float(drift_50),
        applied_col=a_col, robot_col=r_col,
    )
