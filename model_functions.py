"""
model_functions.py
------------------
Mathematical and estimation functions for the replication of
Laurini and Hotta, "Bayesian Extensions to the Diebold-Li Term
Structure Model" on U.S. Treasury yield-curve data (2004-2006).

The functions in this module are intentionally kept short, descriptive,
and written in a Master's-student style:
    - plain functions (no classes),
    - clear docstrings for the mathematical form,
    - simple loops where the logic is clearest,
    - transparent intermediate results.

Yields are modelled on the decimal scale (e.g. 0.0438 for 4.38%).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Nelson-Siegel and Svensson yield functions
# ---------------------------------------------------------------------------

# Small number used to protect "(1 - exp(-x))/x" when x is close to zero.
_EPS = 1e-10


def _ns_slope_loading(maturity: np.ndarray, decay: float) -> np.ndarray:
    """
    Nelson-Siegel 'slope' loading L1(tau, lambda).

    L1 = (1 - exp(-lambda*tau)) / (lambda*tau)

    As lambda*tau -> 0 the loading tends to 1.  We guard the division
    with a small epsilon so the function is numerically safe.
    """
    x = decay * maturity
    x_safe = np.where(np.abs(x) < _EPS, _EPS, x)
    return (1.0 - np.exp(-x_safe)) / x_safe


def _ns_curvature_loading(maturity: np.ndarray, decay: float) -> np.ndarray:
    """
    Nelson-Siegel 'curvature' loading L2(tau, lambda).

    L2 = L1(tau, lambda) - exp(-lambda*tau)

    The curvature loading starts at 0 for very short maturities,
    peaks at an intermediate maturity, and decays to 0 for long
    maturities.
    """
    return _ns_slope_loading(maturity, decay) - np.exp(-decay * maturity)


def nelson_siegel_yield(maturity, beta_level, beta_slope,
                        beta_curvature, decay):
    """
    Evaluate the Nelson-Siegel yield function.

    y(tau) = beta_level
           + beta_slope     * L1(tau, lambda)
           + beta_curvature * L2(tau, lambda)

    Interpretation:
        beta_level     -> long-run level,
        beta_slope     -> short-end minus long-end,
        beta_curvature -> medium-term hump,
        decay          -> speed at which the slope factor decays.
    """
    maturity = np.asarray(maturity, dtype=float)
    L1 = _ns_slope_loading(maturity, decay)
    L2 = _ns_curvature_loading(maturity, decay)
    return beta_level + beta_slope * L1 + beta_curvature * L2


def svensson_yield(maturity, beta_level, beta_slope,
                   beta_curvature_1, beta_curvature_2,
                   decay_1, decay_2):
    """
    Evaluate the Svensson yield function.

    y(tau) = beta_level
           + beta_slope        * L1(tau, lambda_1)
           + beta_curvature_1  * L2(tau, lambda_1)
           + beta_curvature_2  * L2(tau, lambda_2)

    The Svensson extension adds a second curvature term with its own
    decay (lambda_2) so the curve can display two humps.
    """
    maturity = np.asarray(maturity, dtype=float)
    L1_a = _ns_slope_loading(maturity, decay_1)
    L2_a = _ns_curvature_loading(maturity, decay_1)
    L2_b = _ns_curvature_loading(maturity, decay_2)
    return (beta_level
            + beta_slope * L1_a
            + beta_curvature_1 * L2_a
            + beta_curvature_2 * L2_b)


# ---------------------------------------------------------------------------
# Nelson-Siegel estimation (classical Diebold-Li two-step procedure)
# ---------------------------------------------------------------------------

def nelson_siegel_design_matrix(maturity: np.ndarray, decay: float) -> np.ndarray:
    """
    Build the 3-column OLS design matrix [1, L1, L2] used in the
    Diebold-Li two-step procedure with a fixed decay parameter.
    """
    maturity = np.asarray(maturity, dtype=float)
    L1 = _ns_slope_loading(maturity, decay)
    L2 = _ns_curvature_loading(maturity, decay)
    return np.column_stack([np.ones_like(maturity), L1, L2])


def fit_nelson_siegel_ols_daily(long_df: pd.DataFrame, decay: float) -> pd.DataFrame:
    """
    For each date, estimate the Nelson-Siegel factors (level, slope,
    curvature) by OLS, conditional on a fixed decay parameter.

    Parameters
    ----------
    long_df : DataFrame with columns ['date', 'maturity_years', 'yield_decimal']
    decay   : fixed Nelson-Siegel decay lambda

    Returns
    -------
    DataFrame indexed by date with columns:
        beta_level, beta_slope, beta_curvature,
        n_obs, rss, rmse
    """
    factor_records = []
    for current_date, group in long_df.groupby("date", sort=True):
        maturity = group["maturity_years"].to_numpy(dtype=float)
        yields = group["yield_decimal"].to_numpy(dtype=float)
        X = nelson_siegel_design_matrix(maturity, decay)

        # Ordinary least squares via the normal equations (small system).
        coef, residuals_sum, rank, _ = np.linalg.lstsq(X, yields, rcond=None)
        fitted = X @ coef
        residuals = yields - fitted
        rss = float(np.sum(residuals ** 2))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        factor_records.append({
            "date": current_date,
            "beta_level": coef[0],
            "beta_slope": coef[1],
            "beta_curvature": coef[2],
            "n_obs": len(yields),
            "rss": rss,
            "rmse": rmse,
        })

    factors = pd.DataFrame.from_records(factor_records).set_index("date")
    return factors


def choose_ns_decay_by_grid(long_df: pd.DataFrame,
                            decay_grid: np.ndarray | None = None) -> tuple[float, pd.DataFrame]:
    """
    Pick a fixed Nelson-Siegel decay (lambda) by minimising the panel
    sum of squared residuals over a grid.  This is a transparent
    alternative to the classical Diebold-Li value of lambda = 0.0609
    (where tau is in months) and avoids silently fixing a magic number.

    Returns the selected decay and the diagnostic grid as a DataFrame.
    """
    if decay_grid is None:
        decay_grid = np.linspace(0.1, 2.0, 40)

    records = []
    for lam in decay_grid:
        factors = fit_nelson_siegel_ols_daily(long_df, decay=lam)
        rss_total = float(factors["rss"].sum())
        records.append({"decay": lam, "rss_total": rss_total})

    grid_table = pd.DataFrame.from_records(records)
    best_row = grid_table.loc[grid_table["rss_total"].idxmin()]
    return float(best_row["decay"]), grid_table


# ---------------------------------------------------------------------------
# Classical Svensson estimation (nonlinear least squares per date)
# ---------------------------------------------------------------------------

def _svensson_residuals(params, maturity, yields):
    """Residual vector used inside scipy.optimize.least_squares."""
    (beta_level, beta_slope, beta_curv_1, beta_curv_2,
     decay_1, decay_2) = params
    model_yield = svensson_yield(maturity, beta_level, beta_slope,
                                 beta_curv_1, beta_curv_2,
                                 decay_1, decay_2)
    return yields - model_yield


def _svensson_starting_values(maturity: np.ndarray, yields: np.ndarray) -> np.ndarray:
    """
    Rough but reasonable starting values based on observed curve
    characteristics (long yield, slope, 0, 0, two separated decays).
    """
    long_yield = float(np.mean(yields[maturity >= maturity.max() * 0.75]))
    short_yield = float(np.mean(yields[maturity <= max(maturity.min() * 2, 0.5)]))
    starting_level = long_yield
    starting_slope = short_yield - long_yield
    starting_curv_1 = 0.0
    starting_curv_2 = 0.0
    starting_decay_1 = 0.7
    starting_decay_2 = 0.15
    return np.array([starting_level, starting_slope,
                     starting_curv_1, starting_curv_2,
                     starting_decay_1, starting_decay_2])


def fit_svensson_nls_daily(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate the Svensson curve for each date by nonlinear least squares
    with simple bounds and a separation guard between decay_1 and decay_2.

    Returns
    -------
    parameters : DataFrame of estimated parameters per date.
    convergence : DataFrame with status and messages per date.
    """
    parameter_records = []
    convergence_records = []

    # Lower / upper bounds for the six parameters.
    lower_bounds = np.array([-1.0, -1.0, -1.0, -1.0, 1e-3, 1e-3])
    upper_bounds = np.array([ 1.0,  1.0,  1.0,  1.0, 10.0, 10.0])

    for current_date, group in long_df.groupby("date", sort=True):
        maturity = group["maturity_years"].to_numpy(dtype=float)
        yields = group["yield_decimal"].to_numpy(dtype=float)

        p0 = _svensson_starting_values(maturity, yields)

        # The two decays can collapse to the same value.  We fit, and if
        # they are numerically close, we refit from an alternative
        # starting point and keep whichever has the smaller residual sum.
        candidates = []
        for p_start in [p0, p0 * np.array([1, 1, 1, 1, 2.0, 0.5])]:
            p_start = np.clip(p_start, lower_bounds + 1e-6, upper_bounds - 1e-6)
            try:
                result = least_squares(
                    _svensson_residuals, p_start,
                    args=(maturity, yields),
                    bounds=(lower_bounds, upper_bounds),
                    method="trf",
                    max_nfev=5000,
                )
                candidates.append(result)
            except Exception as err:  # keep going even if one start fails
                candidates.append(None)
                convergence_records.append({
                    "date": current_date,
                    "status": "exception",
                    "message": str(err),
                    "cost": np.nan,
                })

        valid = [r for r in candidates if r is not None]
        if not valid:
            parameter_records.append({
                "date": current_date,
                "beta_level": np.nan,
                "beta_slope": np.nan,
                "beta_curvature_1": np.nan,
                "beta_curvature_2": np.nan,
                "decay_1": np.nan,
                "decay_2": np.nan,
                "rss": np.nan,
                "rmse": np.nan,
                "n_obs": len(yields),
                "converged": False,
            })
            continue

        best = min(valid, key=lambda r: r.cost)
        residuals = best.fun
        rss = float(np.sum(residuals ** 2))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # Flag numerically identical decays (ill-identified case).
        decay_1_hat, decay_2_hat = best.x[4], best.x[5]
        decays_too_close = abs(decay_1_hat - decay_2_hat) < 1e-3

        parameter_records.append({
            "date": current_date,
            "beta_level": best.x[0],
            "beta_slope": best.x[1],
            "beta_curvature_1": best.x[2],
            "beta_curvature_2": best.x[3],
            "decay_1": decay_1_hat,
            "decay_2": decay_2_hat,
            "rss": rss,
            "rmse": rmse,
            "n_obs": len(yields),
            "converged": bool(best.success),
            "decays_identical": decays_too_close,
        })
        convergence_records.append({
            "date": current_date,
            "status": "success" if best.success else "did_not_converge",
            "message": best.message,
            "cost": float(best.cost),
            "nfev": int(best.nfev),
        })

    parameters = pd.DataFrame.from_records(parameter_records).set_index("date")
    convergence = pd.DataFrame.from_records(convergence_records).set_index("date")
    return parameters, convergence


# ---------------------------------------------------------------------------
# Factor dynamics: AR(1) per factor
# ---------------------------------------------------------------------------

def fit_ar1_per_factor(factor_df: pd.DataFrame) -> dict:
    """
    Fit an AR(1) model to each column of factor_df using statsmodels.
    AR(1) is used instead of a full VAR(1) because it is simpler, keeps
    the number of parameters small, and is a common choice in the
    Diebold-Li literature when the focus is on one-step forecasts.

    Returns a dictionary mapping column name to the fitted model.
    """
    fitted = {}
    for column_name in factor_df.columns:
        series = factor_df[column_name].astype(float).dropna()
        # statsmodels AR(1) via ARIMA(1,0,0) with a constant.
        model = sm.tsa.ARIMA(series, order=(1, 0, 0),
                             trend="c",
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted[column_name] = model.fit()
    return fitted


def ar1_one_step_forecast(fitted_models: dict,
                          current_values: dict) -> dict:
    """
    Compute a one-step-ahead forecast for each AR(1) model.
    For an AR(1) with mean mu, intercept c and AR coefficient phi,
    the forecast is: E[y_{t+1} | y_t] = c + phi * y_t.
    """
    forecasts = {}
    for name, model in fitted_models.items():
        # statsmodels stores: const (intercept-like) and ar.L1
        params = model.params
        intercept = float(params.get("const", 0.0))
        phi = float(params.get("ar.L1", 0.0))
        y_t = float(current_values[name])
        # ARIMA(1,0,0) with trend "c" parameterises as y_t = const + phi*y_{t-1} + eps
        # The 'const' from statsmodels is actually the process mean mu, so:
        #     y_{t+1} = mu + phi * (y_t - mu)
        mu = intercept
        forecasts[name] = mu + phi * (y_t - mu)
    return forecasts


# ---------------------------------------------------------------------------
# Fit summaries
# ---------------------------------------------------------------------------

def panel_fit_statistics(long_df: pd.DataFrame,
                         fitted_wide: pd.DataFrame) -> dict:
    """
    Compute pooled fit statistics (RMSE, MAE, mean residual, std of
    residuals) across all date-maturity cells where both observed and
    fitted yields are present.

    long_df     : long dataframe with 'date', 'maturity_years', 'yield_decimal'.
    fitted_wide : DataFrame indexed by date, columns are maturity_years values,
                  with fitted yields on the decimal scale.
    """
    merged = []
    for current_date, group in long_df.groupby("date", sort=True):
        if current_date not in fitted_wide.index:
            continue
        row = fitted_wide.loc[current_date]
        for _, obs in group.iterrows():
            maturity = obs["maturity_years"]
            if maturity in row.index and np.isfinite(row[maturity]):
                merged.append({
                    "date": current_date,
                    "maturity_years": maturity,
                    "observed": float(obs["yield_decimal"]),
                    "fitted": float(row[maturity]),
                })
    merged_df = pd.DataFrame.from_records(merged)
    residuals = merged_df["observed"] - merged_df["fitted"]
    return {
        "n_obs": int(len(residuals)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan,
        "merged_long": merged_df,
    }


def forecast_error_metrics(forecast_df: pd.DataFrame) -> dict:
    """
    Compute standard forecast error metrics from a tidy DataFrame with
    columns ['observed', 'forecast'] (both on the decimal yield scale).
    """
    errors = forecast_df["observed"] - forecast_df["forecast"]
    return {
        "n_obs": int(len(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "mean_error": float(np.mean(errors)),
    }
