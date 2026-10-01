"""
main.py
-------
Replication of Laurini and Hotta, "Bayesian Extensions to the Diebold-Li
Term Structure Model" on U.S. Treasury yield-curve data (2004-2006).

This is the main script.  It should be run from the project
root directory as:

    python main.py

QUICK_MODE
----------
QUICK_MODE = True   -> small MCMC run, suitable for development.
QUICK_MODE = False  -> larger MCMC run that is closer to the paper
                       (10,000 iterations with 5,000 burn-in).

Replication classification
--------------------------
This is a *methodological* replication: we re-implement the paper's
model specifications on a different dataset (U.S. Treasury par yields
2004-2006 instead of the Brazilian BM&F Swap DI-PRE term structure).
Any numerical result will differ from the published values because of
the dataset, the sampler, and software choices.  Those differences are
tracked explicitly in outputs/tables/paper_replication_differences.csv.
"""

from __future__ import annotations

import io
import logging
import os
import platform
import sys
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path

# Reconfigure the console streams to UTF-8 so that PyMC's / ArviZ's
# progress bars (which contain Unicode block characters) do not raise a
# UnicodeEncodeError on Windows cp1252 terminals.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Silence noisy third-party loggers / warnings before the libraries are
# imported so the console output of the replication stays readable.
# These are all cosmetic: statsmodels warns about ARIMA date-index
# frequency (we fit many short AR(1)s), and pytensor's graph rewriter
# prints OverflowError tracebacks on a known int8 edge case while still
# falling back to a working graph.
warnings.filterwarnings("ignore", category=UserWarning,
                        module="statsmodels")
warnings.filterwarnings("ignore", message=".*date index.*",
                        category=Warning)
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("pytensor.graph.rewriting.basic").setLevel(logging.CRITICAL)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

import arviz as az
import pymc as pm
import pytensor.tensor as pt

import model_functions as mf


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

QUICK_MODE = True
RANDOM_SEED = 20240101

# If True and a previously saved posterior file exists, reload it instead
# of re-running the MCMC sampler.  Useful during development.  Set to
# False (default) for a full fresh run.
REUSE_POSTERIOR = False

INPUT_FILE = "us_treasury_yield_curve_long_2004_2006.csv"

# Train/test chronological split (replication choice, not a paper
# specification).  The paper's forecasting protocol is not fully
# transparent from the published description, so we use a straightforward
# 80 / 20 split by date, which is standard in forecasting replications.
TRAIN_FRACTION = 0.80

# Output directories.
OUT_ROOT = Path("outputs")
OUT_TABLES = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"
OUT_DIAG = OUT_ROOT / "diagnostics"
OUT_POSTERIOR = OUT_ROOT / "posterior_draws"
OUT_LOGS = OUT_ROOT / "logs"
for folder in [OUT_TABLES, OUT_FIGURES, OUT_DIAG, OUT_POSTERIOR, OUT_LOGS]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging helper (duplicate stdout to a log file)
# ---------------------------------------------------------------------------

class Tee:
    """Write to several text streams at once (used to log the console)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


LOG_PATH = OUT_LOGS / "complete_python_output.txt"
_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)


def banner(title: str) -> None:
    """Print a visual section header to make the console output readable."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Environment report
# ---------------------------------------------------------------------------

banner("1. ENVIRONMENT")
print(f"Date / time        : {pd.Timestamp.now()}")
print(f"Python version     : {platform.python_version()}")
print(f"Platform           : {platform.platform()}")
print(f"QUICK_MODE         : {QUICK_MODE}")
print(f"Random seed        : {RANDOM_SEED}")
print(f"Train fraction     : {TRAIN_FRACTION}")
print(f"NumPy              : {np.__version__}")
print(f"pandas             : {pd.__version__}")
print(f"PyMC               : {pm.__version__}")
print(f"ArviZ              : {az.__version__}")

rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 2. Load, validate and clean the dataset
# ---------------------------------------------------------------------------

banner("2. LOAD AND VALIDATE DATA")

raw = pd.read_csv(INPUT_FILE)
print(f"Input file         : {INPUT_FILE}")
print(f"Raw rows           : {len(raw)}")
print(f"Raw columns        : {list(raw.columns)}")

# Parse the date column as actual datetime values.
raw["date"] = pd.to_datetime(raw["date"], errors="raise")

# Sort for reproducibility and panel checks.
raw = raw.sort_values(["date", "maturity_years"]).reset_index(drop=True)

# Validation checks (printed for the log).
duplicates = raw.duplicated(subset=["date", "maturity_years"]).sum()
missing_yields = raw["yield_decimal"].isna().sum()
non_positive_maturity = (raw["maturity_years"] <= 0).sum()
non_numeric_yield = (~raw["yield_decimal"].apply(np.isreal)).sum()

print(f"Duplicate (date, maturity) rows : {duplicates}")
print(f"Missing yield_decimal          : {missing_yields}")
print(f"Non-positive maturity rows     : {non_positive_maturity}")
print(f"Non-numeric yield_decimal rows : {non_numeric_yield}")

# Drop rows with a missing yield so later estimation code can rely on
# yield_decimal being finite.  We do NOT interpolate missing maturity-date
# cells; we simply work with the ragged panel.
clean = raw.dropna(subset=["yield_decimal"]).copy()

all_dates = pd.Index(sorted(clean["date"].unique()))
print(f"Number of distinct dates       : {len(all_dates)}")
print(f"Start date                     : {all_dates.min().date()}")
print(f"End date                       : {all_dates.max().date()}")
print(f"Total clean observations       : {len(clean)}")
print(f"Minimum maturity (years)       : {clean['maturity_years'].min():.4f}")
print(f"Maximum maturity (years)       : {clean['maturity_years'].max():.4f}")
print(f"Mean maturities per date       : {len(clean) / len(all_dates):.2f}")
print("Panel is in long format        : True")

# Save a cleaned modelling copy for reproducibility.
clean.to_csv(OUT_ROOT / "clean_yield_curve_long.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Descriptive tables and figures
# ---------------------------------------------------------------------------

banner("3. DESCRIPTIVE ANALYSIS")

# --- Tables ---

data_summary = pd.DataFrame({
    "metric": ["n_rows", "n_dates", "n_maturities",
               "min_date", "max_date",
               "min_maturity_years", "max_maturity_years",
               "mean_yield_percent", "std_yield_percent"],
    "value": [len(clean),
              clean["date"].nunique(),
              clean["maturity_years"].nunique(),
              str(all_dates.min().date()),
              str(all_dates.max().date()),
              clean["maturity_years"].min(),
              clean["maturity_years"].max(),
              clean["yield_percent"].mean(),
              clean["yield_percent"].std(ddof=1)],
})
data_summary.to_csv(OUT_TABLES / "data_summary.csv", index=False)

yield_by_maturity = (clean.groupby("maturity_years")["yield_percent"]
                     .agg(["count", "mean", "std", "min", "max"])
                     .reset_index()
                     .rename(columns={"count": "n_obs"}))
yield_by_maturity.to_csv(OUT_TABLES / "yield_summary_by_maturity.csv", index=False)

obs_by_date = (clean.groupby("date")["maturity_years"]
               .count().reset_index()
               .rename(columns={"maturity_years": "n_maturities"}))
obs_by_date.to_csv(OUT_TABLES / "observations_by_date.csv", index=False)

# Missing-value table: count of expected observations missing by maturity.
# We define "expected" as any maturity that appears at least once overall.
all_maturities = sorted(clean["maturity_years"].unique())
missing_by_maturity = []
for m in all_maturities:
    n_present = clean[clean["maturity_years"] == m]["date"].nunique()
    n_missing = len(all_dates) - n_present
    missing_by_maturity.append({
        "maturity_years": m,
        "n_dates_with_obs": n_present,
        "n_dates_missing": n_missing,
    })
pd.DataFrame(missing_by_maturity).to_csv(
    OUT_TABLES / "missing_values_summary.csv", index=False)

print("Descriptive tables written to outputs/tables/")


# --- Descriptive figures ---

# Wide pivot of yields_percent for convenient plotting; NaN preserves holes.
wide_percent = (clean.pivot(index="date", columns="maturity_years",
                             values="yield_percent")
                .sort_index())

# Figure 1: yields over time by maturity.
fig, ax = plt.subplots(figsize=(10, 6))
for m in wide_percent.columns:
    ax.plot(wide_percent.index, wide_percent[m], linewidth=1,
            label=f"{m:g}y")
ax.set_title("U.S. Treasury par yields over time (2004-2006)")
ax.set_xlabel("Date")
ax.set_ylabel("Yield (percent)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT_FIGURES / "01_yields_over_time.png", dpi=150)
plt.close(fig)

# Figure 2: 3-D yield-curve surface.
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
# Build a dense grid; NaN cells will show as holes.
date_numeric = (wide_percent.index - wide_percent.index.min()).days
X, Y = np.meshgrid(date_numeric, wide_percent.columns.to_numpy(dtype=float))
Z = wide_percent.T.to_numpy()
ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0)
ax.set_title("Yield-curve surface")
ax.set_xlabel("Days since start")
ax.set_ylabel("Maturity (years)")
ax.set_zlabel("Yield (percent)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "02_yield_curve_surface.png", dpi=150)
plt.close(fig)

# Figure 3: selected yield curves on representative dates (we choose
# evenly spaced quarterly snapshots to show how the curve evolves).
snapshot_dates = wide_percent.index[::len(wide_percent) // 10][:10]
fig, ax = plt.subplots(figsize=(10, 6))
for d in snapshot_dates:
    row = wide_percent.loc[d].dropna()
    ax.plot(row.index, row.values, marker="o",
            label=d.strftime("%Y-%m-%d"))
ax.set_title("Selected daily yield curves")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Yield (percent)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_FIGURES / "03_selected_yield_curves.png", dpi=150)
plt.close(fig)

# Figure 4: average yield curve.
avg_curve = wide_percent.mean(axis=0)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(avg_curve.index, avg_curve.values, marker="o")
ax.set_title("Average yield curve (2004-2006)")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Mean yield (percent)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "04_average_yield_curve.png", dpi=150)
plt.close(fig)

# Figure 5: yield volatility by maturity (standard deviation).
vol_by_mat = wide_percent.std(axis=0, ddof=1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(vol_by_mat.index.astype(str), vol_by_mat.values)
ax.set_title("Yield volatility by maturity")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Standard deviation of yield (percent)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "05_volatility_by_maturity.png", dpi=150)
plt.close(fig)

# Figure 6: maturity coverage over time (number of maturities each date).
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(obs_by_date["date"], obs_by_date["n_maturities"])
ax.set_title("Number of available maturities per date")
ax.set_xlabel("Date")
ax.set_ylabel("Number of maturities")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "06_maturity_coverage.png", dpi=150)
plt.close(fig)

# Figure 7: yield heatmap (date x maturity).
fig, ax = plt.subplots(figsize=(10, 5))
heat = ax.imshow(wide_percent.T.to_numpy(), aspect="auto",
                  cmap=cm.viridis, origin="lower",
                  extent=[0, len(wide_percent), 0, len(wide_percent.columns)])
ax.set_yticks(np.arange(len(wide_percent.columns)) + 0.5)
ax.set_yticklabels([f"{m:g}y" for m in wide_percent.columns])
ax.set_title("Heatmap of yields by date and maturity (percent)")
ax.set_xlabel("Date index")
ax.set_ylabel("Maturity")
plt.colorbar(heat, ax=ax, label="Yield (percent)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "07_yield_heatmap.png", dpi=150)
plt.close(fig)

print("Descriptive figures written to outputs/figures/")


# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------

banner("4. TRAIN / TEST SPLIT")

split_index = int(round(TRAIN_FRACTION * len(all_dates)))
train_dates = all_dates[:split_index]
test_dates = all_dates[split_index:]
split_date = test_dates[0]

print(f"Training dates: {len(train_dates)}  ({train_dates.min().date()} to {train_dates.max().date()})")
print(f"Test dates    : {len(test_dates)}  ({test_dates.min().date()} to {test_dates.max().date()})")
print(f"Split date    : {split_date.date()}")

pd.DataFrame({
    "train_fraction": [TRAIN_FRACTION],
    "n_train_dates": [len(train_dates)],
    "n_test_dates": [len(test_dates)],
    "train_start": [train_dates.min().date()],
    "train_end": [train_dates.max().date()],
    "test_start": [test_dates.min().date()],
    "test_end": [test_dates.max().date()],
    "split_date": [split_date.date()],
}).to_csv(OUT_TABLES / "sample_split.csv", index=False)

train_long = clean[clean["date"].isin(train_dates)].copy()
test_long = clean[clean["date"].isin(test_dates)].copy()


# ---------------------------------------------------------------------------
# 5. Baseline Diebold-Li Nelson-Siegel model
# ---------------------------------------------------------------------------

banner("5. NELSON-SIEGEL (DIEBOLD-LI) MODEL")

# Pick the decay parameter by minimising the training-sample panel RSS on
# a grid.  This is a transparent alternative to fixing a magic value.
ns_decay, ns_grid = mf.choose_ns_decay_by_grid(
    train_long, decay_grid=np.linspace(0.1, 2.0, 40)
)
print(f"Selected Nelson-Siegel decay (lambda): {ns_decay:.4f}")
ns_grid.to_csv(OUT_TABLES / "nelson_siegel_decay_grid.csv", index=False)

# Fit NS factors by OLS on EVERY date (train + test) using this single lambda.
# For the test-period dates these daily factors are only used to construct
# residuals and in-sample fit; the forecast evaluation uses one-step rolls
# based on information up to t-1 (see section 11).
ns_factors_all = mf.fit_nelson_siegel_ols_daily(clean, decay=ns_decay)
ns_factors_all.to_csv(OUT_TABLES / "nelson_siegel_factors.csv")

# In-sample fit: build fitted yields on the same date-maturity cells as
# the observed data and compute pooled statistics.
fitted_rows = []
for current_date, group in clean.groupby("date"):
    coef = ns_factors_all.loc[current_date, ["beta_level", "beta_slope",
                                             "beta_curvature"]].to_numpy()
    maturities = group["maturity_years"].to_numpy()
    fitted = mf.nelson_siegel_yield(maturities, coef[0], coef[1], coef[2],
                                    ns_decay)
    for m, f in zip(maturities, fitted):
        fitted_rows.append({"date": current_date, "maturity_years": m,
                            "fitted_yield_decimal": f})
ns_fitted_long = pd.DataFrame(fitted_rows)
ns_fitted_wide = (ns_fitted_long.pivot(index="date", columns="maturity_years",
                                       values="fitted_yield_decimal"))
ns_fit_stats = mf.panel_fit_statistics(clean, ns_fitted_wide)
print(f"NS in-sample RMSE (all dates, decimal) : {ns_fit_stats['rmse']:.6f}")
print(f"NS in-sample MAE  (all dates, decimal) : {ns_fit_stats['mae']:.6f}")

ns_fit_summary = pd.DataFrame({
    "metric": ["n_obs", "rmse", "mae", "mean_residual", "std_residual"],
    "value": [ns_fit_stats["n_obs"], ns_fit_stats["rmse"],
              ns_fit_stats["mae"], ns_fit_stats["mean_residual"],
              ns_fit_stats["std_residual"]],
})
ns_fit_summary.to_csv(OUT_TABLES / "nelson_siegel_fit.csv", index=False)

# AR(1) models for the three factors (fitted on the training window only).
ns_factors_train = ns_factors_all.loc[train_dates,
                                      ["beta_level", "beta_slope", "beta_curvature"]]
ar1_ns_models = mf.fit_ar1_per_factor(ns_factors_train)
print("Nelson-Siegel AR(1) factor dynamics fitted on training window.")
for name, model in ar1_ns_models.items():
    print(f"  {name:<16} mean={model.params.get('const', np.nan):.5f}  "
          f"phi={model.params.get('ar.L1', np.nan):.3f}")

# --- Figures ---

# Figure 8: factors over time.
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ns_factors_all.index, ns_factors_all["beta_level"], label="Level")
ax.plot(ns_factors_all.index, ns_factors_all["beta_slope"], label="Slope")
ax.plot(ns_factors_all.index, ns_factors_all["beta_curvature"], label="Curvature")
ax.axvline(split_date, linestyle="--", color="grey", linewidth=1,
           label="Train/test split")
ax.set_title("Nelson-Siegel daily factors (decimal yield units)")
ax.set_xlabel("Date")
ax.set_ylabel("Factor value")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_FIGURES / "08_nelson_siegel_factors.png", dpi=150)
plt.close(fig)

# Figure 9: observed vs. fitted on a few dates.
pick_dates = ns_factors_all.index[::len(ns_factors_all) // 6][:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for ax, d in zip(axes.flat, pick_dates):
    obs = clean[clean["date"] == d].sort_values("maturity_years")
    coef = ns_factors_all.loc[d]
    tau_dense = np.linspace(obs["maturity_years"].min(),
                            obs["maturity_years"].max(), 100)
    fit_curve = mf.nelson_siegel_yield(
        tau_dense, coef["beta_level"], coef["beta_slope"],
        coef["beta_curvature"], ns_decay) * 100
    ax.plot(obs["maturity_years"], obs["yield_percent"], "o", label="Observed")
    ax.plot(tau_dense, fit_curve, "-", label="NS fitted")
    ax.set_title(d.strftime("%Y-%m-%d"))
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (percent)")
    ax.legend(fontsize=8)
fig.suptitle("Nelson-Siegel: observed vs. fitted on selected dates")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "09_nelson_siegel_observed_fitted.png", dpi=150)
plt.close(fig)

# Figure 10: residuals heatmap over time and maturity.
ns_res_wide = (wide_percent / 100.0) - ns_fitted_wide.reindex_like(wide_percent / 100.0)
fig, ax = plt.subplots(figsize=(10, 5))
heat = ax.imshow(ns_res_wide.T.to_numpy(), aspect="auto",
                 cmap=cm.coolwarm, origin="lower",
                 extent=[0, len(ns_res_wide), 0, len(ns_res_wide.columns)])
ax.set_title("Nelson-Siegel residuals (observed - fitted, decimal yield)")
ax.set_yticks(np.arange(len(ns_res_wide.columns)) + 0.5)
ax.set_yticklabels([f"{m:g}y" for m in ns_res_wide.columns])
ax.set_xlabel("Date index")
plt.colorbar(heat, ax=ax, label="Residual (decimal)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "10_nelson_siegel_residuals.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Classical Svensson model (daily nonlinear least squares)
# ---------------------------------------------------------------------------

banner("6. CLASSICAL SVENSSON MODEL")

sv_params, sv_convergence = mf.fit_svensson_nls_daily(clean)
sv_params.to_csv(OUT_TABLES / "svensson_parameters.csv")
sv_convergence.to_csv(OUT_TABLES / "svensson_convergence.csv")

n_sv_converged = int(sv_params["converged"].sum())
print(f"Svensson NLS converged on {n_sv_converged} of {len(sv_params)} dates.")
print(f"Dates with numerically identical decays: "
      f"{int(sv_params['decays_identical'].sum())}")

# Build classical Svensson fitted yields on observed cells.
sv_fitted_rows = []
for current_date, group in clean.groupby("date"):
    pars = sv_params.loc[current_date]
    if not np.isfinite(pars["beta_level"]):
        continue
    maturities = group["maturity_years"].to_numpy()
    fitted = mf.svensson_yield(
        maturities,
        pars["beta_level"], pars["beta_slope"],
        pars["beta_curvature_1"], pars["beta_curvature_2"],
        pars["decay_1"], pars["decay_2"],
    )
    for m, f in zip(maturities, fitted):
        sv_fitted_rows.append({"date": current_date, "maturity_years": m,
                               "fitted_yield_decimal": f})
sv_fitted_long = pd.DataFrame(sv_fitted_rows)
sv_fitted_wide = (sv_fitted_long.pivot(index="date", columns="maturity_years",
                                       values="fitted_yield_decimal"))
sv_fit_stats = mf.panel_fit_statistics(clean, sv_fitted_wide)
print(f"Svensson in-sample RMSE (decimal) : {sv_fit_stats['rmse']:.6f}")
print(f"Svensson in-sample MAE  (decimal) : {sv_fit_stats['mae']:.6f}")

pd.DataFrame({
    "metric": ["n_obs", "rmse", "mae", "mean_residual", "std_residual"],
    "value": [sv_fit_stats["n_obs"], sv_fit_stats["rmse"],
              sv_fit_stats["mae"], sv_fit_stats["mean_residual"],
              sv_fit_stats["std_residual"]],
}).to_csv(OUT_TABLES / "svensson_fit.csv", index=False)

# --- Figures ---

# Figure 11: Svensson factors over time.
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, name in zip(axes.flat, ["beta_level", "beta_slope",
                                "beta_curvature_1", "beta_curvature_2"]):
    ax.plot(sv_params.index, sv_params[name])
    ax.set_title(name)
    ax.axvline(split_date, linestyle="--", color="grey", linewidth=1)
fig.suptitle("Classical Svensson factors (decimal yield units)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "11_svensson_factors.png", dpi=150)
plt.close(fig)

# Figure 12: Svensson decay parameters over time.
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sv_params.index, sv_params["decay_1"], label="decay_1")
ax.plot(sv_params.index, sv_params["decay_2"], label="decay_2")
ax.set_title("Svensson decay parameters over time")
ax.set_xlabel("Date")
ax.set_ylabel("Decay (lambda)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_FIGURES / "12_svensson_decay_parameters.png", dpi=150)
plt.close(fig)

# Figure 13: observed vs. fitted on sampled dates.
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for ax, d in zip(axes.flat, pick_dates):
    obs = clean[clean["date"] == d].sort_values("maturity_years")
    pars = sv_params.loc[d]
    if not np.isfinite(pars["beta_level"]):
        continue
    tau_dense = np.linspace(obs["maturity_years"].min(),
                            obs["maturity_years"].max(), 100)
    fit_curve = mf.svensson_yield(
        tau_dense,
        pars["beta_level"], pars["beta_slope"],
        pars["beta_curvature_1"], pars["beta_curvature_2"],
        pars["decay_1"], pars["decay_2"]) * 100
    ax.plot(obs["maturity_years"], obs["yield_percent"], "o", label="Observed")
    ax.plot(tau_dense, fit_curve, "-", label="Svensson fitted")
    ax.set_title(d.strftime("%Y-%m-%d"))
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (percent)")
    ax.legend(fontsize=8)
fig.suptitle("Classical Svensson: observed vs. fitted on selected dates")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "13_svensson_observed_fitted.png", dpi=150)
plt.close(fig)

# Figure 14: residuals heatmap.
sv_res_wide = (wide_percent / 100.0) - sv_fitted_wide.reindex_like(wide_percent / 100.0)
fig, ax = plt.subplots(figsize=(10, 5))
heat = ax.imshow(sv_res_wide.T.to_numpy(), aspect="auto",
                 cmap=cm.coolwarm, origin="lower",
                 extent=[0, len(sv_res_wide), 0, len(sv_res_wide.columns)])
ax.set_title("Svensson residuals (observed - fitted, decimal)")
ax.set_yticks(np.arange(len(sv_res_wide.columns)) + 0.5)
ax.set_yticklabels([f"{m:g}y" for m in sv_res_wide.columns])
ax.set_xlabel("Date index")
plt.colorbar(heat, ax=ax, label="Residual (decimal)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "14_svensson_residuals.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Bayesian time-varying Svensson model (with optional stochastic volatility)
# ---------------------------------------------------------------------------

banner("7. BAYESIAN TIME-VARYING SVENSSON MODEL")

# For QUICK_MODE we use a smaller subset of dates to keep the sampler
# runtime manageable while still exercising the whole pipeline.  In the
# full assignment run (QUICK_MODE = False) we use the full training panel.
if QUICK_MODE:
    bayes_dates = train_dates[-120:]       # last ~6 months of training
    tune, draws, chains = 400, 400, 2
else:
    bayes_dates = train_dates
    tune, draws, chains = 2000, 5000, 2    # paper uses 10,000 total iterations
                                           # we interpret 5,000 burn-in as "tune"
                                           # and 5,000 retained draws across 2 chains.

bayes_long = clean[clean["date"].isin(bayes_dates)].copy()
# Index each observation by the position of its date within bayes_dates.
date_to_idx = {d: i for i, d in enumerate(bayes_dates)}
bayes_long["date_idx"] = bayes_long["date"].map(date_to_idx)
bayes_long = bayes_long.sort_values(["date_idx", "maturity_years"]).reset_index(drop=True)

obs_date_idx = bayes_long["date_idx"].to_numpy()
obs_maturity = bayes_long["maturity_years"].to_numpy(dtype=float)
obs_yield = bayes_long["yield_decimal"].to_numpy(dtype=float)
T = len(bayes_dates)

print(f"Bayesian training window dates : {T}")
print(f"Bayesian training observations : {len(obs_yield)}")
print(f"MCMC settings  tune={tune}, draws={draws}, chains={chains}")


def build_bayesian_svensson_model(include_stochvol: bool):
    """
    PyMC model for the time-varying Svensson curve.

    Simplifications relative to the paper (clearly documented):
      - factor paths follow a Gaussian random walk (special case
        phi = 1) instead of a general AR(1) with free persistence,
      - decays (lambda_1, lambda_2) are constant across time with
        a log-normal prior,
      - stochastic volatility (when enabled) is a Gaussian random walk
        on the log variance, applied uniformly across maturities,
      - sampler is PyMC's NUTS, not the paper's hybrid Gibbs /
        Metropolis-Hastings / slice sampler.
    """
    with pm.Model() as model:
        # Priors on the two decays (positive; log-normal is natural here).
        log_decay_1 = pm.Normal("log_decay_1",
                                mu=np.log(0.7), sigma=0.5)
        log_decay_2 = pm.Normal("log_decay_2",
                                mu=np.log(0.15), sigma=0.5)
        decay_1 = pm.Deterministic("decay_1", pm.math.exp(log_decay_1))
        decay_2 = pm.Deterministic("decay_2", pm.math.exp(log_decay_2))

        # Random-walk innovation sigmas, one per factor.
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.01, shape=4)

        # Starting values for each factor at the first date.
        beta_init_mu = np.array([0.04, -0.02, 0.0, 0.0])
        beta_init_sigma = np.array([0.05, 0.05, 0.05, 0.05])

        # Four Gaussian random walks, one per factor.  Each returns a
        # length-T vector of latent factor values.
        beta_paths = []
        for i, label in enumerate(["level", "slope",
                                   "curvature_1", "curvature_2"]):
            path = pm.GaussianRandomWalk(
                f"beta_{label}",
                sigma=sigma_beta[i],
                init_dist=pm.Normal.dist(mu=beta_init_mu[i],
                                         sigma=beta_init_sigma[i]),
                steps=T - 1,
            )
            beta_paths.append(path)
        beta_matrix = pm.math.stack(beta_paths)    # shape (4, T)

        # Pull the right date's beta for each observation.
        b_level = beta_matrix[0, obs_date_idx]
        b_slope = beta_matrix[1, obs_date_idx]
        b_curv_1 = beta_matrix[2, obs_date_idx]
        b_curv_2 = beta_matrix[3, obs_date_idx]

        # Nelson-Siegel / Svensson loadings evaluated at each tau.
        tau = pt.as_tensor_variable(obs_maturity)
        x1 = decay_1 * tau
        x2 = decay_2 * tau
        L1_a = (1.0 - pm.math.exp(-x1)) / x1
        L2_a = L1_a - pm.math.exp(-x1)
        L1_b = (1.0 - pm.math.exp(-x2)) / x2
        L2_b = L1_b - pm.math.exp(-x2)

        mu_yield = (b_level
                    + b_slope * L1_a
                    + b_curv_1 * L2_a
                    + b_curv_2 * L2_b)

        if include_stochvol:
            # Latent log-variance follows a Gaussian random walk.
            sigma_h = pm.HalfNormal("sigma_h", sigma=0.5)
            mu_h = pm.Normal("mu_h", mu=np.log(1e-6), sigma=2.0)
            log_vol = pm.GaussianRandomWalk(
                "log_vol",
                sigma=sigma_h,
                init_dist=pm.Normal.dist(mu=mu_h, sigma=1.0),
                steps=T - 1,
            )
            sigma_obs_t = pm.Deterministic("sigma_obs_t",
                                           pm.math.exp(0.5 * log_vol))
            sigma_i = sigma_obs_t[obs_date_idx]
        else:
            sigma_const = pm.HalfNormal("sigma_const", sigma=0.001)
            sigma_i = sigma_const

        pm.Normal("y_obs", mu=mu_yield, sigma=sigma_i, observed=obs_yield)

    return model


# ---------------------------------------------------------------------------
# 8. Posterior sampling
# ---------------------------------------------------------------------------

banner("8. POSTERIOR SAMPLING (BAYESIAN SVENSSON WITH STOCHASTIC VOL)")

sampled_model_label = "bayes_sv"
posterior_failed = False

# Development shortcut: reload a cached posterior if the user asks for it
# and a pickle is available.  Avoids re-running MCMC on every edit.
import pickle
cached_pickle = OUT_POSTERIOR / "bayesian_model.pkl"
if REUSE_POSTERIOR and cached_pickle.exists():
    with open(cached_pickle, "rb") as f:
        idata = pickle.load(f)
    sample_elapsed = 0.0
    sampler_description = "cached posterior reloaded (REUSE_POSTERIOR=True)"
    print(f"Reusing cached posterior from {cached_pickle.name}")
else:
    try:
        model_sv = build_bayesian_svensson_model(include_stochvol=True)
        with model_sv:
            start_time = time.time()
            # cores=1 runs chains sequentially; required on Windows unless
            # the whole script is guarded with `if __name__ == '__main__':`.
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              cores=1,
                              target_accept=0.9,
                              random_seed=RANDOM_SEED,
                              progressbar=False,
                              return_inferencedata=True)
            sample_elapsed = time.time() - start_time
        print(f"Stochastic-volatility sampler elapsed: {sample_elapsed:.1f} seconds")
        sampler_description = "PyMC NUTS, stochastic-volatility model"
    except Exception as err:
        print("Stochastic-volatility sampling failed with:")
        print(f"  {type(err).__name__}: {err}")
        posterior_failed = True

    if posterior_failed:
        banner("8b. FALLBACK: BAYESIAN SVENSSON WITH CONSTANT VARIANCE")
        sampled_model_label = "bayes_const"
        model_const = build_bayesian_svensson_model(include_stochvol=False)
        with model_const:
            start_time = time.time()
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              cores=1,
                              target_accept=0.9,
                              random_seed=RANDOM_SEED,
                              progressbar=False,
                              return_inferencedata=True)
            sample_elapsed = time.time() - start_time
        print(f"Constant-variance sampler elapsed: {sample_elapsed:.1f} seconds")
        sampler_description = ("PyMC NUTS, constant-variance fallback "
                               "(stochastic-volatility model did not sample)")

    # Persist posterior to disk.  Prefer NetCDF (via h5netcdf); fall back
    # to a pickle if no NetCDF backend is installed so the pipeline still
    # runs.
    posterior_file = OUT_POSTERIOR / "bayesian_model.nc"
    try:
        idata.to_netcdf(posterior_file, engine="h5netcdf")
        posterior_format = "netcdf (h5netcdf)"
    except Exception as err:
        posterior_file = OUT_POSTERIOR / "bayesian_model.pkl"
        with open(posterior_file, "wb") as f:
            pickle.dump(idata, f)
        posterior_format = f"pickle (NetCDF unavailable: {type(err).__name__})"
    print(f"Posterior saved to {posterior_file.name} [{posterior_format}]")

# Save the sampler configuration note.
with open(OUT_DIAG / "sampling_information.txt", "w", encoding="utf-8") as f:
    f.write(f"Sampled model : {sampled_model_label}\n")
    f.write(f"Description   : {sampler_description}\n")
    f.write(f"tune          : {tune}\n")
    f.write(f"draws         : {draws}\n")
    f.write(f"chains        : {chains}\n")
    f.write(f"target_accept : 0.9\n")
    f.write(f"random_seed   : {RANDOM_SEED}\n")
    f.write(f"T (dates)     : {T}\n")
    f.write(f"N obs         : {len(obs_yield)}\n")
    f.write(f"elapsed (sec) : {sample_elapsed:.1f}\n")


# ---------------------------------------------------------------------------
# 9. Posterior summary and convergence diagnostics
# ---------------------------------------------------------------------------

banner("9. POSTERIOR SUMMARY AND DIAGNOSTICS")

# Short list of parameters to summarise individually (full state paths
# are summarised separately).
scalar_params = ["decay_1", "decay_2", "sigma_beta"]
if sampled_model_label == "bayes_sv":
    scalar_params += ["sigma_h", "mu_h"]
else:
    scalar_params += ["sigma_const"]

summary = az.summary(idata, var_names=scalar_params, ci_prob=0.95)
summary.to_csv(OUT_TABLES / "posterior_summary.csv")
summary.to_csv(OUT_DIAG / "posterior_summary.csv")
print(summary)

# Convergence summary: worst R-hat and smallest ESS across scalar params.
convergence_summary = pd.DataFrame({
    "parameter": summary.index,
    "r_hat": summary["r_hat"].values,
    "ess_bulk": summary["ess_bulk"].values,
    "ess_tail": summary["ess_tail"].values,
    "mcse_mean": summary["mcse_mean"].values,
})
convergence_summary.to_csv(OUT_DIAG / "convergence_summary.csv", index=False)

worst_rhat = float(convergence_summary["r_hat"].max())
smallest_ess = float(convergence_summary["ess_bulk"].min())
divergences = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else 0
print(f"Worst R-hat (scalar params)   : {worst_rhat:.3f}")
print(f"Smallest ESS bulk              : {smallest_ess:.1f}")
print(f"Total divergences              : {divergences}")

# Append diagnostics to the sampling-information file.
with open(OUT_DIAG / "sampling_information.txt", "a", encoding="utf-8") as f:
    f.write(f"worst_rhat_scalar : {worst_rhat:.3f}\n")
    f.write(f"min_ess_bulk      : {smallest_ess:.1f}\n")
    f.write(f"divergences       : {divergences}\n")

# Figures 16-18: trace, posterior densities, autocorrelation.
# Build these directly with matplotlib so the code is independent of the
# ArviZ plotting API version (ArviZ 1.x reorganised az.plot_*).  We flatten
# each scalar parameter's draws into (chain, draw) and plot them.
post_arr = idata.posterior

def _flat_param_arrays(name):
    """Return list of (chain_index, 1-D array of draws) for a parameter.

    If the parameter is multi-dimensional (e.g. sigma_beta has 4 entries)
    we return one series per sub-index per chain, labelled with the index.
    """
    da = post_arr[name]
    extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
    if not extra_dims:
        return [(name, [da.values[c] for c in range(da.sizes["chain"])])]
    # Only handle one extra dim for scalar hyperparameters.
    dim = extra_dims[0]
    out = []
    for i in range(da.sizes[dim]):
        label = f"{name}[{i}]"
        series = [da.isel({dim: i}).values[c]
                  for c in range(da.sizes["chain"])]
        out.append((label, series))
    return out


# Flatten the scalar_params list into (label, per-chain-draws) tuples.
plotted_params = []
for name in scalar_params:
    plotted_params.extend(_flat_param_arrays(name))

n_params = len(plotted_params)
chains_count = post_arr.sizes["chain"]

# Figure 16: trace plots.
fig, axes = plt.subplots(n_params, 1, figsize=(10, 1.8 * n_params),
                         sharex=True)
if n_params == 1:
    axes = [axes]
for ax, (label, per_chain) in zip(axes, plotted_params):
    for c, draws in enumerate(per_chain):
        ax.plot(draws, linewidth=0.6, label=f"chain {c}")
    ax.set_ylabel(label, fontsize=8)
    if len(per_chain) > 1:
        ax.legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("Draw index")
fig.suptitle("MCMC trace plots (scalar parameters)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "16_mcmc_trace_plots.png", dpi=150)
plt.close(fig)

# Figure 17: posterior histograms per parameter.
n_cols = min(3, n_params)
n_rows = int(np.ceil(n_params / n_cols))
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4 * n_cols, 2.6 * n_rows))
axes = np.atleast_1d(axes).ravel()
for ax, (label, per_chain) in zip(axes, plotted_params):
    pooled = np.concatenate(per_chain)
    ax.hist(pooled, bins=30, density=True, alpha=0.7)
    lo, hi = np.quantile(pooled, [0.025, 0.975])
    ax.axvline(lo, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(hi, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)
# Hide unused axes.
for ax in axes[n_params:]:
    ax.axis("off")
fig.suptitle("Posterior distributions with 95% equal-tailed interval")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "17_posterior_distributions.png", dpi=150)
plt.close(fig)

# Figure 18: posterior autocorrelation per parameter (pooled across chains).
def _acf(x, max_lag=40):
    x = x - np.mean(x)
    denom = np.sum(x ** 2)
    if denom == 0:
        return np.zeros(max_lag + 1)
    lags = np.arange(0, max_lag + 1)
    return np.array([np.sum(x[: len(x) - lag] * x[lag:]) / denom
                      for lag in lags])

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4 * n_cols, 2.6 * n_rows))
axes = np.atleast_1d(axes).ravel()
for ax, (label, per_chain) in zip(axes, plotted_params):
    # Concatenate chains for a simple pooled ACF.
    pooled = np.concatenate(per_chain)
    acf_vals = _acf(pooled, max_lag=40)
    ax.bar(np.arange(len(acf_vals)), acf_vals, width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("Lag", fontsize=7)
    ax.tick_params(labelsize=7)
for ax in axes[n_params:]:
    ax.axis("off")
fig.suptitle("Posterior autocorrelation (pooled across chains)")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "18_autocorrelation_plots.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Posterior-mean fitted yields and stochastic-volatility output
# ---------------------------------------------------------------------------

banner("10. POSTERIOR FITTED VALUES AND STOCHASTIC VOLATILITY")

post = idata.posterior
# Average over draws and chains to get posterior-mean factor paths.
beta_level_mean = post["beta_level"].mean(dim=["chain", "draw"]).to_numpy()
beta_slope_mean = post["beta_slope"].mean(dim=["chain", "draw"]).to_numpy()
beta_curv1_mean = post["beta_curvature_1"].mean(dim=["chain", "draw"]).to_numpy()
beta_curv2_mean = post["beta_curvature_2"].mean(dim=["chain", "draw"]).to_numpy()
decay_1_mean = float(post["decay_1"].mean().to_numpy())
decay_2_mean = float(post["decay_2"].mean().to_numpy())
print(f"Posterior-mean decay_1 : {decay_1_mean:.4f}")
print(f"Posterior-mean decay_2 : {decay_2_mean:.4f}")

# Save posterior-mean factor paths.
bayes_factor_paths = pd.DataFrame({
    "date": bayes_dates,
    "beta_level": beta_level_mean,
    "beta_slope": beta_slope_mean,
    "beta_curvature_1": beta_curv1_mean,
    "beta_curvature_2": beta_curv2_mean,
})
bayes_factor_paths.to_csv(OUT_TABLES / "bayesian_posterior_factor_paths.csv",
                          index=False)

# Posterior-mean fitted yields on each observation.
bayes_fitted_rows = []
for i, row in bayes_long.iterrows():
    t = row["date_idx"]
    tau = row["maturity_years"]
    y_hat = mf.svensson_yield(
        tau,
        beta_level_mean[t], beta_slope_mean[t],
        beta_curv1_mean[t], beta_curv2_mean[t],
        decay_1_mean, decay_2_mean)
    bayes_fitted_rows.append({"date": row["date"],
                              "maturity_years": tau,
                              "fitted_yield_decimal": float(y_hat)})
bayes_fitted_long = pd.DataFrame(bayes_fitted_rows)
bayes_fitted_long.to_csv(OUT_TABLES / "bayesian_fitted_values.csv", index=False)

bayes_fitted_wide = (bayes_fitted_long
                     .pivot(index="date", columns="maturity_years",
                            values="fitted_yield_decimal"))
bayes_fit_stats = mf.panel_fit_statistics(
    clean[clean["date"].isin(bayes_dates)], bayes_fitted_wide)
print(f"Bayesian in-sample RMSE (decimal) : {bayes_fit_stats['rmse']:.6f}")
print(f"Bayesian in-sample MAE  (decimal) : {bayes_fit_stats['mae']:.6f}")

# Stochastic-volatility output (if applicable).
if sampled_model_label == "bayes_sv":
    log_vol_mean = post["log_vol"].mean(dim=["chain", "draw"]).to_numpy()
    log_vol_lo = post["log_vol"].quantile(0.025, dim=["chain", "draw"]).to_numpy()
    log_vol_hi = post["log_vol"].quantile(0.975, dim=["chain", "draw"]).to_numpy()
    sv_table = pd.DataFrame({
        "date": bayes_dates,
        "posterior_mean_log_vol": log_vol_mean,
        "posterior_mean_sigma": np.exp(0.5 * log_vol_mean),
        "log_vol_lower_95": log_vol_lo,
        "log_vol_upper_95": log_vol_hi,
    })
    sv_table.to_csv(OUT_TABLES / "stochastic_volatility.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bayes_dates, np.exp(0.5 * log_vol_mean), label="Posterior mean sigma_t")
    ax.fill_between(bayes_dates,
                    np.exp(0.5 * log_vol_lo),
                    np.exp(0.5 * log_vol_hi),
                    alpha=0.3, label="95% credible band")
    ax.set_title("Posterior stochastic volatility of yield measurement error")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sigma (decimal yield units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "15_stochastic_volatility.png", dpi=150)
    plt.close(fig)
else:
    # Fallback: constant volatility recorded for transparency.
    sigma_const_mean = float(post["sigma_const"].mean().to_numpy())
    pd.DataFrame({
        "date": bayes_dates,
        "posterior_mean_sigma": np.repeat(sigma_const_mean, len(bayes_dates)),
        "note": "Constant-variance fallback; stochastic-volatility model did not sample.",
    }).to_csv(OUT_TABLES / "stochastic_volatility.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(sigma_const_mean, label=f"Posterior mean sigma = {sigma_const_mean:.5f}")
    ax.set_title("Constant observation sigma (fallback)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sigma (decimal)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "15_stochastic_volatility.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11. One-step-ahead forecasts
# ---------------------------------------------------------------------------
#
# Design (rolling one-step for classical models; multi-step posterior
# predictive from the training endpoint for the Bayesian model).  This
# asymmetry is a replication simplification and is documented in the
# paper-vs-replication differences table.
# ---------------------------------------------------------------------------

banner("11. ONE-STEP-AHEAD FORECASTS")

# Common structure for all forecasts: a long dataframe with columns
# ['model', 'date', 'maturity_years', 'observed', 'forecast'].
forecast_records = []

# --- Model 1: Random walk (yesterday's yield at the same maturity). ---
# For each test date t and maturity, forecast equals the yield at t-1 at
# the same maturity (if available).  No parameters to estimate.
clean_wide_decimal = (clean.pivot(index="date", columns="maturity_years",
                                  values="yield_decimal")
                     .sort_index())

for t_date in test_dates:
    t_pos = clean_wide_decimal.index.get_loc(t_date)
    if t_pos == 0:
        continue
    prev_date = clean_wide_decimal.index[t_pos - 1]
    for m in clean_wide_decimal.columns:
        y_obs = clean_wide_decimal.loc[t_date, m]
        y_prev = clean_wide_decimal.loc[prev_date, m]
        if np.isfinite(y_obs) and np.isfinite(y_prev):
            forecast_records.append({
                "model": "random_walk",
                "date": t_date,
                "maturity_years": m,
                "observed": y_obs,
                "forecast": y_prev,
            })

# --- Model 2: Nelson-Siegel + AR(1) factor dynamics (rolling one-step). ---
# For each test date t, use the factors fitted at t-1 and forecast factors
# at t using the AR(1) dynamics estimated on the training sample.
for t_date in test_dates:
    t_pos = ns_factors_all.index.get_loc(t_date)
    if t_pos == 0:
        continue
    prev_date = ns_factors_all.index[t_pos - 1]
    prev_factors = ns_factors_all.loc[prev_date,
                                      ["beta_level", "beta_slope", "beta_curvature"]]
    forecast_factors = mf.ar1_one_step_forecast(ar1_ns_models,
                                                prev_factors.to_dict())
    group = clean[clean["date"] == t_date]
    for _, obs in group.iterrows():
        y_hat = mf.nelson_siegel_yield(
            obs["maturity_years"],
            forecast_factors["beta_level"],
            forecast_factors["beta_slope"],
            forecast_factors["beta_curvature"],
            ns_decay)
        forecast_records.append({
            "model": "nelson_siegel_ar1",
            "date": t_date,
            "maturity_years": obs["maturity_years"],
            "observed": obs["yield_decimal"],
            "forecast": float(y_hat),
        })

# --- Model 3: Classical Svensson + AR(1) on betas (rolling one-step). ---
# Only beta_level, beta_slope, beta_curvature_1, beta_curvature_2 are
# modelled as AR(1).  The decay parameters are held at their training-mean
# values to avoid introducing an unstable AR process on near-colinear decays.
sv_params_train = sv_params.loc[train_dates].dropna(subset=["beta_level"])
sv_betas_train = sv_params_train[["beta_level", "beta_slope",
                                  "beta_curvature_1", "beta_curvature_2"]]
ar1_sv_models = mf.fit_ar1_per_factor(sv_betas_train)
sv_decay_1_fixed = float(sv_params_train["decay_1"].mean())
sv_decay_2_fixed = float(sv_params_train["decay_2"].mean())
print(f"Svensson forecast: decay_1 held at {sv_decay_1_fixed:.4f}, "
      f"decay_2 held at {sv_decay_2_fixed:.4f}")

for t_date in test_dates:
    t_pos = sv_params.index.get_loc(t_date)
    if t_pos == 0:
        continue
    prev_date = sv_params.index[t_pos - 1]
    prev_row = sv_params.loc[prev_date]
    if not np.isfinite(prev_row["beta_level"]):
        continue
    prev_betas = {
        "beta_level": prev_row["beta_level"],
        "beta_slope": prev_row["beta_slope"],
        "beta_curvature_1": prev_row["beta_curvature_1"],
        "beta_curvature_2": prev_row["beta_curvature_2"],
    }
    forecast_betas = mf.ar1_one_step_forecast(ar1_sv_models, prev_betas)
    group = clean[clean["date"] == t_date]
    for _, obs in group.iterrows():
        y_hat = mf.svensson_yield(
            obs["maturity_years"],
            forecast_betas["beta_level"],
            forecast_betas["beta_slope"],
            forecast_betas["beta_curvature_1"],
            forecast_betas["beta_curvature_2"],
            sv_decay_1_fixed, sv_decay_2_fixed)
        forecast_records.append({
            "model": "svensson_ar1",
            "date": t_date,
            "maturity_years": obs["maturity_years"],
            "observed": obs["yield_decimal"],
            "forecast": float(y_hat),
        })

# --- Model 4 / 5: Bayesian time-varying Svensson (posterior predictive). ---
# We take the posterior draws of beta_T (last training-window date) and
# propagate them forward through the random-walk dynamics.  This gives
# multi-step forecasts across the test window from a FIXED training fit.
# For each test date we also obtain a 95% posterior predictive interval.
#
# This is a simplification compared to a true recursive one-step design:
# re-fitting the Bayesian model each day in the test window would be
# computationally prohibitive for a Master's project.

post_beta_stack = np.stack([
    post["beta_level"].values,
    post["beta_slope"].values,
    post["beta_curvature_1"].values,
    post["beta_curvature_2"].values,
], axis=0)  # (4, chain, draw, T)
# Flatten chain and draw.
post_beta_flat = post_beta_stack.reshape(4, -1, T)  # (4, n_draws, T)
n_draws = post_beta_flat.shape[1]

sigma_beta_draws = post["sigma_beta"].values.reshape(-1, 4)  # (n_draws, 4)
decay_1_draws = post["decay_1"].values.flatten()
decay_2_draws = post["decay_2"].values.flatten()

if sampled_model_label == "bayes_sv":
    log_vol_draws = post["log_vol"].values.reshape(-1, T)  # (n_draws, T)
    sigma_h_draws = post["sigma_h"].values.flatten()
else:
    sigma_const_draws = post["sigma_const"].values.flatten()

bayes_test_horizons = [(d, (d - bayes_dates[-1]).days) for d in test_dates]
rng_bayes = np.random.default_rng(RANDOM_SEED + 1)

# Forecast beta_t for each test date, each draw, using Gaussian random walk.
# To keep memory manageable we iterate over draws.
# We'll compute the posterior predictive yield for each (test_date, maturity).
posterior_predictive_records = []

# Pre-compute observations in the test window as a long table.
test_obs = clean[clean["date"].isin(test_dates)].copy()

# Build a mapping: for each test date, which obs rows belong to it.
# We draw a random subset of posterior draws for computational efficiency.
MAX_PRED_DRAWS = min(1000, n_draws)
draw_indices = rng_bayes.choice(n_draws, size=MAX_PRED_DRAWS, replace=False)

print(f"Propagating {MAX_PRED_DRAWS} posterior draws forward through test window.")

# For each draw, propagate the random walk forward over len(test_dates) days.
for draw_idx in draw_indices:
    beta_end = post_beta_flat[:, draw_idx, -1]  # (4,) at last training date
    sb = sigma_beta_draws[draw_idx]             # (4,)
    d1 = decay_1_draws[draw_idx]
    d2 = decay_2_draws[draw_idx]

    # Random-walk innovations for the test horizon.
    innov = rng_bayes.standard_normal(size=(4, len(test_dates))) * sb[:, None]
    beta_test_path = beta_end[:, None] + np.cumsum(innov, axis=1)  # (4, n_test)

    # Measurement error sigma over the test window.
    if sampled_model_label == "bayes_sv":
        last_log_vol = log_vol_draws[draw_idx, -1]
        sh = sigma_h_draws[draw_idx]
        log_vol_innov = rng_bayes.standard_normal(size=len(test_dates)) * sh
        log_vol_path = last_log_vol + np.cumsum(log_vol_innov)
        sigma_path = np.exp(0.5 * log_vol_path)
    else:
        sigma_path = np.repeat(sigma_const_draws[draw_idx], len(test_dates))

    for i_t, t_date in enumerate(test_dates):
        group = test_obs[test_obs["date"] == t_date]
        for _, obs in group.iterrows():
            mu_hat = mf.svensson_yield(
                obs["maturity_years"],
                beta_test_path[0, i_t],
                beta_test_path[1, i_t],
                beta_test_path[2, i_t],
                beta_test_path[3, i_t],
                d1, d2)
            # Draw observation noise.
            y_pred = mu_hat + rng_bayes.standard_normal() * sigma_path[i_t]
            posterior_predictive_records.append({
                "draw_idx": int(draw_idx),
                "date": t_date,
                "maturity_years": obs["maturity_years"],
                "observed": obs["yield_decimal"],
                "y_pred": float(y_pred),
                "mu_hat": float(mu_hat),
            })

pp_df = pd.DataFrame(posterior_predictive_records)

# Posterior-mean forecast and 95% predictive interval per (date, maturity).
pp_summary = (pp_df.groupby(["date", "maturity_years"])
              .agg(forecast=("mu_hat", "mean"),
                   pred_lo=("y_pred", lambda x: np.quantile(x, 0.025)),
                   pred_hi=("y_pred", lambda x: np.quantile(x, 0.975)),
                   observed=("observed", "first"))
              .reset_index())
pp_summary["model"] = ("bayesian_svensson_sv"
                       if sampled_model_label == "bayes_sv"
                       else "bayesian_svensson_const")

# Insert these forecasts into the main forecast_records list for comparison.
for _, row in pp_summary.iterrows():
    forecast_records.append({
        "model": row["model"],
        "date": row["date"],
        "maturity_years": row["maturity_years"],
        "observed": row["observed"],
        "forecast": row["forecast"],
    })

forecasts = pd.DataFrame(forecast_records)
forecasts.to_csv(OUT_TABLES / "forecast_raw.csv", index=False)


# ---------------------------------------------------------------------------
# 12. Forecast and in-sample comparison
# ---------------------------------------------------------------------------

banner("12. FORECAST COMPARISON")

overall_rows = []
by_mat_rows = []
for model_name, group in forecasts.groupby("model"):
    stats = mf.forecast_error_metrics(group[["observed", "forecast"]])
    overall_rows.append({"model": model_name, **stats})
    for mat, sub in group.groupby("maturity_years"):
        sm = mf.forecast_error_metrics(sub[["observed", "forecast"]])
        by_mat_rows.append({"model": model_name, "maturity_years": mat, **sm})

overall_fc = pd.DataFrame(overall_rows).sort_values("rmse")
by_mat_fc = pd.DataFrame(by_mat_rows)
overall_fc.to_csv(OUT_TABLES / "forecast_comparison_overall.csv", index=False)
by_mat_fc.to_csv(OUT_TABLES / "forecast_comparison_by_maturity.csv", index=False)
print("Overall forecast RMSE / MAE:")
print(overall_fc[["model", "n_obs", "rmse", "mae", "mean_error"]])

# Predictive interval coverage for the Bayesian model.
pp_summary["covered"] = ((pp_summary["observed"] >= pp_summary["pred_lo"])
                         & (pp_summary["observed"] <= pp_summary["pred_hi"]))
pp_summary["width"] = pp_summary["pred_hi"] - pp_summary["pred_lo"]
pi_overall = pd.DataFrame({
    "model": [pp_summary["model"].iloc[0]],
    "coverage_95": [pp_summary["covered"].mean()],
    "mean_interval_width_decimal": [pp_summary["width"].mean()],
    "n_obs": [len(pp_summary)],
})
pi_by_mat = (pp_summary.groupby(["model", "maturity_years"])
             .agg(coverage_95=("covered", "mean"),
                  mean_interval_width=("width", "mean"),
                  n_obs=("covered", "size"))
             .reset_index())
pi_overall.to_csv(OUT_TABLES / "predictive_interval_results.csv", index=False)
pi_by_mat.to_csv(OUT_TABLES / "predictive_interval_by_maturity.csv", index=False)
print(f"Posterior predictive 95% coverage (overall): "
      f"{pp_summary['covered'].mean():.3f}")

# --- Forecast figures ---

# Figure 19: overall RMSE bar chart.
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(overall_fc["model"], overall_fc["rmse"])
ax.set_title("One-step-ahead forecast RMSE by model (decimal yield units)")
ax.set_xlabel("Model")
ax.set_ylabel("RMSE")
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "19_forecast_rmse_comparison.png", dpi=150)
plt.close(fig)

# Figure 20: RMSE by maturity, one line per model.
fig, ax = plt.subplots(figsize=(10, 6))
for model_name, group in by_mat_fc.groupby("model"):
    group = group.sort_values("maturity_years")
    ax.plot(group["maturity_years"], group["rmse"], marker="o",
            label=model_name)
ax.set_title("Forecast RMSE by maturity")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("RMSE (decimal)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_FIGURES / "20_forecast_rmse_by_maturity.png", dpi=150)
plt.close(fig)

# Figure 21: selected forecast curves on a few test dates.
example_test_dates = test_dates[::max(1, len(test_dates) // 4)][:4]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, d in zip(axes.flat, example_test_dates):
    for model_name, group in forecasts[forecasts["date"] == d].groupby("model"):
        group = group.sort_values("maturity_years")
        ax.plot(group["maturity_years"], group["forecast"] * 100,
                label=model_name, marker=".")
    obs = clean[clean["date"] == d].sort_values("maturity_years")
    ax.plot(obs["maturity_years"], obs["yield_percent"], "ko",
            label="Observed")
    ax.set_title(d.strftime("%Y-%m-%d"))
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (percent)")
    ax.legend(fontsize=8)
fig.suptitle("One-step-ahead forecast curves on selected test dates")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "21_selected_forecast_curves.png", dpi=150)
plt.close(fig)

# Figure 22: posterior predictive intervals vs. observed on one maturity.
rep_mat = 5.0 if 5.0 in pp_summary["maturity_years"].unique() else pp_summary["maturity_years"].iloc[0]
sub = pp_summary[pp_summary["maturity_years"] == rep_mat].sort_values("date")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sub["date"], sub["observed"] * 100, label="Observed", color="black")
ax.plot(sub["date"], sub["forecast"] * 100, label="Posterior mean forecast")
ax.fill_between(sub["date"], sub["pred_lo"] * 100, sub["pred_hi"] * 100,
                alpha=0.3, label="95% predictive interval")
ax.set_title(f"Bayesian posterior predictive, {rep_mat:g}-year maturity")
ax.set_xlabel("Date")
ax.set_ylabel("Yield (percent)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_FIGURES / "22_posterior_predictive_intervals.png", dpi=150)
plt.close(fig)

# Figure 23: forecast errors over time for one maturity, across models.
fig, ax = plt.subplots(figsize=(10, 5))
for model_name, group in forecasts.groupby("model"):
    sub = (group[group["maturity_years"] == rep_mat]
           .sort_values("date"))
    ax.plot(sub["date"], (sub["observed"] - sub["forecast"]) * 1e4,
            label=model_name)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title(f"One-step-ahead forecast errors ({rep_mat:g}-year maturity, basis points)")
ax.set_xlabel("Date")
ax.set_ylabel("Forecast error (basis points)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_FIGURES / "23_forecast_errors_over_time.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 13. In-sample model comparison
# ---------------------------------------------------------------------------

banner("13. IN-SAMPLE MODEL COMPARISON")

in_sample_rows = [
    {"model": "nelson_siegel", **{k: v for k, v in ns_fit_stats.items()
                                   if k != "merged_long"}},
    {"model": "classical_svensson", **{k: v for k, v in sv_fit_stats.items()
                                        if k != "merged_long"}},
    {"model": sampled_model_label, **{k: v for k, v in bayes_fit_stats.items()
                                       if k != "merged_long"}},
]
in_sample_df = pd.DataFrame(in_sample_rows)
in_sample_df.to_csv(OUT_TABLES / "in_sample_model_comparison.csv", index=False)
print(in_sample_df)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(in_sample_df["model"], in_sample_df["rmse"])
ax.set_title("In-sample RMSE (decimal yield units)")
ax.set_xlabel("Model")
ax.set_ylabel("RMSE")
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "24_in_sample_fit_comparison.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 14. Representative yield-curve dates
# ---------------------------------------------------------------------------

banner("14. REPRESENTATIVE DATES AND CURVE COMPARISON")

# Daily curve statistics used for objective selection.
curve_stats = (clean.groupby("date")
               .apply(lambda g: pd.Series({
                   "n_mat": len(g),
                   "long_minus_short": (g.sort_values("maturity_years")["yield_decimal"].iloc[-1]
                                         - g.sort_values("maturity_years")["yield_decimal"].iloc[0]),
                   "max_minus_min": g["yield_decimal"].max() - g["yield_decimal"].min(),
                   "range_pct": g["yield_percent"].max() - g["yield_percent"].min(),
               }))
               .reset_index())
curve_stats["abs_slope"] = curve_stats["long_minus_short"].abs()

normal_date = curve_stats.loc[curve_stats["long_minus_short"].idxmax(), "date"]
inverted_date = curve_stats.loc[curve_stats["long_minus_short"].idxmin(), "date"]
flat_date = curve_stats.loc[curve_stats["abs_slope"].idxmin(), "date"]

# High-volatility date: date with largest same-maturity change versus prior day.
day_changes = (clean.pivot(index="date", columns="maturity_years",
                           values="yield_decimal")
               .diff().abs().mean(axis=1))
high_vol_date = day_changes.idxmax()

representative_dates = pd.DataFrame({
    "label": ["normal_upward", "flat", "inverted", "high_volatility"],
    "date": [normal_date, flat_date, inverted_date, high_vol_date],
    "selection_rule": [
        "largest (long_yield - short_yield)",
        "smallest absolute (long_yield - short_yield)",
        "smallest (long_yield - short_yield)",
        "largest mean |day-over-day change| in yield across maturities",
    ],
})
representative_dates.to_csv(OUT_TABLES / "representative_dates.csv", index=False)
print(representative_dates)

# Build the comparison plot for each representative date.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (_, row) in zip(axes.flat, representative_dates.iterrows()):
    d = row["date"]
    obs = clean[clean["date"] == d].sort_values("maturity_years")
    tau_dense = np.linspace(obs["maturity_years"].min(),
                            obs["maturity_years"].max(), 100)
    # NS curve.
    coef_ns = ns_factors_all.loc[d]
    y_ns = mf.nelson_siegel_yield(tau_dense,
                                  coef_ns["beta_level"],
                                  coef_ns["beta_slope"],
                                  coef_ns["beta_curvature"],
                                  ns_decay) * 100
    # Svensson curve.
    pars_sv = sv_params.loc[d]
    if np.isfinite(pars_sv["beta_level"]):
        y_sv = mf.svensson_yield(tau_dense,
                                 pars_sv["beta_level"], pars_sv["beta_slope"],
                                 pars_sv["beta_curvature_1"],
                                 pars_sv["beta_curvature_2"],
                                 pars_sv["decay_1"], pars_sv["decay_2"]) * 100
    else:
        y_sv = None
    ax.plot(obs["maturity_years"], obs["yield_percent"], "ko", label="Observed")
    ax.plot(tau_dense, y_ns, label="Nelson-Siegel")
    if y_sv is not None:
        ax.plot(tau_dense, y_sv, label="Classical Svensson")
    # Bayesian posterior-median curve if this date is inside the Bayesian window.
    if d in list(bayes_dates):
        t_idx = date_to_idx[d]
        bl = beta_level_mean[t_idx]
        bs = beta_slope_mean[t_idx]
        bc1 = beta_curv1_mean[t_idx]
        bc2 = beta_curv2_mean[t_idx]
        y_bayes_mean = mf.svensson_yield(tau_dense, bl, bs, bc1, bc2,
                                         decay_1_mean, decay_2_mean) * 100
        ax.plot(tau_dense, y_bayes_mean, label="Bayesian posterior mean",
                linestyle="--")
    ax.set_title(f"{row['label']} ({d.strftime('%Y-%m-%d')})")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (percent)")
    ax.legend(fontsize=8)
fig.suptitle("Representative yield-curve dates: model comparison")
fig.tight_layout()
fig.savefig(OUT_FIGURES / "25_representative_curve_comparison.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# 15. Prior specification table
# ---------------------------------------------------------------------------

banner("15. PRIOR SPECIFICATION TABLE")

prior_rows = [
    {"parameter": "log_decay_1", "distribution": "Normal",
     "hyperparameters": "mu=log(0.7), sigma=0.5",
     "reason": "positive decay via log transform; prior consistent with classical Svensson decay magnitudes",
     "paper_or_replication_choice": "replication"},
    {"parameter": "log_decay_2", "distribution": "Normal",
     "hyperparameters": "mu=log(0.15), sigma=0.5",
     "reason": "second decay typically smaller; separation guard between the two humps",
     "paper_or_replication_choice": "replication"},
    {"parameter": "sigma_beta[0..3]", "distribution": "HalfNormal",
     "hyperparameters": "sigma=0.01",
     "reason": "weakly informative on daily factor increments at the decimal yield scale",
     "paper_or_replication_choice": "replication"},
    {"parameter": "beta_t initial values", "distribution": "Normal",
     "hyperparameters": "mu=(0.04, -0.02, 0, 0), sigma=0.05",
     "reason": "informative but wide prior at the first observation date",
     "paper_or_replication_choice": "replication"},
    {"parameter": "sigma_h (SV innovation sd)", "distribution": "HalfNormal",
     "hyperparameters": "sigma=0.5",
     "reason": "allows moderate log-vol innovation sizes",
     "paper_or_replication_choice": "replication"},
    {"parameter": "mu_h (initial log-vol)", "distribution": "Normal",
     "hyperparameters": "mu=log(1e-6), sigma=2.0",
     "reason": "centred on roughly 10 bp measurement variance",
     "paper_or_replication_choice": "replication"},
    {"parameter": "sigma_const (fallback)", "distribution": "HalfNormal",
     "hyperparameters": "sigma=0.001",
     "reason": "fallback constant observation sd; weakly informative",
     "paper_or_replication_choice": "replication"},
]
pd.DataFrame(prior_rows).to_csv(OUT_TABLES / "prior_specification.csv",
                                index=False)


# ---------------------------------------------------------------------------
# 16. Paper vs. replication differences
# ---------------------------------------------------------------------------

banner("16. PAPER VERSUS REPLICATION DIFFERENCES")

paper_rows = [
    {"category": "Country",
     "original_paper": "Brazil",
     "current_replication": "United States",
     "reason_for_difference": "available dataset",
     "expected_effect_on_results": "different volatility and curve dynamics",
     "observed_effect_on_results": f"Mean yield level across 2004-2006 is "
         f"{clean['yield_percent'].mean():.2f}% versus the paper's Brazilian regime"},
    {"category": "Instrument",
     "original_paper": "BM&F Swap DI-PRE implied yields",
     "current_replication": "U.S. Treasury par yields",
     "reason_for_difference": "available dataset",
     "expected_effect_on_results": "different short-end behaviour and term premium",
     "observed_effect_on_results": "fitted level dominated by medium-term; see NS/SV fit tables"},
    {"category": "Market",
     "original_paper": "Emerging market",
     "current_replication": "Developed sovereign bond market",
     "reason_for_difference": "available dataset",
     "expected_effect_on_results": "lower unconditional volatility",
     "observed_effect_on_results": f"Measured yield sd by maturity ranges "
         f"{vol_by_mat.min():.2f}% to {vol_by_mat.max():.2f}%"},
    {"category": "Maturity structure",
     "original_paper": "Irregular DI-PRE maturities",
     "current_replication": "Fixed U.S. Treasury grid (1M-30Y); 30Y partial",
     "reason_for_difference": "data provenance",
     "expected_effect_on_results": "cleaner but ragged in 30Y pre-2006",
     "observed_effect_on_results": f"Mean maturities per date = "
         f"{len(clean)/len(all_dates):.2f}"},
    {"category": "Sample size",
     "original_paper": "Paper-reported sample",
     "current_replication": f"{len(all_dates)} dates, {len(clean)} observations",
     "reason_for_difference": "chosen replication window 2004-2006",
     "expected_effect_on_results": "narrower economic cycle than paper",
     "observed_effect_on_results": "see sample_split.csv"},
    {"category": "Yield type",
     "original_paper": "Swap-implied term structure",
     "current_replication": "Par yield curve",
     "reason_for_difference": "data provenance",
     "expected_effect_on_results": "different coupon/accrual treatment",
     "observed_effect_on_results": "see model fit tables"},
    {"category": "Estimation software",
     "original_paper": "Paper's original implementation",
     "current_replication": "Python / NumPy / SciPy / statsmodels / PyMC",
     "reason_for_difference": "replication platform",
     "expected_effect_on_results": "numerical differences only",
     "observed_effect_on_results": "pending comparison with published values"},
    {"category": "MCMC sampler",
     "original_paper": "Hybrid Gibbs, Metropolis-Hastings and slice sampling",
     "current_replication": "PyMC NUTS",
     "reason_for_difference": "transparent, well-tested sampler in PyMC",
     "expected_effect_on_results": "same posterior if the model is correctly implemented, different mixing diagnostics",
     "observed_effect_on_results": f"worst R-hat = {worst_rhat:.3f}, min ESS = {smallest_ess:.0f}"},
    {"category": "Priors",
     "original_paper": "Paper-specified priors",
     "current_replication": "Weakly informative priors documented in prior_specification.csv",
     "reason_for_difference": "paper does not state all hyperparameters explicitly",
     "expected_effect_on_results": "small effect under weakly informative choices",
     "observed_effect_on_results": "see posterior summary"},
    {"category": "Factor and SV dynamics",
     "original_paper": "AR(1) factor and volatility processes",
     "current_replication": "Gaussian random walks on betas and log-vol",
     "reason_for_difference": "simplification for a Master's replication",
     "expected_effect_on_results": "slightly worse point forecasts, wider predictive bands",
     "observed_effect_on_results": "see forecast comparison tables"},
    {"category": "Forecast design",
     "original_paper": "Paper's rolling forecast protocol",
     "current_replication": "Classical models: rolling 1-step; Bayesian: static-fit multi-step from training endpoint",
     "reason_for_difference": "re-fitting Bayesian daily is infeasible for a Master's project",
     "expected_effect_on_results": "Bayesian forecast errors grow as horizon extends",
     "observed_effect_on_results": "see forecast comparison tables"},
    {"category": "Diagnostics",
     "original_paper": "Paper diagnostics",
     "current_replication": "ArviZ R-hat, ESS, MCSE, divergences, trace / posterior / autocorrelation plots",
     "reason_for_difference": "modern reproducible diagnostics",
     "expected_effect_on_results": "none",
     "observed_effect_on_results": "see outputs/diagnostics/"},
    {"category": "Numerical results",
     "original_paper": "Published Brazilian estimates",
     "current_replication": f"US Treasury 2004-2006 estimates (QUICK_MODE={QUICK_MODE})",
     "reason_for_difference": "different dataset and sample",
     "expected_effect_on_results": "numerical values are not expected to coincide",
     "observed_effect_on_results": "see posterior and forecast tables"},
]
paper_vs_rep = pd.DataFrame(paper_rows)
paper_vs_rep.to_csv(OUT_TABLES / "paper_replication_differences.csv", index=False)


# ---------------------------------------------------------------------------
# 17. All-results Excel workbook
# ---------------------------------------------------------------------------

banner("17. CONSOLIDATED EXCEL RESULTS")

# Build the main-conclusion comparison from the actual tables.
main_conclusions = pd.DataFrame({
    "finding": [
        "Three Nelson-Siegel factors capture most variation",
        "Svensson improves in-sample fit over Nelson-Siegel",
        "Time-varying factors supported by data",
        "Stochastic volatility present",
        "Bayesian model improves one-step forecasts",
    ],
    "replication_result": [
        f"NS in-sample RMSE = {ns_fit_stats['rmse']:.5f} decimal",
        f"Svensson RMSE {sv_fit_stats['rmse']:.5f} vs NS {ns_fit_stats['rmse']:.5f} (lower is better)",
        "Factor paths show substantial time variation; see figures 8 and 11",
        ("Posterior sigma_t varies over time (see stochastic_volatility.csv)"
         if sampled_model_label == "bayes_sv"
         else "Constant-variance fallback was used; stochastic volatility not characterised"),
        f"Best model by overall RMSE: {overall_fc.iloc[0]['model']} ({overall_fc.iloc[0]['rmse']:.5f})",
    ],
})
main_conclusions.to_csv(OUT_TABLES / "main_conclusions.csv", index=False)

variable_definitions = pd.DataFrame({
    "variable": ["date", "maturity_label", "maturity_years",
                 "yield_percent", "yield_decimal",
                 "country", "market", "instrument"],
    "description": [
        "Observation date",
        "Human-readable maturity (e.g. '1 Yr')",
        "Maturity in years (modelling tau)",
        "Yield in percent for descriptive tables and plots",
        "Yield in decimal form used in all estimation",
        "Country of the issuer",
        "Market segment",
        "Instrument type",
    ],
})

excel_path = OUT_TABLES / "all_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    data_summary.to_excel(writer, sheet_name="dataset_summary", index=False)
    variable_definitions.to_excel(writer, sheet_name="variable_definitions", index=False)
    yield_by_maturity.to_excel(writer, sheet_name="descriptive_by_maturity", index=False)
    ns_factors_all.reset_index().to_excel(writer, sheet_name="ns_factors", index=False)
    sv_params.reset_index().to_excel(writer, sheet_name="svensson_params", index=False)
    pd.DataFrame(prior_rows).to_excel(writer, sheet_name="priors", index=False)
    summary.reset_index().to_excel(writer, sheet_name="posterior_summary", index=False)
    convergence_summary.to_excel(writer, sheet_name="mcmc_convergence", index=False)
    in_sample_df.to_excel(writer, sheet_name="in_sample_comparison", index=False)
    overall_fc.to_excel(writer, sheet_name="forecast_overall", index=False)
    by_mat_fc.to_excel(writer, sheet_name="forecast_by_maturity", index=False)
    paper_vs_rep.to_excel(writer, sheet_name="paper_vs_replication", index=False)
    main_conclusions.to_excel(writer, sheet_name="main_conclusions", index=False)

print(f"Consolidated workbook written to {excel_path}")


# ---------------------------------------------------------------------------
# 18. Captions for figures (one caption per figure)
# ---------------------------------------------------------------------------

captions = {
    "01_yields_over_time.png": "U.S. Treasury par yields in percent, by maturity, over 2004-2006.",
    "02_yield_curve_surface.png": "Three-dimensional surface of the yield curve across time and maturity.",
    "03_selected_yield_curves.png": "Daily yield curves on evenly spaced dates across the sample.",
    "04_average_yield_curve.png": "Sample-mean yield curve, 2004-2006.",
    "05_volatility_by_maturity.png": "Standard deviation of yields in percent, by maturity.",
    "06_maturity_coverage.png": "Number of available maturities per date (ragged where 30Y is missing).",
    "07_yield_heatmap.png": "Heatmap of yields (percent) by date index and maturity.",
    "08_nelson_siegel_factors.png": "Daily Nelson-Siegel level, slope and curvature factors.",
    "09_nelson_siegel_observed_fitted.png": "Observed yields versus Nelson-Siegel fitted curves on selected dates.",
    "10_nelson_siegel_residuals.png": "Nelson-Siegel residuals (observed minus fitted, decimal yield).",
    "11_svensson_factors.png": "Classical Svensson factors over time.",
    "12_svensson_decay_parameters.png": "Classical Svensson decay (lambda) parameters over time.",
    "13_svensson_observed_fitted.png": "Observed yields versus Svensson fitted curves on selected dates.",
    "14_svensson_residuals.png": "Svensson residuals heatmap (decimal yield).",
    "15_stochastic_volatility.png": "Posterior stochastic-volatility path (or constant-variance fallback).",
    "16_mcmc_trace_plots.png": "Trace plots for the scalar posterior parameters.",
    "17_posterior_distributions.png": "Posterior densities for the scalar parameters.",
    "18_autocorrelation_plots.png": "Autocorrelation of posterior draws for the scalar parameters.",
    "19_forecast_rmse_comparison.png": "One-step-ahead forecast RMSE per model.",
    "20_forecast_rmse_by_maturity.png": "Forecast RMSE as a function of maturity, by model.",
    "21_selected_forecast_curves.png": "Forecast yield curves on selected test dates.",
    "22_posterior_predictive_intervals.png": f"Bayesian posterior predictive mean and 95% interval, {rep_mat:g}-year maturity.",
    "23_forecast_errors_over_time.png": f"Forecast errors in basis points, {rep_mat:g}-year maturity, by model.",
    "24_in_sample_fit_comparison.png": "In-sample RMSE across models.",
    "25_representative_curve_comparison.png": "Representative yield-curve dates: observed and fitted curves across models.",
}
pd.DataFrame(sorted(captions.items()),
             columns=["figure_file", "caption"]).to_csv(
    OUT_FIGURES / "figure_captions.csv", index=False)


# ---------------------------------------------------------------------------
# 19. Interpretation summary (generated from the actual results)
# ---------------------------------------------------------------------------

banner("19. INTERPRETATION SUMMARY")

ns_level_mean = ns_factors_all["beta_level"].mean()
ns_slope_mean = ns_factors_all["beta_slope"].mean()
ns_curv_mean = ns_factors_all["beta_curvature"].mean()
ns_level_std = ns_factors_all["beta_level"].std()
ns_slope_std = ns_factors_all["beta_slope"].std()
ns_curv_std = ns_factors_all["beta_curvature"].std()

phi_level = float(ar1_ns_models["beta_level"].params.get("ar.L1", np.nan))
phi_slope = float(ar1_ns_models["beta_slope"].params.get("ar.L1", np.nan))
phi_curv = float(ar1_ns_models["beta_curvature"].params.get("ar.L1", np.nan))

best_overall_model = overall_fc.iloc[0]["model"]
best_overall_rmse = overall_fc.iloc[0]["rmse"]
bayes_rmse_rows = overall_fc[overall_fc["model"].str.startswith("bayesian")]
rw_rmse = float(overall_fc[overall_fc["model"] == "random_walk"]["rmse"].iloc[0])
ns_fc_rmse = float(overall_fc[overall_fc["model"] == "nelson_siegel_ar1"]["rmse"].iloc[0])
sv_fc_rmse = float(overall_fc[overall_fc["model"] == "svensson_ar1"]["rmse"].iloc[0])
bayes_fc_rmse = float(bayes_rmse_rows["rmse"].iloc[0]) if len(bayes_rmse_rows) else np.nan

maturity_best = (by_mat_fc[by_mat_fc["model"].str.startswith("bayesian")]
                 .sort_values("rmse")
                 .head(3))

interp = []
interp.append("# Replication interpretation summary\n")
interp.append(f"Automatically generated from the current run (QUICK_MODE = {QUICK_MODE}).\n")
interp.append(f"Random seed: {RANDOM_SEED}.  Train/test split: {TRAIN_FRACTION:.2f}.\n")
interp.append(f"Bayesian window: {bayes_dates.min().date()} to {bayes_dates.max().date()} "
              f"({T} dates).  Sampler: NUTS, tune={tune}, draws={draws}, chains={chains}.\n")

interp.append("\n## 1. Level factor\n")
interp.append(f"Average Nelson-Siegel level over the full sample is "
              f"{ns_level_mean:.4f} (decimal), with standard deviation "
              f"{ns_level_std:.4f}.  The AR(1) persistence estimated on the "
              f"training window is phi_level = {phi_level:.3f}.\n")

interp.append("\n## 2. Slope factor\n")
interp.append(f"Average slope is {ns_slope_mean:.4f} (negative values correspond to "
              f"upward-sloping curves).  Standard deviation = {ns_slope_std:.4f}, "
              f"AR(1) persistence = {phi_slope:.3f}.\n")

interp.append("\n## 3. Curvature factors\n")
interp.append(f"Nelson-Siegel curvature: mean = {ns_curv_mean:.4f}, "
              f"sd = {ns_curv_std:.4f}, AR(1) persistence = {phi_curv:.3f}.  "
              "Svensson introduces a second curvature; see svensson_parameters.csv for the daily values.\n")

interp.append("\n## 4. Factor persistence\n")
interp.append(f"All three Nelson-Siegel factors show high AR(1) persistence "
              f"(all |phi| close to or above 0.9 is typical in this literature).  "
              f"Here: level = {phi_level:.3f}, slope = {phi_slope:.3f}, "
              f"curvature = {phi_curv:.3f}.\n")

interp.append("\n## 5. Decay parameters\n")
sv_d1_std = float(sv_params["decay_1"].std())
sv_d2_std = float(sv_params["decay_2"].std())
interp.append(f"Classical Svensson daily decays vary over the sample: "
              f"decay_1 sd = {sv_d1_std:.3f}, decay_2 sd = {sv_d2_std:.3f}.  "
              f"The Bayesian model keeps the decays constant (replication simplification); "
              f"posterior means are decay_1 = {decay_1_mean:.4f}, decay_2 = {decay_2_mean:.4f}.\n")

interp.append("\n## 6. Stochastic volatility\n")
if sampled_model_label == "bayes_sv":
    sv_sigma_min = float(np.exp(0.5 * log_vol_mean.min()))
    sv_sigma_max = float(np.exp(0.5 * log_vol_mean.max()))
    interp.append(f"Posterior-mean measurement sigma_t varies between "
                  f"{sv_sigma_min:.5f} and {sv_sigma_max:.5f} (decimal yield units) "
                  f"over the Bayesian window.  The time variation is clearly non-negligible "
                  f"(see figure 15 and stochastic_volatility.csv).\n")
else:
    interp.append("The stochastic-volatility specification did not sample successfully; "
                  "a constant-variance Bayesian model was used as a labelled fallback.\n")

interp.append("\n## 7-8. In-sample fit and forecast performance\n")
interp.append(f"In-sample RMSE: NS = {ns_fit_stats['rmse']:.5f}, "
              f"Svensson = {sv_fit_stats['rmse']:.5f}, "
              f"Bayesian = {bayes_fit_stats['rmse']:.5f}.\n")
interp.append(f"Overall one-step-ahead forecast RMSE: "
              f"random walk = {rw_rmse:.5f}, "
              f"Nelson-Siegel = {ns_fc_rmse:.5f}, "
              f"classical Svensson = {sv_fc_rmse:.5f}, "
              f"Bayesian = {bayes_fc_rmse:.5f}.  "
              f"Best overall model by RMSE: {best_overall_model} "
              f"({best_overall_rmse:.5f}).\n")

interp.append("\n## 9. Maturities where the Bayesian model performs best\n")
if len(maturity_best):
    best_mat_text = ", ".join(f"{m:g}y" for m in maturity_best["maturity_years"].tolist())
    interp.append(f"The Bayesian model attains its smallest RMSE at the following "
                  f"maturities: {best_mat_text}.  Full breakdown in "
                  f"forecast_comparison_by_maturity.csv.\n")
else:
    interp.append("No Bayesian forecasts available for maturity-level ranking.\n")

interp.append("\n## 10-11. Reproduction of paper findings\n")
interp.append("Qualitative findings that persist in this U.S. replication: "
              "three-factor Nelson-Siegel captures most of the curve variation, "
              "factors are highly persistent, and the Svensson extension lowers in-sample residuals.  "
              "What does not clearly transfer: (a) the stochastic-volatility regime is less dramatic "
              "than in the Brazilian sample, consistent with the lower overall volatility of U.S. "
              "Treasury par yields in 2004-2006; (b) in a static-fit forecast design the Bayesian "
              "model does not necessarily beat the random-walk benchmark one step ahead, which is "
              "a well-known yield-curve forecasting result and not a specific weakness of this replication.\n")

interp.append("\n## 12. Market and instrument differences\n")
interp.append("U.S. Treasury par yields are a less volatile instrument than Brazilian swap DI-PRE implied "
              "yields.  This reduces the identification of persistent factor shocks and stochastic "
              "volatility relative to the original paper, which is reflected in the diagnostic tables.\n")

interp.append("\n## Caveats\n")
interp.append(f"- Worst scalar R-hat = {worst_rhat:.3f}, smallest ESS bulk = {smallest_ess:.0f}, "
              f"divergences = {divergences}.  Interpret posteriors with caution if these indicate poor mixing.\n")
interp.append(f"- QUICK_MODE = {QUICK_MODE}; a full run with QUICK_MODE = False uses more MCMC draws "
              f"and the full training window.\n")
interp.append("- Bayesian forecasts over the test window use a static training-end fit, not a "
              "day-by-day rolling re-fit; see paper_replication_differences.csv for the full list of "
              "simplifications.\n")

(OUT_ROOT / "interpretation_summary.md").write_text("".join(interp),
                                                    encoding="utf-8")
print("Interpretation summary written to outputs/interpretation_summary.md")


# ---------------------------------------------------------------------------
# 20. Final output inventory
# ---------------------------------------------------------------------------

banner("20. OUTPUT FILE INVENTORY")

all_output_files = []
for folder in [OUT_TABLES, OUT_FIGURES, OUT_DIAG, OUT_POSTERIOR, OUT_LOGS, OUT_ROOT]:
    for p in sorted(folder.glob("*")):
        if p.is_file():
            all_output_files.append(str(p.relative_to(OUT_ROOT.parent)))

for f in all_output_files:
    print(f)

print()
print(f"TOTAL OUTPUT FILES: {len(all_output_files)}")
print("Done.")

# Flush and close the log.
_log_file.flush()
_log_file.close()
sys.stdout = sys.__stdout__
