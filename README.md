# Bayesian Extensions to the Diebold-Li Term Structure Model — U.S. Treasury Replication

## Project

A methodological replication of:

> Laurini, M. P. and Hotta, L. K. *"Bayesian Extensions to the Diebold-Li Term Structure Model"*.

The paper models daily Brazilian BM&F Swap DI-PRÉ implied yield curves.
This project re-implements the paper's model specifications on daily
**U.S. Treasury par yield curve data for 2004-2006**, using Python.

This is **not** an exact replication:

- the dataset is different (U.S. Treasury par yields vs. Brazilian swap DI-PRÉ),
- the sampler is different (PyMC NUTS vs. the paper's hybrid Gibbs / Metropolis-Hastings / slice sampler),
- certain priors and factor-dynamics choices are replication-specific where the paper does not provide full detail.

The differences are tracked explicitly in
`outputs/tables/paper_replication_differences.csv`.

## Dataset

- **File:** `us_treasury_yield_curve_long_2004_2006.csv`
- **Source:** U.S. Department of the Treasury Daily Treasury Par Yield Curve Rates.
- **Period:** 2004-01-02 to 2006-12-29, business-day frequency.
- **Maturities:** 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y (30Y partial — Treasury suspended the 30Y bond until Feb 2006).
- **Format:** long (date × maturity).

All estimation uses `yield_decimal`.  Descriptive tables and plots use `yield_percent`.

## Python version and packages

Tested with **Python 3.11**.  Required packages are listed in
[requirements.txt](requirements.txt):

```
numpy pandas scipy matplotlib statsmodels pymc arviz openpyxl
```

Install with:

```
pip install -r requirements.txt
```

## File structure

```
us_treasury_yield_curve_long_2004_2006.csv
main.py
model_functions.py
requirements.txt
README.md
outputs/
    tables/
    figures/
    diagnostics/
    posterior_draws/
    logs/
    interpretation_summary.md
    clean_yield_curve_long.csv
```

- `main.py` runs the complete analysis end to end.
- `model_functions.py` contains the Nelson-Siegel and Svensson yield
  functions and the classical estimation routines (OLS factor fit,
  nonlinear-least-squares Svensson, AR(1) factor dynamics).

## How to run

From the project root directory:

```
python main.py
```

All outputs are written under `outputs/`.  A full console log is
automatically captured in `outputs/logs/complete_python_output.txt`.

## QUICK_MODE

At the top of `main.py` the flag `QUICK_MODE` controls the
computational budget:

- `QUICK_MODE = True` — Bayesian model estimated on the last ~120
  training dates with `tune = 400`, `draws = 400`, `chains = 2`.  The
  whole pipeline runs quickly and is suitable for development.
- `QUICK_MODE = False` — Bayesian model estimated on the full training
  window with `tune = 2000`, `draws = 5000`, `chains = 2`, which is close
  to the paper's reported 10,000 iterations with 5,000 burn-in and is
  intended for the final assignment run.

The random seed is `RANDOM_SEED = 20240101`.

## Train / test split

A chronological 80 / 20 split is used (replication choice, not a paper
specification).  The split date is reported in
`outputs/tables/sample_split.csv`.

## Known limitations

- The Bayesian forecasts over the test window use a static training-end
  fit propagated forward through the random walk, not a day-by-day
  rolling re-fit.  Re-fitting the Bayesian model every day in the test
  window would be computationally prohibitive for a Master's project.
- Factor dynamics are modelled with Gaussian random walks (phi = 1
  special case) instead of a general AR(1) with free persistence.  This
  is a documented simplification.
- Decay parameters `lambda_1` and `lambda_2` are kept constant across
  time inside the Bayesian model; only the classical Svensson model
  estimates them daily.
- The stochastic-volatility specification applies a single volatility
  state across all maturity observations rather than one state per
  maturity.
- If the stochastic-volatility sampler fails, the script falls back to a
  constant-variance Bayesian model and labels the result accordingly.
- Numerical results will not reproduce the paper's values because the
  dataset, instrument, country and sampler are all different.  See
  `outputs/tables/paper_replication_differences.csv` for the full list.
