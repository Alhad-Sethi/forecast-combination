# Can anything beat 1/N? Forecast combination for German day-ahead electricity prices

## The problem

When several forecasts of the same quantity are available, the simplest way to combine them is to average them with
equal weights (1/N). Methods that *learn* the weights should do better, yet in practice they often lose to the plain
average: estimating weights adds noise that eats up the gain. This is the **forecast combination puzzle**
(Smith & Wallis 2009; Genre et al. 2013).

This project tests the puzzle on **German day-ahead electricity prices** (ENTSO-E Transparency Platform, 2015 onward).
Each day, before the auction closes at 12:00 CET on d-1, we forecast the 24 hourly prices of day d with a pool of
15 forecasting models ("experts") and combine them. Electricity prices give long, free, high-frequency data with
calm years, a COVID shock and the 2021-22 energy crisis, so a combiner has to cope with very different regimes.

## Hypothesis

> Can an online combiner beat equal weights **consistently** in a walk-forward setting?

Three challengers, none of them the favourite, against 1/N:

| Combiner | Idea |
|---|---|
| **1/N** | Plain average of all experts (the benchmark). |
| **EWA** | Exponentially weighted average: weights ∝ exp(-eta × discounted past loss), starting uniform; eta and the forgetting half-life are selected on training years. Variants: weights per hour, and a reset each test year. |
| **BOA** | Gradient Bernstein Online Aggregation, the fully adaptive version of Wintenberger (2017); no hyperparameters. |
| **Tree** | A shallow regression tree that learns a correction to 1/N from the experts' deviations from 1/N, the hour and the weekday. |

## Data and protocol

- **Target:** 24 hourly day-ahead prices (EUR/MWh) for Germany (DE-AT-LU before October 2018, DE-LU after).
- **Inputs:** lagged German prices (d-1, d-2, d-3, d-7), German load forecasts (d, d-1, d-7), German wind + solar
  forecasts (d-1, d-7), weekday dummies; some experts also see French prices (d-1) and the French load forecast (d).
- **No look-ahead:** an input is used only if EU Regulation 543/2013 requires its publication before the 12:00
  auction close. Load forecasts are due two hours before the close and are used for day d; wind and solar forecasts
  are due only by 18:00 on d-1, so the forecast for day d is **not** used (only the one for d-1).
- **Cleaning:** hourly averaging, daylight-saving fixes, gap filling, causal winsorizing of load; outlier days
  (detected from the inputs only) are excluded from training but kept in evaluation.
- **Walk-forward:** each combiner is tuned on 3 years and tested on the next one (2020, 2021, 2022, 2023). Losses are
  scaled by the recent error of a seasonal naive forecast so that crisis days don't dominate.
- **Holdout:** 2024 onward is reserved for a single final run once the configuration is frozen. It has not been run.
- **Measures:** MAE and the standard deviation of hourly errors (EUR/MWh), plus bias.

## The experts

| Expert | Description |
|---|---|
| `naive` | Same hour yesterday (same hour last week for Monday, Saturday and Sunday). |
| `arx` | Linear regression per hour on recent prices, yesterday's min/max/last price, load and weekday. |
| `lear_56`, `lear_84`, `lear_112`, `lear_364`, `lear_728` | LASSO per hour on all 223 inputs (Lago et al. 2021), trained on the last 56 to 728 days. |
| `lear_fr_112`, `lear_fr_728` | The same LASSO with 48 extra French inputs, trained on the last 112 or 728 days. |
| `lgbm` | One gradient-boosted tree model (LightGBM) for all 24 hours, refit quarterly. |
| `lgbm_fr` | `lgbm` with the French inputs. |
| `lgbm_change` | LightGBM predicting the change from yesterday's same-hour price instead of the price level. |
| `lgbm_change_fr` | `lgbm_change` with the French inputs. |
| `rf` | Random forest predicting the change from yesterday's same-hour price. |
| `knn` | Similar days: averages the price changes of the most similar past days of the same day type. |

Prices are variance-stabilized with an asinh transform (Uniejewski, Weron & Ziel 2018) before fitting.

## Development results (2020-2023)

Walk-forward test years. Each year is out of sample for the combiners, but these years were used while designing
the pipeline, so they are development results, not the final holdout.

**MAE (EUR/MWh), lower is better:**

| Method | 2020 | 2021 | 2022 | 2023 | Beats 1/N |
|---|---|---|---|---|---|
| 1/N | 6.07 | 20.28 | 43.46 | 17.64 | |
| EWA | 5.90 | 16.99 | 40.20 | 17.30 | 4 of 4 |
| EWA per hour | 5.87 | 16.72 | 40.24 | 17.19 | 4 of 4 |
| EWA reset | 5.90 | 17.00 | 40.17 | 17.30 | 4 of 4 |
| BOA | **5.82** | 17.16 | **40.09** | **17.04** | 4 of 4 |
| Tree | 5.96 | 17.10 | 40.62 | 17.72 | 3 of 4 |
| Best single expert (hindsight) | 5.93 | 17.79 | 41.33 | 18.47 | |

**Error standard deviation (EUR/MWh):**

| Method | 2020 | 2021 | 2022 | 2023 | Beats 1/N |
|---|---|---|---|---|---|
| 1/N | 9.89 | 32.49 | 58.63 | 26.50 | |
| EWA | 9.78 | 27.86 | 56.21 | 25.99 | 4 of 4 |
| BOA | **9.64** | 27.66 | **55.28** | **25.65** | 4 of 4 |
| Tree | 9.72 | 27.71 | 56.13 | 25.78 | 4 of 4 |

### What we see so far

- **EWA and BOA beat 1/N in every test year**, on both MAE and error spread, and they also beat the best single
  expert chosen in hindsight. 1/N beats that expert in only 1 of 4 years.
- **Most of the gain comes from the crisis years** (about 15-18% lower MAE in 2021, 7-8% in 2022). Some experts that
  were good in calm years (the LightGBM models and the random forest) break when prices rise fast; learned weights
  drop them within days, while 1/N keeps giving them 1/15 each.
- **In calm years the gain is small** (2-4%): close to the classic puzzle, where 1/N is hard to beat.
- **Information matters more than the combiner.** Removing the day-d wind/solar forecast (to avoid look-ahead) made
  the experts 30-45% worse; adding French inputs gave the largest improvement of any change.
- Caveats: one market and four test years, and no formal significance tests yet. More markets and the frozen 2024+
  holdout are the next steps.

## Repository layout

```
src/download_entsoe.py   # download raw ENTSO-E data per year (needs an API token)
src/clean_entsoe.py      # clean to one row per (date, hour)
src/anomalies.py         # outlier-day detection from the inputs
src/features.py          # 223 day-level inputs (+48 French)
src/experts.py           # the 15 experts -> data/experts/
src/combine.py           # 1/N, EWA, BOA, shallow tree
src/walk_forward.py      # folds and hyperparameter selection -> data/results/
src/evaluate.py          # MAE, error std, bias, comparisons
src/build_notebook.py    # builds the notebooks
tests/test_combine.py    # hand-worked checks of the combiners
notebooks/               # data exploration and combiner diagnostics (weights over time)
```

## Reproducing

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export ENTSOE_API_KEY=...                         # free token from the ENTSO-E Transparency Platform
.venv/bin/python src/download_entsoe.py           # German data (slow API; resumable)
.venv/bin/python src/download_entsoe.py --market FR
.venv/bin/python src/clean_entsoe.py
.venv/bin/python src/clean_entsoe.py --market FR
.venv/bin/python src/anomalies.py
.venv/bin/python src/experts.py                   # ~15 min on a laptop
.venv/bin/python src/walk_forward.py              # ~4 min
.venv/bin/python src/evaluate.py
.venv/bin/python tests/test_combine.py
```

Data files are not in the repository (`data/` is gitignored); the scripts download and rebuild them.

## References

- Smith & Wallis (2009): a simple explanation of the forecast combination puzzle.
- Genre, Kenny, Meyler & Timmermann (2013): combining expert forecasts: can anything beat the simple average?
- Lago, Marcjasz, De Schutter & Weron (2021): forecasting day-ahead electricity prices: a review and open benchmark.
- Wintenberger (2017): optimal learning with Bernstein Online Aggregation.
- Gaillard & Goude: the `opera` R package for online expert aggregation.
- Uniejewski, Weron & Ziel (2018): variance-stabilizing transformations for electricity price forecasting.
- Diebold & Shin (2019): machine learning for regularized survey forecast combination.
