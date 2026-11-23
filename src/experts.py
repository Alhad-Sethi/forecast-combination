"""Generate walk-forward forecasts from every expert and cache them in data/experts/.
Each model is fit only on days before the first day it forecasts; outlier days are never training targets."""

import sys
import time
import warnings
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path

from features import (
    DAYS_PER_WEEK,
    HOURS_PER_DAY,
    column_indices,
    hour_suffix,
    lag_columns,
    load_day_data,
    price_feature_indices,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXPERTS_DIR = PROJECT_DIR / "data" / "experts"
FORECASTS_PATH = EXPERTS_DIR / "forecasts.parquet"
LEAR_PENALTIES_PATH = EXPERTS_DIR / "lear_penalties.parquet"

FIRST_FORECAST_DATE = date(2017, 1, 1)
HOLDOUT_START = date(2024, 1, 1)
# Monday, Saturday and Sunday differ from the day before, so the naive forecast uses last week.
NAIVE_LAST_WEEK_WEEKDAYS = [1, 6, 7]

MAD_TO_STD = 1.4826
# Below this many usable training days a model is not fit.
MIN_TRAINING_DAYS = 28

ARX_WINDOW_DAYS = 364
ARX_REFIT_DAYS = 1
ARX_OWN_HOUR_PRICE_LAGS_DAYS = [1, 2, 7]
MW_PER_GW = 1000

LEAR_WINDOWS_DAYS = [56, 84, 112, 364, 728]
LEAR_REFIT_DAYS = 7
LEAR_CV_FOLDS = 4
LEAR_CV_FOLD_DAYS = 21
# Penalties searched on standardized inputs and asinh-scaled prices; the LARS path is read off at these.
LEAR_PENALTY_GRID = np.logspace(-4, 0, 30)
NUM_PARALLEL_JOBS = 4

LGBM_WINDOW_DAYS = 728
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "num_threads": 4,
    "random_state": 0,
    "verbose": -1,
}
# Same-hour copies spare the trees from learning which of 24 columns matches the target hour.
LGBM_OWN_HOUR_INPUTS = [("price", 1), ("price", 2), ("price", 7), ("load", 0), ("renewables", 0)]
QUARTER_START_MONTHS = [1, 4, 7, 10]
JANUARY = [1]


# ---------- shared helpers ----------


def fit_price_scale(prices):
    """Return the median and scaled MAD of `prices`, used to centre and scale before asinh."""
    median = np.nanmedian(prices)
    mad = np.nanmedian(np.abs(prices - median)) * MAD_TO_STD
    return median, mad


def to_asinh(prices, scale):
    """Centre and scale prices, then apply asinh to tame spikes and negative prices."""
    median, mad = scale
    return np.arcsinh((prices - median) / mad)


def from_asinh(values, scale):
    """Invert `to_asinh`."""
    median, mad = scale
    return np.sinh(values) * mad + median


def training_days(data, end_index, window_days):
    """Return usable training days in the `window_days` before `end_index`: complete and not outliers."""
    candidates = np.arange(max(0, end_index - window_days), end_index)
    is_usable = data.has_features[candidates] & data.has_target[candidates] & ~data.is_outlier[candidates]
    return candidates[is_usable]


def forecast_days(data, start_index, end_index):
    """Return days in [start_index, end_index) whose inputs are all available."""
    candidates = np.arange(start_index, end_index)
    return candidates[data.has_features[candidates]]


def forecast_index_range(data):
    """Return the first and one-past-last day index to forecast."""
    first_index = data.dates.index(FIRST_FORECAST_DATE)
    last_index = int(np.where(data.has_features)[0][-1])
    return first_index, last_index + 1


def every_n_days(first_index, end_index, step_days):
    """Return refit points every `step_days` from `first_index`."""
    return list(range(first_index, end_index, step_days))


def calendar_points(data, first_index, end_index, months):
    """Return `first_index` plus every first-of-month in `months` up to `end_index`."""
    points = [first_index]
    for index in range(first_index + 1, end_index):
        day = data.dates[index]
        if day.day == 1 and day.month in months:
            points.append(index)
    return points


def schedule_blocks(points, end_index):
    """Turn sorted refit points into (start, end) blocks, each forecast by one fitted model."""
    blocks = []
    for position, start in enumerate(points):
        if position + 1 < len(points):
            end = points[position + 1]
        else:
            end = end_index
        blocks.append((start, end))
    return blocks


def standardize(train_inputs, other_inputs):
    """Scale both arrays with the training mean and standard deviation (constant columns left unscaled)."""
    means = train_inputs.mean(axis=0)
    stds = train_inputs.std(axis=0)
    stds[stds == 0] = 1.0
    return (train_inputs - means) / stds, (other_inputs - means) / stds


# ---------- seasonal naive ----------


def naive_forecasts(data, first_index, end_index):
    """Same hour yesterday; same hour last week for Monday, Saturday and Sunday."""
    forecasts = np.full(data.targets.shape, np.nan)
    days = forecast_days(data, first_index, end_index)
    yesterday = data.features[days][:, column_indices(data, lag_columns("price", 1))]
    last_week = data.features[days][:, column_indices(data, lag_columns("price", 7))]
    uses_last_week = np.isin(data.weekdays[days], NAIVE_LAST_WEEK_WEEKDAYS)
    forecasts[days] = np.where(uses_last_week[:, np.newaxis], last_week, yesterday)
    return forecasts


# ---------- ARX ----------


def arx_inputs(data, days, hour, scale):
    """Return ARX inputs for one hour: own-hour price lags, yesterday's min/max/last price, load, weekday dummies."""
    own_hour_names = []
    for lag_days in ARX_OWN_HOUR_PRICE_LAGS_DAYS:
        own_hour_names.append(f"price_lag{lag_days}_{hour_suffix(hour)}")
    own_hour_prices = data.features[days][:, column_indices(data, own_hour_names)]
    yesterday = data.features[days][:, column_indices(data, lag_columns("price", 1))]
    yesterday_summary = np.column_stack([yesterday.min(axis=1), yesterday.max(axis=1), yesterday[:, -1]])
    prices = to_asinh(np.column_stack([own_hour_prices, yesterday_summary]), scale)

    load = data.features[days][:, column_indices(data, [f"load_lag0_{hour_suffix(hour)}"])]
    # Load in GW keeps the least-squares problem well scaled.
    load_gw = load / MW_PER_GW
    weekdays = data.features[days][:, column_indices(data, weekday_names())]
    return np.column_stack([prices, load_gw, weekdays])


def weekday_names():
    """Return the seven weekday dummy column names."""
    names = []
    for weekday in range(1, DAYS_PER_WEEK + 1):
        names.append(f"weekday_{weekday}")
    return names


def arx_forecasts(data, first_index, end_index):
    """Refit 24 hourly OLS models every day on the previous 364 days."""
    forecasts = np.full(data.targets.shape, np.nan)
    for start, end in schedule_blocks(every_n_days(first_index, end_index, ARX_REFIT_DAYS), end_index):
        train = training_days(data, start, ARX_WINDOW_DAYS)
        predict = forecast_days(data, start, end)
        if len(train) < MIN_TRAINING_DAYS or len(predict) == 0:
            continue
        scale = fit_price_scale(data.targets[train])
        for hour in range(HOURS_PER_DAY):
            train_target = to_asinh(data.targets[train, hour], scale)
            coefficients = np.linalg.lstsq(arx_inputs(data, train, hour, scale), train_target, rcond=None)[0]
            predicted = arx_inputs(data, predict, hour, scale) @ coefficients
            forecasts[predict, hour] = from_asinh(predicted, scale)
    return forecasts


# ---------- LEAR ----------


def lear_inputs(data, days, scale, price_indices):
    """Return all 247 inputs with the lagged prices asinh-transformed."""
    inputs = data.features[days].copy()
    inputs[:, price_indices] = to_asinh(inputs[:, price_indices], scale)
    return inputs


def coefficients_on_grid(path_penalties, path_coefficients, grid):
    """Read the piecewise-linear LARS path off at each grid penalty; returns (features, grid)."""
    ascending_penalties = path_penalties[::-1]
    ascending_coefficients = path_coefficients[:, ::-1]
    on_grid = np.empty((path_coefficients.shape[0], len(grid)))
    for feature in range(path_coefficients.shape[0]):
        on_grid[feature] = np.interp(grid, ascending_penalties, ascending_coefficients[feature])
    return on_grid


def lear_fold_errors(data, train, validation, price_indices):
    """Return the validation MAE (EUR/MWh) of every grid penalty, per hour: shape (24, grid)."""
    scale = fit_price_scale(data.targets[train])
    train_inputs, validation_inputs = standardize(
        lear_inputs(data, train, scale, price_indices),
        lear_inputs(data, validation, scale, price_indices),
    )
    errors = np.empty((HOURS_PER_DAY, len(LEAR_PENALTY_GRID)))
    for hour in range(HOURS_PER_DAY):
        train_target = to_asinh(data.targets[train, hour], scale)
        target_mean = train_target.mean()
        path_penalties, _, path_coefficients = lars_path(train_inputs, train_target - target_mean, method="lasso")
        grid_coefficients = coefficients_on_grid(path_penalties, path_coefficients, LEAR_PENALTY_GRID)
        predictions = from_asinh(validation_inputs @ grid_coefficients + target_mean, scale)
        actual = data.targets[validation, hour][:, np.newaxis]
        errors[hour] = np.abs(predictions - actual).mean(axis=0)
    return errors


def tune_lear_penalties(data, retune_index, window_days, price_indices):
    """Pick each hour's penalty by rolling-origin CV over the 4 x 21 days before `retune_index`."""
    fold_errors = []
    for fold in range(LEAR_CV_FOLDS):
        validation_start = retune_index - (LEAR_CV_FOLDS - fold) * LEAR_CV_FOLD_DAYS
        validation = training_days(data, validation_start + LEAR_CV_FOLD_DAYS, LEAR_CV_FOLD_DAYS)
        train = training_days(data, validation_start, window_days)
        if len(train) < MIN_TRAINING_DAYS or len(validation) == 0:
            continue
        fold_errors.append(lear_fold_errors(data, train, validation, price_indices))
    mean_errors = np.mean(fold_errors, axis=0)
    return LEAR_PENALTY_GRID[np.argmin(mean_errors, axis=1)]


def fit_and_predict_lear(data, train, predict, penalties, price_indices):
    """Fit 24 hourly LASSO models with fixed penalties and return (days, 24) price forecasts."""
    scale = fit_price_scale(data.targets[train])
    train_inputs, predict_inputs = standardize(
        lear_inputs(data, train, scale, price_indices),
        lear_inputs(data, predict, scale, price_indices),
    )
    forecasts = np.empty((len(predict), HOURS_PER_DAY))
    for hour in range(HOURS_PER_DAY):
        train_target = to_asinh(data.targets[train, hour], scale)
        target_mean = train_target.mean()
        # Following the LARS path down to the penalty gives the exact LASSO solution, about 20x faster
        # than coordinate descent, which needs thousands of iterations on these correlated price lags.
        _, _, path_coefficients = lars_path(
            train_inputs,
            train_target - target_mean,
            method="lasso",
            alpha_min=penalties[hour],
        )
        predicted = predict_inputs @ path_coefficients[:, -1] + target_mean
        forecasts[:, hour] = from_asinh(predicted, scale)
    return forecasts


def lear_forecasts(data, window_days, first_index, end_index):
    """Retune penalties yearly and refit weekly for one window; returns forecasts and the penalty log."""
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    price_indices = price_feature_indices(data)
    retune_points = calendar_points(data, first_index, end_index, JANUARY)
    refit_points = sorted(set(every_n_days(first_index, end_index, LEAR_REFIT_DAYS)) | set(retune_points))

    forecasts = np.full(data.targets.shape, np.nan)
    penalty_log = []
    penalties = None
    for start, end in schedule_blocks(refit_points, end_index):
        if start in retune_points:
            print(f"lear_{window_days}: starting {data.dates[start].year}", flush=True)
            penalties = tune_lear_penalties(data, start, window_days, price_indices)
            penalty_log.append((data.dates[start], window_days, penalties))
        train = training_days(data, start, window_days)
        predict = forecast_days(data, start, end)
        if len(train) < MIN_TRAINING_DAYS or len(predict) == 0:
            continue
        forecasts[predict] = fit_and_predict_lear(data, train, predict, penalties, price_indices)
    return forecasts, penalty_log


def all_lear_forecasts(data, first_index, end_index):
    """Run every LEAR window in parallel; returns {expert name: forecasts} and the penalty log."""
    results = Parallel(n_jobs=NUM_PARALLEL_JOBS)(
        delayed(lear_forecasts)(data, window_days, first_index, end_index) for window_days in LEAR_WINDOWS_DAYS
    )
    forecasts = {}
    penalty_log = []
    for window_days, (window_forecasts, window_log) in zip(LEAR_WINDOWS_DAYS, results):
        forecasts[f"lear_{window_days}"] = window_forecasts
        penalty_log.extend(window_log)
    return forecasts, penalty_log


# ---------- LightGBM ----------


def lgbm_rows(data, days):
    """Stack one row per (day, hour): the day's 247 inputs, hour of day, and that hour's own inputs."""
    day_inputs = np.repeat(data.features[days], HOURS_PER_DAY, axis=0)
    hours = np.tile(np.arange(HOURS_PER_DAY), len(days))
    own_hour_blocks = []
    for prefix, lag_days in LGBM_OWN_HOUR_INPUTS:
        block = data.features[days][:, column_indices(data, lag_columns(prefix, lag_days))]
        own_hour_blocks.append(block.reshape(-1))
    return np.column_stack([day_inputs, hours, *own_hour_blocks])


def lgbm_forecasts(data, first_index, end_index):
    """Refit one pooled LightGBM model each quarter on the previous 728 days."""
    forecasts = np.full(data.targets.shape, np.nan)
    refit_points = calendar_points(data, first_index, end_index, QUARTER_START_MONTHS)
    for start, end in schedule_blocks(refit_points, end_index):
        train = training_days(data, start, LGBM_WINDOW_DAYS)
        predict = forecast_days(data, start, end)
        if len(train) < MIN_TRAINING_DAYS or len(predict) == 0:
            continue
        scale = fit_price_scale(data.targets[train])
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(lgbm_rows(data, train), to_asinh(data.targets[train].reshape(-1), scale))
        predicted = from_asinh(model.predict(lgbm_rows(data, predict)), scale)
        forecasts[predict] = predicted.reshape(len(predict), HOURS_PER_DAY)
    return forecasts


# ---------- output ----------


def to_long(data, forecasts_by_expert):
    """Return one row per (date, hour, expert) with a forecast."""
    frames = []
    for expert, forecasts in forecasts_by_expert.items():
        has_forecast = ~np.isnan(forecasts).any(axis=1)
        days = np.where(has_forecast)[0]
        frame = pl.DataFrame({
            "date": np.repeat(np.array(data.dates, dtype="datetime64[D]")[days], HOURS_PER_DAY),
            "hour": np.tile(np.arange(HOURS_PER_DAY, dtype=np.int8), len(days)),
            "forecast": forecasts[days].reshape(-1),
        })
        frames.append(frame.with_columns(pl.lit(expert).alias("expert")).select("date", "hour", "expert", "forecast"))
    return pl.concat(frames)


def penalties_to_frame(penalty_log):
    """Return one row per (retune date, window, hour) with the chosen LEAR penalty."""
    rows = []
    for retune_date, window_days, penalties in penalty_log:
        for hour in range(HOURS_PER_DAY):
            rows.append({"retune_date": retune_date, "window_days": window_days, "hour": hour, "penalty": penalties[hour]})
    return pl.DataFrame(rows)


def development_mae_table(data, forecasts_by_expert):
    """Return MAE per expert and year over development days (before the holdout) only."""
    years = np.array([day.year for day in data.dates])
    is_development = np.array([day < HOLDOUT_START for day in data.dates])
    rows = []
    for expert, forecasts in forecasts_by_expert.items():
        errors = np.abs(forecasts - data.targets).mean(axis=1)
        for year in np.unique(years[is_development & ~np.isnan(errors)]):
            in_year = is_development & (years == year) & ~np.isnan(errors)
            rows.append({"expert": expert, "year": int(year), "mae": float(errors[in_year].mean())})
    table = pl.DataFrame(rows).pivot(on="year", index="expert", values="mae")
    return table.with_columns(pl.selectors.numeric().round(2))


def refuse_overwrite():
    """Stop if cached forecasts exist, unless --overwrite was passed: experts are computed once."""
    if FORECASTS_PATH.exists() and "--overwrite" not in sys.argv:
        raise SystemExit(f"{FORECASTS_PATH} exists. Expert forecasts are computed once; pass --overwrite to redo.")


def timed(label, function, *arguments):
    """Run `function`, print how long it took, and return its result."""
    start = time.perf_counter()
    result = function(*arguments)
    print(f"{label}: {time.perf_counter() - start:.0f} s", flush=True)
    return result


def main():
    refuse_overwrite()
    data = load_day_data()
    first_index, end_index = forecast_index_range(data)

    forecasts = {}
    forecasts["naive"] = timed("naive", naive_forecasts, data, first_index, end_index)
    forecasts["arx"] = timed("arx", arx_forecasts, data, first_index, end_index)
    lear, penalty_log = timed("lear (5 windows)", all_lear_forecasts, data, first_index, end_index)
    forecasts.update(lear)
    forecasts["lgbm"] = timed("lgbm", lgbm_forecasts, data, first_index, end_index)

    EXPERTS_DIR.mkdir(parents=True, exist_ok=True)
    to_long(data, forecasts).write_parquet(FORECASTS_PATH)
    penalties_to_frame(penalty_log).write_parquet(LEAR_PENALTIES_PATH)
    with pl.Config(tbl_cols=-1):
        print(development_mae_table(data, forecasts))


if __name__ == "__main__":
    main()
