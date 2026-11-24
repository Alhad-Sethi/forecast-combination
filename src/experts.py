"""Generate walk-forward forecasts from every expert and cache them in data/experts/ (`--add <names>` appends new experts only).
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
from sklearn.ensemble import RandomForestRegressor
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
POOLED_OWN_HOUR_INPUTS = [("price", 1), ("price", 2), ("price", 7), ("load", 0), ("renewables", 1)]
FRENCH_OWN_HOUR_INPUTS = [("fr_price", 1), ("fr_load", 0)]
FRENCH_LEAR_SHORT_WINDOW_DAYS = 112
FRENCH_LEAR_LONG_WINDOW_DAYS = 728
QUARTER_START_MONTHS = [1, 4, 7, 10]
JANUARY = [1]

RF_WINDOW_DAYS = 728
RF_PARAMS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "n_jobs": NUM_PARALLEL_JOBS,
    "random_state": 0,
}

KNN_WINDOW_DAYS = 728
KNN_NEIGHBOUR_GRID = [5, 10, 20, 40]
# A day is described by yesterday's prices, its own load forecast and yesterday's renewables forecast
# (the day's own renewables forecast is published after the auction).
KNN_DESCRIPTOR_BLOCKS = [("price", 1), ("load", 0), ("renewables", 1)]
# Neighbours come from the same day type (Monday, Tuesday-Friday, Saturday, Sunday), so weekly shape matches.
DAY_TYPE_BY_WEEKDAY = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3}
KNN_CV_DAYS = LEAR_CV_FOLDS * LEAR_CV_FOLD_DAYS


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


def yesterday_prices(data, days):
    """Return the 24 prices of the day before each of `days`: (days, 24)."""
    return data.features[days][:, column_indices(data, lag_columns("price", 1))]


def price_changes(data, days, scale):
    """Return the asinh-scale change from yesterday's same-hour price to each day's price: (days, 24)."""
    return to_asinh(data.targets[days], scale) - to_asinh(yesterday_prices(data, days), scale)


def add_change_to_yesterday(data, days, predicted_changes, scale):
    """Turn predicted asinh-scale changes into price forecasts by adding them to yesterday's prices."""
    return from_asinh(to_asinh(yesterday_prices(data, days), scale) + predicted_changes, scale)


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
    """Return all 223 inputs with the lagged prices asinh-transformed."""
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


def own_hour_inputs(data):
    """Return the (prefix, lag) blocks copied per hour: the German ones, plus the French ones if `data` has them."""
    if "fr_price_lag1_h00" in data.feature_names:
        return POOLED_OWN_HOUR_INPUTS + FRENCH_OWN_HOUR_INPUTS
    return POOLED_OWN_HOUR_INPUTS


def pooled_rows(data, days):
    """Stack one row per (day, hour): all the day's inputs, hour of day, and that hour's own inputs."""
    day_inputs = np.repeat(data.features[days], HOURS_PER_DAY, axis=0)
    hours = np.tile(np.arange(HOURS_PER_DAY), len(days))
    own_hour_blocks = []
    for prefix, lag_days in own_hour_inputs(data):
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
        model.fit(pooled_rows(data, train), to_asinh(data.targets[train].reshape(-1), scale))
        predicted = from_asinh(model.predict(pooled_rows(data, predict)), scale)
        forecasts[predict] = predicted.reshape(len(predict), HOURS_PER_DAY)
    return forecasts


def lgbm_change_forecasts(data, first_index, end_index):
    """Same model and schedule as lgbm, but predicting the change from yesterday's same-hour price."""
    forecasts = np.full(data.targets.shape, np.nan)
    refit_points = calendar_points(data, first_index, end_index, QUARTER_START_MONTHS)
    for start, end in schedule_blocks(refit_points, end_index):
        train = training_days(data, start, LGBM_WINDOW_DAYS)
        predict = forecast_days(data, start, end)
        if len(train) < MIN_TRAINING_DAYS or len(predict) == 0:
            continue
        scale = fit_price_scale(data.targets[train])
        # lgbm breaks when prices rise fast because trees cannot extrapolate the level; changes stay in range.
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(pooled_rows(data, train), price_changes(data, train, scale).reshape(-1))
        predicted_changes = model.predict(pooled_rows(data, predict)).reshape(len(predict), HOURS_PER_DAY)
        forecasts[predict] = add_change_to_yesterday(data, predict, predicted_changes, scale)
    return forecasts


# ---------- random forest ----------


def rf_forecasts(data, first_index, end_index):
    """Refit one pooled random forest each quarter on the previous 728 days, predicting the change from yesterday."""
    forecasts = np.full(data.targets.shape, np.nan)
    refit_points = calendar_points(data, first_index, end_index, QUARTER_START_MONTHS)
    for start, end in schedule_blocks(refit_points, end_index):
        train = training_days(data, start, RF_WINDOW_DAYS)
        predict = forecast_days(data, start, end)
        if len(train) < MIN_TRAINING_DAYS or len(predict) == 0:
            continue
        scale = fit_price_scale(data.targets[train])
        # Trees cannot extrapolate; predicting the change from yesterday lets the level follow the market.
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(pooled_rows(data, train), price_changes(data, train, scale).reshape(-1))
        predicted_changes = model.predict(pooled_rows(data, predict)).reshape(len(predict), HOURS_PER_DAY)
        forecasts[predict] = add_change_to_yesterday(data, predict, predicted_changes, scale)
    return forecasts


# ---------- similar days (k nearest neighbours) ----------


def day_types(data, days):
    """Return the day type (Monday, Tuesday-Friday, Saturday, Sunday) of each of `days`."""
    types = []
    for weekday in data.weekdays[days]:
        types.append(DAY_TYPE_BY_WEEKDAY[weekday])
    return np.array(types)


def knn_descriptors(data, days, scale):
    """Return each day's descriptor: yesterday's asinh prices, the day's load and yesterday's renewables forecasts."""
    blocks = []
    for prefix, lag_days in KNN_DESCRIPTOR_BLOCKS:
        block = data.features[days][:, column_indices(data, lag_columns(prefix, lag_days))]
        if prefix == "price":
            block = to_asinh(block, scale)
        blocks.append(block)
    return np.column_stack(blocks)


def nearest_day_changes(data, day, window_days):
    """Return the price scale and the price changes of same-type training days, nearest to `day` first."""
    train = training_days(data, day, window_days)
    same_type = train[day_types(data, train) == DAY_TYPE_BY_WEEKDAY[data.weekdays[day]]]
    scale = fit_price_scale(data.targets[train])
    train_descriptors, day_descriptor = standardize(
        knn_descriptors(data, same_type, scale),
        knn_descriptors(data, np.array([day]), scale),
    )
    distances = np.linalg.norm(train_descriptors - day_descriptor, axis=1)
    nearest_first = same_type[np.argsort(distances)]
    return scale, price_changes(data, nearest_first, scale)


def knn_day_forecast(data, day, neighbour_changes, num_neighbours, scale):
    """Return one day's 24 forecasts: yesterday's prices plus the mean change of the nearest days."""
    mean_change = neighbour_changes[:num_neighbours].mean(axis=0)
    return add_change_to_yesterday(data, np.array([day]), mean_change[np.newaxis, :], scale)[0]


def tune_knn_neighbours(data, retune_index):
    """Pick the number of neighbours with the lowest MAE over the 84 usable days before `retune_index`."""
    validation = training_days(data, retune_index, KNN_CV_DAYS)
    errors = np.zeros(len(KNN_NEIGHBOUR_GRID))
    for day in validation:
        scale, neighbour_changes = nearest_day_changes(data, day, KNN_WINDOW_DAYS)
        for position, num_neighbours in enumerate(KNN_NEIGHBOUR_GRID):
            forecast = knn_day_forecast(data, day, neighbour_changes, num_neighbours, scale)
            errors[position] += np.abs(forecast - data.targets[day]).mean()
    return KNN_NEIGHBOUR_GRID[int(np.argmin(errors))]


def knn_forecasts(data, first_index, end_index):
    """Forecast each day from its most similar past days; the number of neighbours is retuned yearly."""
    forecasts = np.full(data.targets.shape, np.nan)
    retune_points = calendar_points(data, first_index, end_index, JANUARY)
    num_neighbours = None
    for day in range(first_index, end_index):
        if day in retune_points:
            num_neighbours = tune_knn_neighbours(data, day)
            print(f"knn: {data.dates[day].year} uses {num_neighbours} neighbours", flush=True)
        if not data.has_features[day]:
            continue
        scale, neighbour_changes = nearest_day_changes(data, day, KNN_WINDOW_DAYS)
        forecasts[day] = knn_day_forecast(data, day, neighbour_changes, num_neighbours, scale)
    return forecasts



# ---------- experts with French inputs ----------


def lear_fr_short_forecasts(data, first_index, end_index):
    """LEAR with a 112-day window; with French data it also sees the 48 French inputs."""
    forecasts, _ = lear_forecasts(data, FRENCH_LEAR_SHORT_WINDOW_DAYS, first_index, end_index)
    return forecasts


def lear_fr_long_forecasts(data, first_index, end_index):
    """LEAR with a 728-day window; with French data it also sees the 48 French inputs."""
    forecasts, _ = lear_forecasts(data, FRENCH_LEAR_LONG_WINDOW_DAYS, first_index, end_index)
    return forecasts


def fill_from_twin(name, forecasts, twin_forecasts, data, first_index, end_index):
    """Use the German-only twin's forecast on days the French-input expert could not forecast (no French data)."""
    filled = forecasts.copy()
    days = np.arange(first_index, end_index)
    needs_twin = np.isnan(forecasts[days]).any(axis=1)
    filled[days[needs_twin]] = twin_forecasts[days[needs_twin]]
    is_development = np.array([data.dates[day] < HOLDOUT_START for day in days[needs_twin]], dtype=bool)
    print(f"{name}: {needs_twin.sum()} days from its twin ({is_development.sum()} before the holdout)", flush=True)
    return filled


# Experts that `--add` can compute on their own: the function, whether it uses the French inputs,
# and for French-input experts the German-only twin that fills days without French data.
ADDABLE_EXPERTS = {
    "rf": {"function": rf_forecasts, "french": False},
    "knn": {"function": knn_forecasts, "french": False},
    "lgbm_change": {"function": lgbm_change_forecasts, "french": False},
    "lear_fr_112": {"function": lear_fr_short_forecasts, "french": True, "twin": "lear_112"},
    "lear_fr_728": {"function": lear_fr_long_forecasts, "french": True, "twin": "lear_728"},
    "lgbm_fr": {"function": lgbm_forecasts, "french": True, "twin": "lgbm"},
    "lgbm_change_fr": {"function": lgbm_change_forecasts, "french": True, "twin": "lgbm_change"},
}


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


def requested_additions():
    """Return the expert names listed after --add on the command line (empty without --add)."""
    if "--add" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--add") + 1:]


def check_additions(names, existing_experts):
    """Stop unless every name is an addable expert that is not cached yet and whose twin (if any) is cached."""
    for name in names:
        if name not in ADDABLE_EXPERTS:
            raise SystemExit(f"--add {name}: unknown; addable experts are {list(ADDABLE_EXPERTS)}")
        if name in existing_experts:
            raise SystemExit(f"--add {name}: already cached; use a full run with --overwrite to recompute it")
        twin = ADDABLE_EXPERTS[name].get("twin")
        if twin is not None and twin not in existing_experts:
            raise SystemExit(f"--add {name}: its twin {twin} must be cached first")


def cached_forecast_array(cached, expert, dates):
    """Return one cached expert's forecasts as a (days, 24) array aligned with `dates` (NaN where absent)."""
    rows = cached.filter(pl.col("expert") == expert)
    positions = pl.DataFrame({"date": dates, "day_index": np.arange(len(dates))})
    rows = rows.join(positions, on="date", how="inner")
    array = np.full((len(dates), HOURS_PER_DAY), np.nan)
    array[rows["day_index"].to_numpy(), rows["hour"].to_numpy()] = rows["forecast"].to_numpy()
    return array


def compute_addable_expert(name, german_data, french_data, twin_forecasts, first_index, end_index):
    """Run one addable expert on its feature set; French-input experts fall back to their twin where needed."""
    spec = ADDABLE_EXPERTS[name]
    if not spec["french"]:
        return timed(name, spec["function"], german_data, first_index, end_index)
    forecasts = timed(name, spec["function"], french_data, first_index, end_index)
    return fill_from_twin(name, forecasts, twin_forecasts, german_data, first_index, end_index)


def needs_french_data(names):
    """Return True if any of `names` uses the French inputs."""
    for name in names:
        if ADDABLE_EXPERTS[name]["french"]:
            return True
    return False


def write_forecasts(frame):
    """Write the long forecast table via a temporary file, so a failed write never leaves a broken cache."""
    temporary_path = FORECASTS_PATH.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary_path)
    temporary_path.replace(FORECASTS_PATH)


def add_experts(data, names):
    """Compute only `names` and append them to the cached forecasts; existing experts are never touched."""
    existing = pl.read_parquet(FORECASTS_PATH)
    check_additions(names, set(existing["expert"].unique()))
    first_index, end_index = forecast_index_range(data)
    french_data = None
    if needs_french_data(names):
        french_data = load_day_data(include_france=True)
    forecasts = {}
    for name in names:
        twin_forecasts = None
        twin = ADDABLE_EXPERTS[name].get("twin")
        if twin is not None:
            twin_forecasts = cached_forecast_array(existing, twin, data.dates)
        forecasts[name] = compute_addable_expert(name, data, french_data, twin_forecasts, first_index, end_index)
    write_forecasts(pl.concat([existing, to_long(data, forecasts)]))
    return forecasts


def compute_all_experts(data):
    """Compute every expert from scratch and write the forecasts and LEAR penalties."""
    refuse_overwrite()
    first_index, end_index = forecast_index_range(data)
    forecasts = {}
    forecasts["naive"] = timed("naive", naive_forecasts, data, first_index, end_index)
    forecasts["arx"] = timed("arx", arx_forecasts, data, first_index, end_index)
    lear, penalty_log = timed("lear (5 windows)", all_lear_forecasts, data, first_index, end_index)
    forecasts.update(lear)
    forecasts["lgbm"] = timed("lgbm", lgbm_forecasts, data, first_index, end_index)
    french_data = load_day_data(include_france=True)
    # Dict order puts every twin before the French-input expert that needs it.
    for name, spec in ADDABLE_EXPERTS.items():
        twin_forecasts = forecasts.get(spec.get("twin"))
        forecasts[name] = compute_addable_expert(name, data, french_data, twin_forecasts, first_index, end_index)

    EXPERTS_DIR.mkdir(parents=True, exist_ok=True)
    write_forecasts(to_long(data, forecasts))
    penalties_to_frame(penalty_log).write_parquet(LEAR_PENALTIES_PATH)
    return forecasts


def main():
    data = load_day_data()
    additions = requested_additions()
    if additions:
        forecasts = add_experts(data, additions)
    else:
        forecasts = compute_all_experts(data)
    with pl.Config(tbl_cols=-1):
        print(development_mae_table(data, forecasts))


if __name__ == "__main__":
    main()
