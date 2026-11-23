"""Build one row of forecasting inputs per delivery day from the clean hourly table.
Every input for day d is known before the day-ahead auction for d closes on d-1."""

from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
HOURLY_PATH = CLEAN_DIR / "de_hourly.parquet"
FLAGS_PATH = CLEAN_DIR / "day_flags.parquet"
FRENCH_HOURLY_PATH = CLEAN_DIR / "fr_hourly.parquet"

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
PRICE_LAGS_DAYS = [1, 2, 3, 7]
# Lag 0 is the delivery day itself: its load and renewables forecasts are published on d-1.
FUNDAMENTAL_LAGS_DAYS = [0, 1, 7]
FUNDAMENTAL_COLUMNS = {"load": "load_forecast_winsorized", "renewables": "renewables_total"}
# French inputs as (prefix, clean column, lag in days). France clears in the same coupled auction as
# Germany, so its prices for day d are unknown at forecast time: only yesterday's prices are used.
FRENCH_BLOCKS = [
    ("fr_price", "price", 1),
    ("fr_load", "load_forecast_winsorized", 0),
    ("fr_renewables", "renewables_total", 0),
]
LAGGED_PRICE_PREFIXES = ("price_lag", "fr_price_lag")


class DayData(NamedTuple):
    """Arrays aligned by delivery day, one row per date."""

    dates: list
    weekdays: np.ndarray
    targets: np.ndarray
    features: np.ndarray
    feature_names: list
    is_outlier: np.ndarray
    has_features: np.ndarray
    has_target: np.ndarray


def hour_suffix(hour):
    """Return the column suffix for `hour`, e.g. h07."""
    return f"h{hour:02d}"


def wide_by_hour(hourly, column, prefix):
    """Return one row per date with 24 columns `<prefix>_hHH` holding `column`."""
    wide = hourly.pivot(on="hour", index="date", values=column).sort("date")
    renames = {}
    for hour in range(HOURS_PER_DAY):
        renames[str(hour)] = f"{prefix}_{hour_suffix(hour)}"
    return wide.rename(renames).select("date", *renames.values())


def lagged_block(wide, prefix, lag_days):
    """Shift a wide block down by `lag_days` rows and name its columns `<prefix>_lag<k>_hHH`."""
    shifted = []
    for hour in range(HOURS_PER_DAY):
        source = f"{prefix}_{hour_suffix(hour)}"
        target = f"{prefix}_lag{lag_days}_{hour_suffix(hour)}"
        shifted.append(pl.col(source).shift(lag_days).alias(target))
    return wide.select(shifted)


def weekday_dummies(dates):
    """Return seven 0/1 columns, weekday_1 (Monday) to weekday_7 (Sunday)."""
    dummies = []
    for weekday in range(1, DAYS_PER_WEEK + 1):
        is_weekday = (pl.col("date").dt.weekday() == weekday).cast(pl.Float64)
        dummies.append(is_weekday.alias(f"weekday_{weekday}"))
    return dates.select(dummies)


def check_contiguous(dates):
    """Raise if the dates skip a day, since lags are taken by shifting rows."""
    steps = dates.diff().drop_nulls().unique()
    if steps.len() != 1 or steps.item() != timedelta(days=1):
        raise ValueError("clean table dates are not contiguous; row shifts would not be day lags")


def build_day_table(hourly):
    """Return one row per date: 24 target prices (price_hHH) followed by every input column."""
    prices = wide_by_hour(hourly, "price", "price")
    check_contiguous(prices["date"])
    blocks = [prices]
    for lag_days in PRICE_LAGS_DAYS:
        blocks.append(lagged_block(prices, "price", lag_days))
    for prefix, column in FUNDAMENTAL_COLUMNS.items():
        fundamentals = wide_by_hour(hourly, column, prefix)
        for lag_days in FUNDAMENTAL_LAGS_DAYS:
            blocks.append(lagged_block(fundamentals, prefix, lag_days))
    blocks.append(weekday_dummies(prices.select("date")))
    return pl.concat(blocks, how="horizontal")


def french_day_table():
    """Return one row per date with the 72 French inputs: prices on d-1, load and renewables forecasts for d."""
    hourly = pl.read_parquet(FRENCH_HOURLY_PATH)
    blocks = []
    dates = None
    for prefix, column, lag_days in FRENCH_BLOCKS:
        wide = wide_by_hour(hourly, column, prefix)
        check_contiguous(wide["date"])
        dates = wide.select("date")
        blocks.append(lagged_block(wide, prefix, lag_days))
    return pl.concat([dates, *blocks], how="horizontal")


def target_columns():
    """Return the 24 target column names."""
    names = []
    for hour in range(HOURS_PER_DAY):
        names.append(f"price_{hour_suffix(hour)}")
    return names


def load_day_data(include_france=False):
    """Return all delivery days as aligned arrays, with outlier-day flags attached (and the French inputs if asked)."""
    hourly = pl.read_parquet(HOURLY_PATH)
    flags = pl.read_parquet(FLAGS_PATH).select("date", "is_outlier")
    table = build_day_table(hourly).join(flags, on="date", how="left")
    if include_france:
        # Left join on German dates: days without French data get nulls, so has_features is False there.
        table = table.join(french_day_table(), on="date", how="left")
    table = table.sort("date").with_columns(pl.col("is_outlier").fill_null(False))

    targets = table.select(target_columns()).to_numpy().astype(float)
    feature_names = table.drop("date", "is_outlier", *target_columns()).columns
    features = table.select(feature_names).to_numpy().astype(float)
    return DayData(
        dates=table["date"].to_list(),
        weekdays=table["date"].dt.weekday().to_numpy(),
        targets=targets,
        features=features,
        feature_names=feature_names,
        is_outlier=table["is_outlier"].to_numpy(),
        has_features=~np.isnan(features).any(axis=1),
        has_target=~np.isnan(targets).any(axis=1),
    )


def column_indices(data, names):
    """Return the positions of feature `names` in `data.features`."""
    indices = []
    for name in names:
        indices.append(data.feature_names.index(name))
    return np.array(indices)


def lag_columns(prefix, lag_days):
    """Return the 24 column names of one lagged block, e.g. price_lag1_h00 .. price_lag1_h23."""
    names = []
    for hour in range(HOURS_PER_DAY):
        names.append(f"{prefix}_lag{lag_days}_{hour_suffix(hour)}")
    return names


def price_feature_indices(data):
    """Return the positions of every lagged-price input, German or French (these get the asinh transform)."""
    indices = []
    for position, name in enumerate(data.feature_names):
        if name.startswith(LAGGED_PRICE_PREFIXES):
            indices.append(position)
    return np.array(indices)
