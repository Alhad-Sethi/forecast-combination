"""Clean raw ENTSO-E downloads into one table with 24 local hours per day.
Applies impossible-value rules, gap filling, daylight-saving handling and causal load winsorizing."""

from pathlib import Path

import polars as pl

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "entsoe"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
LOCAL_TIMEZONE = "Europe/Berlin"
HOLDOUT_START_YEAR = 2024

# entsoe-py column names, mapped to clean names.
RAW_TIMESTAMP_COLUMN = "__index_level_0__"
RAW_COLUMNS = {
    "day_ahead_prices": {"price_eur_mwh": "price"},
    "load_forecast": {"Forecasted Load": "load_forecast"},
    "wind_solar_forecast": {
        "Wind Onshore": "wind_onshore",
        "Wind Offshore": "wind_offshore",
        "Solar": "solar",
    },
}
VALUE_COLUMNS = ["price", "load_forecast", "wind_onshore", "wind_offshore", "solar"]
FORECAST_COLUMNS = ["load_forecast", "wind_onshore", "wind_offshore", "solar"]
RENEWABLE_COLUMNS = ["wind_onshore", "wind_offshore", "solar"]

# Harmonised day-ahead auction limits; 4,000 is the post-2022 cap, used for the whole period.
PRICE_FLOOR = -500.0
PRICE_CAP = 4000.0
# Generous caps (MW) above German installed capacity at the end of 2023 (about 61, 8.5 and 82 GW).
CAPACITY_CAPS_MW = {"wind_onshore": 80_000.0, "wind_offshore": 15_000.0, "solar": 150_000.0}
FROZEN_RUN_HOURS = 24

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
MAX_INTERPOLATION_HOURS = 3
MAX_PREVIOUS_WEEK_FILL_HOURS = DAYS_PER_WEEK * HOURS_PER_DAY

WINSOR_LOWER_QUANTILE = 0.005
WINSOR_UPPER_QUANTILE = 0.995
WINSOR_WINDOW_DAYS = 364
# With less history than this, the load is left unclipped.
WINSOR_MIN_DAYS = 28

OBSERVED = "observed"
INTERPOLATED = "interpolated"
PREVIOUS_WEEK = "previous_week"
MISSING = "missing"


def read_series(series_name):
    """Return one raw series with UTC timestamps and clean column names, duplicates dropped."""
    yearly_frames = []
    for path in sorted((RAW_DIR / series_name).glob("*.parquet")):
        yearly_frames.append(pl.read_parquet(path))
    frame = pl.concat(yearly_frames, how="vertical_relaxed")

    renames = {RAW_TIMESTAMP_COLUMN: "timestamp"}
    renames.update(RAW_COLUMNS[series_name])
    frame = frame.rename(renames)
    frame = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    frame = frame.sort("timestamp")
    # entsoe-py truncates inclusively, so year boundaries come back twice.
    return frame.unique(subset="timestamp", keep="first", maintain_order=True)


def to_hourly(frame):
    """Average each clock hour, in UTC, so 15-minute series become hourly."""
    value_columns = []
    for column in frame.columns:
        if column != "timestamp":
            value_columns.append(column)
    return frame.group_by_dynamic("timestamp", every="1h").agg(pl.col(value_columns).mean())


def join_on_hourly_grid(hourly_frames):
    """Join all series onto one complete hourly UTC grid, so missing hours become explicit nulls."""
    starts = []
    ends = []
    for frame in hourly_frames:
        starts.append(frame["timestamp"].min())
        ends.append(frame["timestamp"].max())
    grid_timestamps = pl.datetime_range(min(starts), max(ends), interval="1h", eager=True)
    table = pl.DataFrame({"timestamp": grid_timestamps})

    for frame in hourly_frames:
        table = table.join(frame, on="timestamp", how="left")
    local_dates = pl.col("timestamp").dt.convert_time_zone(LOCAL_TIMEZONE).dt.date()
    return table.with_columns(local_dates.alias("local_date"))


def frozen_feed_condition(column):
    """Return a condition that is true inside runs of 24+ identical nonzero hourly values."""
    run_length = pl.len().over(pl.col(column).rle_id())
    is_frozen = (run_length >= FROZEN_RUN_HOURS) & (pl.col(column) != 0)
    return is_frozen & pl.col(column).is_not_null()


def add_same_price_as_day_before(table):
    """Add a helper column: is each hourly price equal to the price 24 hours earlier?"""
    # Computed over the whole series first; shifting inside a per-day window would never reach the day before.
    is_same = (pl.col("price") == pl.col("price").shift(HOURS_PER_DAY)).fill_null(False)
    return table.with_columns(is_same.alias("same_price_as_day_before"))


def repeated_price_day_condition():
    """Return a condition that is true on local days whose prices all equal the same hour a day earlier."""
    return pl.col("same_price_as_day_before").all().over("local_date")


def impossible_value_rules():
    """Return (column, rule name, condition) for every impossible-value rule."""
    rules = [
        ("price", "below_price_floor", pl.col("price") < PRICE_FLOOR),
        ("price", "above_price_cap", pl.col("price") > PRICE_CAP),
        ("price", "repeated_price_day", repeated_price_day_condition()),
    ]
    for column in RENEWABLE_COLUMNS:
        rules.append((column, "negative_generation", pl.col(column) < 0))
        rules.append((column, "above_capacity_cap", pl.col(column) > CAPACITY_CAPS_MW[column]))
    for column in FORECAST_COLUMNS:
        rules.append((column, "frozen_feed", frozen_feed_condition(column)))
    return rules


def apply_impossible_value_rules(table):
    """Set values that break a rule to null. Return the table and one row per flagged value."""
    table = add_same_price_as_day_before(table)
    flag_frames = []
    null_conditions = {}
    for column in VALUE_COLUMNS:
        null_conditions[column] = pl.lit(False)

    for column, rule_name, condition in impossible_value_rules():
        condition = condition.fill_null(False)
        flagged = table.filter(condition).select(
            "timestamp",
            pl.lit(column).alias("column"),
            pl.col(column).cast(pl.Float64).alias("value"),
            pl.lit(rule_name).alias("rule"),
        )
        flag_frames.append(flagged)
        null_conditions[column] = null_conditions[column] | condition

    replacements = []
    for column in VALUE_COLUMNS:
        replacement = pl.when(null_conditions[column]).then(None).otherwise(pl.col(column))
        replacements.append(replacement.alias(column))
    table = table.with_columns(replacements).drop("same_price_as_day_before")
    return table, pl.concat(flag_frames)


def null_run_length(column):
    """Return, for each null, the length of the run of nulls it belongs to (0 for non-null values)."""
    is_missing = pl.col(column).is_null()
    run_length = pl.len().over(is_missing.rle_id())
    return pl.when(is_missing).then(run_length).otherwise(0)


def interpolate_short_gaps(table):
    """Linearly interpolate gaps of up to 3 hours and record each value's fill status."""
    updates = []
    for column in VALUE_COLUMNS:
        is_short_gap = (null_run_length(column) > 0) & (null_run_length(column) <= MAX_INTERPOLATION_HOURS)
        filled = pl.when(is_short_gap).then(pl.col(column).interpolate()).otherwise(pl.col(column))
        status = pl.when(pl.col(column).is_not_null()).then(pl.lit(OBSERVED))
        status = status.when(is_short_gap & filled.is_not_null()).then(pl.lit(INTERPOLATED))
        status = status.otherwise(pl.lit(MISSING))
        updates.append(filled.alias(column))
        updates.append(status.alias(f"{column}_fill"))
    return table.with_columns(updates)


def to_local_days(table):
    """Re-index to (local date, hour 0-23): average the doubled October hour, interpolate the missing March hour."""
    local_time = pl.col("timestamp").dt.convert_time_zone(LOCAL_TIMEZONE)
    table = table.with_columns(local_time.dt.date().alias("date"), local_time.dt.hour().alias("hour"))

    fill_columns = []
    for column in VALUE_COLUMNS:
        fill_columns.append(f"{column}_fill")
    daily = table.group_by("date", "hour").agg(
        pl.col(VALUE_COLUMNS).mean(),
        pl.col(fill_columns).first(),
    )

    dates = pl.date_range(table["date"].min(), table["date"].max(), interval="1d", eager=True)
    hours = pl.Series("hour", range(HOURS_PER_DAY), dtype=daily["hour"].dtype)
    grid = pl.DataFrame({"date": dates}).join(pl.DataFrame({"hour": hours}), how="cross")
    daily = grid.join(daily, on=["date", "hour"], how="left").sort("date", "hour")

    # Only the spring daylight-saving hour is absent from the grid join. Interpolate it only
    # when both neighbours exist, so it never bridges a longer gap.
    updates = []
    for column in VALUE_COLUMNS:
        is_absent_hour = pl.col(f"{column}_fill").is_null()
        is_isolated = null_run_length(column) == 1
        interpolated = pl.col(column).interpolate()
        filled = pl.when(is_absent_hour & is_isolated).then(interpolated).otherwise(pl.col(column))
        status = pl.when(is_absent_hour & filled.is_not_null()).then(pl.lit(INTERPOLATED))
        status = status.when(is_absent_hour).then(pl.lit(MISSING))
        status = status.otherwise(pl.col(f"{column}_fill"))
        updates.append(filled.alias(column))
        updates.append(status.alias(f"{column}_fill"))
    return daily.with_columns(updates)


def fill_from_previous_week(daily):
    """Fill gaps of up to 7 days with the same local hour one week earlier."""
    updates = []
    for column in VALUE_COLUMNS:
        is_medium_gap = (null_run_length(column) > 0) & (null_run_length(column) <= MAX_PREVIOUS_WEEK_FILL_HOURS)
        previous_week = pl.col(column).shift(DAYS_PER_WEEK).over("hour")
        filled = pl.when(is_medium_gap).then(previous_week).otherwise(pl.col(column))
        status = pl.when(is_medium_gap & filled.is_not_null()).then(pl.lit(PREVIOUS_WEEK))
        status = status.otherwise(pl.col(f"{column}_fill"))
        updates.append(filled.alias(column))
        updates.append(status.alias(f"{column}_fill"))
    return daily.with_columns(updates)


def trailing_load_quantile(quantile):
    """Return the load quantile for the same hour over the previous 364 days (excluding today)."""
    rolling = pl.col("load_forecast").rolling_quantile(
        quantile=quantile,
        window_size=WINSOR_WINDOW_DAYS,
        min_samples=WINSOR_MIN_DAYS,
    )
    return rolling.shift(1).over("hour")


def winsorize_load(daily):
    """Add a winsorized load column, clipped causally to trailing per-hour quantiles."""
    lower = trailing_load_quantile(WINSOR_LOWER_QUANTILE)
    upper = trailing_load_quantile(WINSOR_UPPER_QUANTILE)
    clipped = pl.min_horizontal(pl.max_horizontal(pl.col("load_forecast"), lower), upper)
    has_history = lower.is_not_null() & upper.is_not_null()
    winsorized = pl.when(has_history).then(clipped).otherwise(pl.col("load_forecast"))
    return daily.with_columns(winsorized.alias("load_forecast_winsorized"))


def add_renewables_total(daily):
    """Add onshore + offshore + solar; null if any part is missing."""
    total = pl.col("wind_onshore") + pl.col("wind_offshore") + pl.col("solar")
    return daily.with_columns(total.alias("renewables_total"))


def order_columns(daily):
    """Return the clean table with values first and fill statuses last."""
    value_columns = [
        "price",
        "load_forecast",
        "load_forecast_winsorized",
        "wind_onshore",
        "wind_offshore",
        "solar",
        "renewables_total",
    ]
    fill_columns = []
    for column in VALUE_COLUMNS:
        fill_columns.append(f"{column}_fill")
    return daily.select("date", "hour", *value_columns, *fill_columns)


def fill_report(daily):
    """Count hours per year, column and fill status."""
    fill_columns = []
    for column in VALUE_COLUMNS:
        fill_columns.append(f"{column}_fill")
    statuses = daily.select(pl.col("date").dt.year().alias("year"), *fill_columns)
    long = statuses.unpivot(index="year", variable_name="column", value_name="status")
    counts = long.group_by("year", "column", "status").len()
    report = counts.pivot(on="status", index=["year", "column"], values="len").fill_null(0)
    return report.sort("year", "column")


def flag_counts(flags):
    """Count impossible-value flags per year, column and rule."""
    with_year = flags.with_columns(pl.col("timestamp").dt.year().alias("year"))
    return with_year.group_by("year", "column", "rule").len().sort("year", "column", "rule")


def development_rows(frame, year_column):
    """Keep rows before the holdout year, so holdout values are never shown."""
    return frame.filter(pl.col(year_column) < HOLDOUT_START_YEAR)


def save_outputs(daily, flags):
    """Write the clean table, fill report, flag counts and development-period flag details."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(CLEAN_DIR / "de_hourly.parquet")
    fill_report(daily).write_csv(CLEAN_DIR / "fill_report.csv")
    flag_counts(flags).write_csv(CLEAN_DIR / "flag_counts.csv")
    flags_with_year = flags.with_columns(pl.col("timestamp").dt.year().alias("year"))
    development_rows(flags_with_year, "year").drop("year").write_csv(CLEAN_DIR / "impossible_values_dev.csv")


def print_development_summary(daily, flags):
    """Print fill statuses and flag counts for development years only."""
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(development_rows(fill_report(daily), "year"))
        print(development_rows(flag_counts(flags), "year"))


def main():
    hourly_frames = []
    for series_name in RAW_COLUMNS:
        hourly_frames.append(to_hourly(read_series(series_name)))

    hourly = join_on_hourly_grid(hourly_frames)
    hourly, flags = apply_impossible_value_rules(hourly)
    hourly = interpolate_short_gaps(hourly)

    daily = to_local_days(hourly)
    daily = fill_from_previous_week(daily)
    daily = winsorize_load(daily)
    daily = add_renewables_total(daily)
    daily = order_columns(daily)

    save_outputs(daily, flags)
    print_development_summary(daily, flags)


if __name__ == "__main__":
    main()
