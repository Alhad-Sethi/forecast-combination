"""Flag outlier days from the day-ahead load, wind and solar forecasts: union of two detectors.
Flagged days are dropped as training targets downstream but stay in evaluation."""

import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
HOURLY_PATH = CLEAN_DIR / "de_hourly.parquet"
FLAGS_PATH = CLEAN_DIR / "day_flags.parquet"
HOLDOUT_START_YEAR = 2024
HOURS_PER_DAY = 24

# Raw (not winsorized) load, because winsorizing would hide the extremes we want to detect.
DETECTOR_SERIES = ["load_forecast", "wind", "solar"]

Z_BASELINE_DAYS = 28
# Scales the median absolute deviation so it matches the standard deviation for normal data.
MAD_TO_STD = 1.4826
MAD_FLOOR_FRACTION = 0.02
MAX_ABS_Z = 10.0
SCORE_QUANTILE = 0.99
SCORE_WINDOW_DAYS = 364
# The z-score threshold needs half a year of past scores before it flags anything.
SCORE_MIN_DAYS = 182

# Half the z-score detector's 1%, because out of sample the forest also flags record
# wind/solar days from capacity growth (kept on purpose: it adds a different kind of flag).
CONTAMINATION = 0.005
FOREST_TRAINING_DAYS = 728
NUM_TREES = 200
RANDOM_SEED = 0
# Years whose training window has fewer complete days than this are left unscored.
MIN_FOREST_TRAINING_DAYS = 364
# All of 2015-2016 is known before the first forecast in 2017, so those years are scored in-sample.
FIRST_OUT_OF_SAMPLE_YEAR = 2017


def load_detector_inputs():
    """Return the clean hourly table with total wind and residual load added."""
    hourly = pl.read_parquet(HOURLY_PATH)
    wind = pl.col("wind_onshore") + pl.col("wind_offshore")
    hourly = hourly.with_columns(wind.alias("wind"))
    residual_load = pl.col("load_forecast") - pl.col("wind") - pl.col("solar")
    return hourly.with_columns(residual_load.alias("residual_load")).sort("date", "hour")


def day_by_hour_matrix(hourly, column):
    """Return `column` as a (days, 24) array, one row per date in sorted order."""
    wide = hourly.pivot(on="hour", index="date", values=column).sort("date")
    hour_columns = []
    for hour in range(HOURS_PER_DAY):
        hour_columns.append(str(hour))
    return wide.select(hour_columns).to_numpy().astype(float)


def robust_z_scores(matrix):
    """Score each day-hour against the same hour over the previous 28 days: (x - median) / scaled MAD."""
    # Window i covers days i .. i+27, so it is the baseline for day i+28; the last window has no day to score.
    windows = sliding_window_view(matrix, Z_BASELINE_DAYS, axis=0)[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(windows, axis=2)
        deviations = np.abs(windows - medians[:, :, np.newaxis])
        scaled_mads = np.nanmedian(deviations, axis=2) * MAD_TO_STD
        # Solar at dawn and dusk is near zero for weeks, so its MAD is tiny and z explodes.
        # Floor the scale at a small fraction of the series' average level over the same window.
        window_levels = np.nanmean(np.abs(windows), axis=(1, 2))
        mad_floors = MAD_FLOOR_FRACTION * window_levels
        scaled_mads = np.maximum(scaled_mads, mad_floors[:, np.newaxis])
        z_scores = (matrix[Z_BASELINE_DAYS:] - medians) / scaled_mads

    # A zero scale (a series that was zero for the whole window) gives no information.
    z_scores[scaled_mads == 0] = 0.0
    # Cap each hour so no single hour can dominate a day's average.
    z_scores = np.clip(z_scores, -MAX_ABS_Z, MAX_ABS_Z)
    padding = np.full((Z_BASELINE_DAYS, HOURS_PER_DAY), np.nan)
    return np.vstack([padding, z_scores])


def average_z_score(hourly):
    """Return one score per day: mean |z| over all hours of load, wind and solar."""
    absolute_z_scores = []
    for column in DETECTOR_SERIES:
        absolute_z_scores.append(np.abs(robust_z_scores(day_by_hour_matrix(hourly, column))))
    stacked = np.stack(absolute_z_scores, axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stacked, axis=(1, 2))


def z_score_flags(hourly):
    """Flag days whose average z-score exceeds the 99th percentile of the previous 364 days' scores."""
    dates = hourly["date"].unique().sort()
    scores = pl.DataFrame({"date": dates, "z_score": average_z_score(hourly)})
    scores = scores.with_columns(pl.col("z_score").fill_nan(None))
    threshold = pl.col("z_score").rolling_quantile(
        quantile=SCORE_QUANTILE,
        window_size=SCORE_WINDOW_DAYS,
        min_samples=SCORE_MIN_DAYS,
    )
    scores = scores.with_columns(threshold.shift(1).alias("z_threshold"))
    is_flagged = (pl.col("z_score") > pl.col("z_threshold")).fill_null(False)
    return scores.with_columns(is_flagged.alias("z_flag"))


def daily_summaries(hourly):
    """Return the daily mean, min and max of load, wind, solar and residual load."""
    aggregations = []
    for column in DETECTOR_SERIES + ["residual_load"]:
        aggregations.append(pl.col(column).mean().alias(f"{column}_mean"))
        aggregations.append(pl.col(column).min().alias(f"{column}_min"))
        aggregations.append(pl.col(column).max().alias(f"{column}_max"))
    return hourly.group_by("date").agg(aggregations).sort("date")


def fit_forest(training_days):
    """Fit an isolation forest on the summary features of complete training days."""
    features = training_days.drop("date").drop_nulls().to_numpy()
    forest = IsolationForest(
        n_estimators=NUM_TREES,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
    )
    return forest.fit(features)


def unscored(days):
    """Return null forest scores for `days`, used when there is too little data to fit or score."""
    return days.select("date").with_columns(
        pl.lit(None, dtype=pl.Float64).alias("forest_score"),
        pl.lit(None, dtype=pl.Boolean).alias("forest_flag"),
    )


def score_with_forest(forest, days):
    """Score complete days with `forest`; days with missing inputs get null scores and no flag."""
    complete_days = days.drop_nulls()
    if complete_days.height == 0:
        return unscored(days)
    features = complete_days.drop("date").to_numpy()
    scored = complete_days.select("date").with_columns(
        pl.Series("forest_score", forest.decision_function(features)),
        pl.Series("forest_flag", forest.predict(features) == -1),
    )
    return days.select("date").join(scored, on="date", how="left")


def forest_flags(summaries):
    """Fit yearly on the previous 728 days and score the next year; 2015-2016 are scored in-sample."""
    first_out_of_sample_day = date(FIRST_OUT_OF_SAMPLE_YEAR, 1, 1)
    in_sample_days = summaries.filter(pl.col("date") < first_out_of_sample_day)
    scored_years = [score_with_forest(fit_forest(in_sample_days), in_sample_days)]

    last_year = summaries["date"].max().year
    for year in range(FIRST_OUT_OF_SAMPLE_YEAR, last_year + 1):
        year_start = date(year, 1, 1)
        training_start = year_start - timedelta(days=FOREST_TRAINING_DAYS)
        training_days = summaries.filter(pl.col("date").is_between(training_start, year_start, closed="left"))
        scoring_days = summaries.filter(pl.col("date").dt.year() == year)
        if training_days.drop_nulls().height < MIN_FOREST_TRAINING_DAYS:
            scored_years.append(unscored(scoring_days))
            continue
        scored_years.append(score_with_forest(fit_forest(training_days), scoring_days))

    flags = pl.concat(scored_years)
    return flags.with_columns(pl.col("forest_flag").fill_null(False))


def combine_flags(z_flags, forest_flags_frame):
    """Join both detectors; a day is an outlier if either one fires."""
    flags = z_flags.join(forest_flags_frame, on="date", how="left")
    is_outlier = pl.col("z_flag") | pl.col("forest_flag").fill_null(False)
    return flags.with_columns(is_outlier.alias("is_outlier"))


def print_development_summary(flags):
    """Print flagged-day counts per year for development years only."""
    development = flags.filter(pl.col("date").dt.year() < HOLDOUT_START_YEAR)
    counts = development.group_by(pl.col("date").dt.year().alias("year")).agg(
        pl.len().alias("days"),
        pl.col("z_flag").sum(),
        pl.col("forest_flag").sum(),
        (pl.col("z_flag") & pl.col("forest_flag")).sum().alias("both"),
        pl.col("is_outlier").sum(),
    )
    with pl.Config(tbl_rows=-1):
        print(counts.sort("year"))


def main():
    hourly = load_detector_inputs()
    z_flags = z_score_flags(hourly)
    forest_flags_frame = forest_flags(daily_summaries(hourly))
    flags = combine_flags(z_flags, forest_flags_frame)

    flags.write_parquet(FLAGS_PATH)
    print_development_summary(flags)


if __name__ == "__main__":
    main()
