"""Score combined forecasts and experts on the development test years: error metrics and a directional backtest.
Reads data/results/ from walk_forward.py; the holdout is never loaded."""

from pathlib import Path

import numpy as np
import polars as pl

from combine import load_combiner_inputs
from data import load_dev_hourly
from features import HOURS_PER_DAY

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "data" / "results"
EXPERT_FORECASTS_PATH = PROJECT_DIR / "data" / "experts" / "forecasts.parquet"

BASELINE_METHOD = "equal_weights"
NAIVE_EXPERT = "expert:naive"
CHALLENGERS = ["ewa", "ewa_per_hour", "ewa_reset", "boa", "tree"]
# Directional test: each hour, position = (forecast - naive) / the day's loss scale, capped at +/- 5 MWh.
MAX_POSITION_MWH = 5.0
DAYS_PER_YEAR = 365
PERCENT = 100


def load_evaluation_frame():
    """Return one row per (date, hour, method) with forecast and actual price, for combiners and experts."""
    combined = pl.read_parquet(RESULTS_DIR / "combined_forecasts.parquet")
    test_dates = combined.select("date").unique()
    experts = pl.read_parquet(EXPERT_FORECASTS_PATH).join(test_dates, on="date", how="semi")
    experts = experts.with_columns(("expert:" + pl.col("expert")).alias("method")).drop("expert")
    actual = load_dev_hourly().select("date", "hour", pl.col("price").alias("actual"))
    frame = pl.concat([combined, experts.select(combined.columns)])
    frame = frame.join(actual, on=["date", "hour"], how="inner")
    return frame.with_columns(pl.col("date").dt.year().alias("year"))


def error_metrics(frame):
    """Return MAE, rMAE (vs. the seasonal naive), RMSE and sMAPE per year and method."""
    absolute_error = (pl.col("forecast") - pl.col("actual")).abs()
    denominator = pl.col("forecast").abs() + pl.col("actual").abs()
    # sMAPE is undefined when forecast and price are both 0; count those hours as 0 error.
    symmetric_error = pl.when(denominator > 0).then(2 * absolute_error / denominator).otherwise(0.0)
    metrics = frame.group_by("year", "method").agg(
        absolute_error.mean().alias("mae"),
        (absolute_error**2).mean().sqrt().alias("rmse"),
        (symmetric_error.mean() * PERCENT).alias("smape"),
    )
    naive_mae = metrics.filter(pl.col("method") == NAIVE_EXPERT).select("year", pl.col("mae").alias("naive_mae"))
    metrics = metrics.join(naive_mae, on="year").with_columns((pl.col("mae") / pl.col("naive_mae")).alias("rmae"))
    return metrics.select("year", "method", "mae", "rmae", "rmse", "smape").sort("year", "mae")


def day_by_hour(frame, column):
    """Return dates and a (days, 24) array of `column` for one method's rows."""
    wide = frame.pivot(on="hour", index="date", values=column).sort("date")
    hour_columns = []
    for hour in range(HOURS_PER_DAY):
        hour_columns.append(str(hour))
    return wide["date"].to_list(), wide.select(hour_columns).to_numpy()


def directional_daily_pnl(forecasts, actual, reference, scale):
    """Each hour, bet that the price moves from the naive forecast toward ours, sized by the gap."""
    # Measuring against the seasonal naive, not yesterday's price, removes the predictable weekend
    # jumps every method would bet on; a method only earns when it knows more than the naive.
    positions = (forecasts - reference) / scale[:, np.newaxis]
    positions = np.clip(positions, -MAX_POSITION_MWH, MAX_POSITION_MWH)
    return (positions * (actual - reference)).sum(axis=1)


def max_drawdown(dates, daily_pnl):
    """Return the largest peak-to-trough fall of cumulative P&L, with its start and end dates."""
    cumulative = np.concatenate([[0.0], np.cumsum(daily_pnl)])
    running_peak = np.maximum.accumulate(cumulative)
    drawdowns = running_peak - cumulative
    trough = int(drawdowns.argmax())
    if drawdowns[trough] == 0:
        return 0.0, None, None
    peak = int(cumulative[: trough + 1].argmax())
    # Position 0 is the starting point before the first day, so position k ends on dates[k - 1].
    peak_date = dates[peak - 1] if peak > 0 else dates[0]
    return float(drawdowns[trough]), peak_date, dates[trough - 1]


def daily_loss_scale():
    """Return {date: seasonal naive's trailing 28-day MAE}, the same scale the combiners use."""
    inputs = load_combiner_inputs()
    scales = {}
    for day, scale in zip(inputs.dates, inputs.scale):
        scales[day] = scale
    return scales


def trading_metrics(frame):
    """Return total P&L, annualized Sharpe and max drawdown of the directional test per year and method."""
    scales = daily_loss_scale()
    naive_rows = frame.filter(pl.col("method") == NAIVE_EXPERT)
    rows = []
    for (year, method), group in frame.filter(pl.col("method") != NAIVE_EXPERT).group_by("year", "method"):
        dates, forecasts = day_by_hour(group, "forecast")
        _, actual = day_by_hour(group, "actual")
        _, reference = day_by_hour(naive_rows.filter(pl.col("year") == year), "forecast")
        scale = np.array([scales[day] for day in dates])
        daily_pnl = directional_daily_pnl(forecasts, actual, reference, scale)
        drawdown, drawdown_start, drawdown_end = max_drawdown(dates, daily_pnl)
        rows.append({
            "year": year,
            "method": method,
            "total_pnl": float(daily_pnl.sum()),
            "sharpe": float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(DAYS_PER_YEAR)),
            "max_drawdown": drawdown,
            "drawdown_start": drawdown_start,
            "drawdown_end": drawdown_end,
            "trade_days": int((daily_pnl != 0).sum()),
            "losing_days": int((daily_pnl < 0).sum()),
        })
    return pl.DataFrame(rows).sort("year", "sharpe", descending=[False, True])


def win_loss_tally(metrics, trading):
    """Count test years where each challenger beats 1/N on MAE and on battery Sharpe."""
    baseline_mae = metrics.filter(pl.col("method") == BASELINE_METHOD).select("year", pl.col("mae").alias("baseline_mae"))
    baseline_sharpe = trading.filter(pl.col("method") == BASELINE_METHOD).select(
        "year", pl.col("sharpe").alias("baseline_sharpe")
    )
    compared = metrics.join(baseline_mae, on="year").join(trading, on=["year", "method"]).join(baseline_sharpe, on="year")
    compared = compared.filter(pl.col("method").is_in(CHALLENGERS))
    return compared.group_by("method").agg(
        (pl.col("mae") < pl.col("baseline_mae")).sum().alias("lower_mae"),
        (pl.col("sharpe") > pl.col("baseline_sharpe")).sum().alias("higher_sharpe"),
        pl.len().alias("years"),
    ).sort("method")


def main():
    frame = load_evaluation_frame()
    metrics = error_metrics(frame)
    trading = trading_metrics(frame)
    tally = win_loss_tally(metrics, trading)

    metrics.write_csv(RESULTS_DIR / "metrics.csv")
    trading.write_csv(RESULTS_DIR / "trading.csv")
    tally.write_csv(RESULTS_DIR / "tally.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2):
        print(metrics)
        print(trading)
        print(tally)


if __name__ == "__main__":
    main()
