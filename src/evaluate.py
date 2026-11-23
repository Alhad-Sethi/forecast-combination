"""Score combined forecasts and experts on the development test years: MAE and the spread of the errors.
Reads data/results/ (or data/results/subset_<name>/ with --experts <name>) from walk_forward.py; the holdout is never loaded."""

from pathlib import Path

import numpy as np
import polars as pl

from combine import load_combiner_inputs
from data import load_dev_hourly
from walk_forward import FOLD_TEST_YEARS, fold_bounds, results_dir_for, scaled_mae, selected_subset

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXPERT_FORECASTS_PATH = PROJECT_DIR / "data" / "experts" / "forecasts.parquet"

BASELINE_METHOD = "equal_weights"
NAIVE_EXPERT = "expert:naive"
EXPERT_PREFIX = "expert:"
CHALLENGERS = ["ewa", "ewa_per_hour", "ewa_reset", "boa", "tree"]
# Both headline measures are errors, so lower is better for each.
HEADLINE_METRICS = ["mae", "error_std"]
PERCENT = 100


def load_evaluation_frame(results_dir, expert_names):
    """Return one row per (date, hour, method) with forecast and actual, for combiners and the pool's experts (+ naive)."""
    combined = pl.read_parquet(results_dir / "combined_forecasts.parquet")
    test_dates = combined.select("date").unique()
    experts = pl.read_parquet(EXPERT_FORECASTS_PATH).join(test_dates, on="date", how="semi")
    if expert_names is not None:
        # The seasonal naive stays in for rMAE even when it is not in the pool.
        experts = experts.filter(pl.col("expert").is_in(expert_names + ["naive"]))
    experts = experts.with_columns((EXPERT_PREFIX + pl.col("expert")).alias("method")).drop("expert")
    actual = load_dev_hourly().select("date", "hour", pl.col("price").alias("actual"))
    frame = pl.concat([combined, experts.select(combined.columns)])
    frame = frame.join(actual, on=["date", "hour"], how="inner")
    return frame.with_columns(pl.col("date").dt.year().alias("year"))


def error_metrics(frame):
    """Return MAE, rMAE (vs. the seasonal naive), error std, bias, RMSE and sMAPE per year and method."""
    error = pl.col("forecast") - pl.col("actual")
    denominator = pl.col("forecast").abs() + pl.col("actual").abs()
    # sMAPE is undefined when forecast and price are both 0; count those hours as 0 error.
    symmetric_error = pl.when(denominator > 0).then(2 * error.abs() / denominator).otherwise(0.0)
    # Population std, so that RMSE^2 = bias^2 + error_std^2 holds exactly.
    metrics = frame.group_by("year", "method").agg(
        error.abs().mean().alias("mae"),
        error.std(ddof=0).alias("error_std"),
        error.mean().alias("bias"),
        (error**2).mean().sqrt().alias("rmse"),
        (symmetric_error.mean() * PERCENT).alias("smape"),
    )
    naive_mae = metrics.filter(pl.col("method") == NAIVE_EXPERT).select("year", pl.col("mae").alias("naive_mae"))
    metrics = metrics.join(naive_mae, on="year").with_columns((pl.col("mae") / pl.col("naive_mae")).alias("rmae"))
    return metrics.select("year", "method", "mae", "rmae", "error_std", "bias", "rmse", "smape").sort("year", "mae")


def win_loss_tally(metrics):
    """Count test years where each challenger has a lower MAE and a lower error std than 1/N."""
    baseline = metrics.filter(pl.col("method") == BASELINE_METHOD).select(
        "year",
        pl.col("mae").alias("baseline_mae"),
        pl.col("error_std").alias("baseline_error_std"),
    )
    compared = metrics.filter(pl.col("method").is_in(CHALLENGERS)).join(baseline, on="year")
    return compared.group_by("method").agg(
        (pl.col("mae") < pl.col("baseline_mae")).sum().alias("lower_mae"),
        (pl.col("error_std") < pl.col("baseline_error_std")).sum().alias("lower_error_std"),
        pl.len().alias("years"),
    ).sort("method")


def pool_methods(frame, expert_names):
    """Return the method names ("expert:<name>") of the experts the combiners used."""
    if expert_names is None:
        methods = frame.filter(pl.col("method").str.starts_with(EXPERT_PREFIX))["method"].unique()
        return sorted(methods.to_list())
    methods = []
    for name in expert_names:
        methods.append(EXPERT_PREFIX + name)
    return sorted(methods)


def selected_experts(expert_names):
    """Return, per test year, the pool expert with the best combiner selection score over that fold's training years."""
    inputs = load_combiner_inputs(expert_names=expert_names)
    rows = []
    for test_year in FOLD_TEST_YEARS:
        training_start, test_start, _ = fold_bounds(inputs, test_year)
        best_name = None
        best_score = np.inf
        for expert_index, expert_name in enumerate(inputs.expert_names):
            expert_forecasts = inputs.forecasts[training_start:test_start, expert_index]
            score = scaled_mae(inputs, expert_forecasts, training_start, test_start)
            if score < best_score:
                best_name = expert_name
                best_score = score
        rows.append({"year": test_year, "selected_expert": EXPERT_PREFIX + best_name})
    return pl.DataFrame(rows)


def best_expert_in_hindsight(metrics, column, pool):
    """Return, per year, the pool expert with the lowest test-year `column` and its value."""
    experts = metrics.filter(pl.col("method").is_in(pool)).sort("year", column)
    best = experts.group_by("year", maintain_order=True).first()
    return best.select(
        "year",
        pl.col("method").alias(f"hindsight_expert_{column}"),
        pl.col(column).alias(f"hindsight_{column}"),
    )


def selected_expert_scores(metrics, column, selected):
    """Return, per year, the training-selected expert's test-year `column`."""
    scores = metrics.select("year", pl.col("method").alias("selected_expert"), pl.col(column).alias(f"selected_{column}"))
    return selected.join(scores, on=["year", "selected_expert"], how="left").select("year", f"selected_{column}")


def expert_comparison(metrics, selected, pool):
    """Return each combiner's test-year MAE and error std next to the hindsight-best and training-selected experts."""
    combiners = metrics.filter(pl.col("method").is_in([BASELINE_METHOD] + CHALLENGERS))
    compared = combiners.select("year", "method", *HEADLINE_METRICS).join(selected, on="year")
    for column in HEADLINE_METRICS:
        compared = compared.join(best_expert_in_hindsight(metrics, column, pool), on="year")
        compared = compared.join(selected_expert_scores(metrics, column, selected), on="year")
    return compared.sort("year", "mae")


def expert_tally(comparison):
    """Count test years where each combiner beats the hindsight-best and the training-selected expert."""
    counts = []
    for column in HEADLINE_METRICS:
        counts.append((pl.col(column) < pl.col(f"hindsight_{column}")).sum().alias(f"{column}_beats_hindsight"))
        counts.append((pl.col(column) < pl.col(f"selected_{column}")).sum().alias(f"{column}_beats_selected"))
    return comparison.group_by("method").agg(*counts, pl.len().alias("years")).sort("method")


def expert_error_matrix(frame, year, pool):
    """Return the pool's expert names and an (experts, hours) array of their forecast errors for one test year."""
    experts = frame.filter(pl.col("method").is_in(pool) & (pl.col("year") == year))
    experts = experts.with_columns((pl.col("forecast") - pl.col("actual")).alias("error"))
    wide = experts.pivot(on="method", index=["date", "hour"], values="error").sort("date", "hour")
    names = []
    for column in wide.columns:
        if column.startswith(EXPERT_PREFIX):
            names.append(column)
    names.sort()
    return names, wide.select(names).to_numpy().T


def expert_diversity(frame, pool):
    """Per test year and pool expert: MAE, mean error correlation with the others, and how much 1/N worsens without it."""
    rows = []
    for year in sorted(frame["year"].unique().to_list()):
        names, errors = expert_error_matrix(frame, year, pool)
        correlations = np.corrcoef(errors)
        # The error of an average forecast is the average of the experts' errors.
        equal_weight_mae = np.abs(errors.mean(axis=0)).mean()
        for index, name in enumerate(names):
            others = np.delete(np.arange(len(names)), index)
            mae_without = np.abs(errors[others].mean(axis=0)).mean()
            rows.append({
                "year": year,
                "expert": name,
                "mae": float(np.abs(errors[index]).mean()),
                "mean_error_correlation": float(correlations[index, others].mean()),
                "equal_weight_mae_without": float(mae_without),
                "contribution": float(mae_without - equal_weight_mae),
            })
    return pl.DataFrame(rows).sort("year", "contribution", descending=[False, True])


def main():
    subset_name, expert_names = selected_subset()
    results_dir = results_dir_for(subset_name)
    frame = load_evaluation_frame(results_dir, expert_names)
    pool = pool_methods(frame, expert_names)
    metrics = error_metrics(frame)
    tally = win_loss_tally(metrics)
    comparison = expert_comparison(metrics, selected_experts(expert_names), pool)
    comparison_tally = expert_tally(comparison)
    diversity = expert_diversity(frame, pool)

    metrics.write_csv(results_dir / "metrics.csv")
    tally.write_csv(results_dir / "tally.csv")
    comparison.write_csv(results_dir / "expert_comparison.csv")
    comparison_tally.write_csv(results_dir / "expert_tally.csv")
    diversity.write_csv(results_dir / "expert_diversity.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2):
        print(metrics)
        print(tally)
        print(comparison)
        print(comparison_tally)
        print(diversity)


if __name__ == "__main__":
    main()
