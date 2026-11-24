"""Run the development folds: pick each combiner's hyperparameters on the 3 years before a test year,
then run every method through that year. Writes forecasts, selections, grid scores and weights to data/results/
(or data/results/subset_<name>/ with --experts <name>, where the combiners see only that subset of experts).
With --holdout it runs the same procedure once on the holdout years (frozen config only) into data/results/holdout/."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from combine import equal_weights, load_combiner_inputs, run_boa, run_ewa, run_tree
from features import HOURS_PER_DAY

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "data" / "results"

FOLD_TEST_YEARS = [2020, 2021, 2022, 2023]
# The last holdout fold ends on the latest day with expert forecasts, not on 31 December.
HOLDOUT_TEST_YEARS = [2024, 2025, 2026]
TRAINING_YEARS = 3
ETA_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
HALF_LIFE_GRID_DAYS = [1, 3, 7, 30, 365, np.inf]
TREE_DEPTH_GRID = [1, 2, 3]
TREE_MIN_LEAF_GRID_HOURS = [200, 500, 1000, 2000]

# Expert subsets, each fixed by a rule on 2017-2019 only (before the first test fold).
# A: drop the experts whose removal improved 1/N in all three pre-test years (naive, arx, rf, knn).
EXPERT_SUBSETS = {
    "A": [
        "lear_56",
        "lear_84",
        "lear_112",
        "lear_364",
        "lear_728",
        "lear_fr_112",
        "lear_fr_728",
        "lgbm",
        "lgbm_change",
        "lgbm_fr",
        "lgbm_change_fr",
    ],
}


def holdout_requested():
    """Return True if --holdout is on the command line (the single run of the frozen config)."""
    return "--holdout" in sys.argv


def test_years_for(holdout):
    """Return the test years of the development folds or of the holdout folds."""
    if holdout:
        return HOLDOUT_TEST_YEARS
    return FOLD_TEST_YEARS


def selected_subset():
    """Return (subset name, expert names) from --experts on the command line, or (None, None) for all experts."""
    if "--experts" not in sys.argv:
        return None, None
    if holdout_requested():
        raise SystemExit("--holdout runs the frozen config on all experts only; drop --experts")
    name = sys.argv[sys.argv.index("--experts") + 1]
    if name not in EXPERT_SUBSETS:
        raise SystemExit(f"--experts {name}: unknown; known subsets are {list(EXPERT_SUBSETS)}")
    return name, EXPERT_SUBSETS[name]


def results_dir_for(subset_name, holdout=False):
    """Return where a run's results go: data/results/ (or holdout/) for all experts, data/results/subset_<name>/ otherwise."""
    if holdout:
        return RESULTS_DIR / "holdout"
    if subset_name is None:
        return RESULTS_DIR
    return RESULTS_DIR / f"subset_{subset_name}"


def day_index(inputs, day):
    """Return the position of `day` in the combiner inputs."""
    return inputs.dates.index(day)


def fold_bounds(inputs, test_year):
    """Return (training start, test start, test end) day indices for one fold; the test ends with the year or the data."""
    training_start = day_index(inputs, date(test_year - TRAINING_YEARS, 1, 1))
    test_start = day_index(inputs, date(test_year, 1, 1))
    test_end = test_start
    while test_end < len(inputs.dates) and inputs.dates[test_end].year == test_year:
        test_end += 1
    return training_start, test_start, test_end


def scaled_mae(inputs, combined, start, end):
    """Return the mean over updatable days of the daily MAE divided by that day's scale."""
    days = np.arange(start, end)
    daily_errors = np.abs(combined - inputs.actual[days]).mean(axis=1) / inputs.scale[days]
    return float(daily_errors[inputs.can_update[days]].mean())


def select_ewa(inputs, training_start, test_start, per_hour):
    """Score every (eta, half-life) on the training years; return the best pair and all scores."""
    scores = []
    for eta in ETA_GRID:
        for half_life_days in HALF_LIFE_GRID_DAYS:
            combined, _ = run_ewa(inputs, training_start, test_start, eta, half_life_days, per_hour)
            score = scaled_mae(inputs, combined, training_start, test_start)
            scores.append({"eta": eta, "half_life_days": half_life_days, "score": score})
    # Grid order puts smaller eta first, so ties go to the more regularized setting.
    best = min(scores, key=lambda row: row["score"])
    return best, scores


def select_tree(inputs, training_start, test_start):
    """Score every (depth, minimum leaf) on the training years; return the best pair and all scores."""
    scores = []
    for max_depth in TREE_DEPTH_GRID:
        for min_leaf_hours in TREE_MIN_LEAF_GRID_HOURS:
            combined = run_tree(inputs, training_start, test_start, max_depth, min_leaf_hours)
            score = scaled_mae(inputs, combined, training_start, test_start)
            scores.append({"max_depth": max_depth, "min_leaf_hours": min_leaf_hours, "score": score})
    best = min(scores, key=lambda row: row["score"])
    return best, scores


def run_fold(inputs, test_year):
    """Select hyperparameters on the training years and return test-year forecasts, selections and weights."""
    training_start, test_start, test_end = fold_bounds(inputs, test_year)
    test_length = test_end - test_start
    forecasts = {"equal_weights": equal_weights(inputs, test_start, test_end)}
    selections = {}
    grid_scores = {}
    weights = {}

    best_ewa, grid_scores["ewa"] = select_ewa(inputs, training_start, test_start, per_hour=False)
    combined, ewa_weights = run_ewa(inputs, training_start, test_end, best_ewa["eta"], best_ewa["half_life_days"])
    forecasts["ewa"] = combined[-test_length:]
    weights["ewa"] = ewa_weights[-test_length:]
    selections["ewa"] = best_ewa

    best_hourly, grid_scores["ewa_per_hour"] = select_ewa(inputs, training_start, test_start, per_hour=True)
    combined, _ = run_ewa(inputs, training_start, test_end, best_hourly["eta"], best_hourly["half_life_days"], True)
    forecasts["ewa_per_hour"] = combined[-test_length:]
    selections["ewa_per_hour"] = best_hourly

    # Variant: same selected settings, but weights restart from uniform on 1 January of the test year.
    forecasts["ewa_reset"], _ = run_ewa(inputs, test_start, test_end, best_ewa["eta"], best_ewa["half_life_days"])
    selections["ewa_reset"] = best_ewa

    # BOA has nothing to select; it runs continuously from the first forecast day.
    combined, boa_weights = run_boa(inputs, 0, test_end)
    forecasts["boa"] = combined[-test_length:]
    weights["boa"] = boa_weights[-test_length:]

    best_tree, grid_scores["tree"] = select_tree(inputs, training_start, test_start)
    forecasts["tree"] = run_tree(inputs, test_start, test_end, best_tree["max_depth"], best_tree["min_leaf_hours"])
    selections["tree"] = best_tree
    return forecasts, selections, grid_scores, weights


def forecasts_frame(inputs, test_year, forecasts):
    """Return one row per (date, hour, method) for a fold's test year."""
    _, test_start, test_end = fold_bounds(inputs, test_year)
    dates = np.array(inputs.dates[test_start:test_end], dtype="datetime64[D]")
    frames = []
    for method, combined in forecasts.items():
        frame = pl.DataFrame({
            "date": np.repeat(dates, HOURS_PER_DAY),
            "hour": np.tile(np.arange(HOURS_PER_DAY, dtype=np.int8), len(dates)),
            "forecast": combined.reshape(-1),
        })
        frames.append(frame.with_columns(pl.lit(method).alias("method")))
    return pl.concat(frames).select("date", "hour", "method", "forecast")


def weights_frame(inputs, test_year, weights):
    """Return one row per (date, method, expert) with the shared daily weight."""
    _, test_start, test_end = fold_bounds(inputs, test_year)
    dates = np.array(inputs.dates[test_start:test_end], dtype="datetime64[D]")
    num_experts = len(inputs.expert_names)
    frames = []
    for method, history in weights.items():
        frame = pl.DataFrame({
            "date": np.repeat(dates, num_experts),
            "expert": np.tile(np.array(inputs.expert_names), len(dates)),
            "weight": history.reshape(-1),
        })
        frames.append(frame.with_columns(pl.lit(method).alias("method")))
    return pl.concat(frames).select("date", "method", "expert", "weight")


def grid_frame(test_year, grid_scores):
    """Return every hyperparameter setting's training score for one fold."""
    rows = []
    for method, scores in grid_scores.items():
        for score in scores:
            row = {"test_year": test_year, "method": method}
            row.update(score)
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def selections_frame(test_year, selections):
    """Return the chosen hyperparameters per method for one fold."""
    rows = []
    for method, selection in selections.items():
        row = {"test_year": test_year, "method": method}
        row.update(selection)
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def main():
    holdout = holdout_requested()
    subset_name, expert_names = selected_subset()
    results_dir = results_dir_for(subset_name, holdout)
    inputs = load_combiner_inputs(include_holdout=holdout, expert_names=expert_names)
    print(f"experts: {inputs.expert_names}", flush=True)
    print(f"days: {inputs.dates[0]} to {inputs.dates[-1]}", flush=True)
    forecast_frames = []
    weight_frames = []
    grid_frames = []
    selection_frames = []
    for test_year in test_years_for(holdout):
        print(f"fold {test_year}", flush=True)
        forecasts, selections, grid_scores, weights = run_fold(inputs, test_year)
        forecast_frames.append(forecasts_frame(inputs, test_year, forecasts))
        weight_frames.append(weights_frame(inputs, test_year, weights))
        grid_frames.append(grid_frame(test_year, grid_scores))
        selection_frames.append(selections_frame(test_year, selections))

    results_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(forecast_frames).write_parquet(results_dir / "combined_forecasts.parquet")
    pl.concat(weight_frames).write_parquet(results_dir / "weights.parquet")
    pl.concat(grid_frames, how="diagonal").write_csv(results_dir / "grid_scores.csv")
    selections = pl.concat(selection_frames, how="diagonal")
    selections.write_csv(results_dir / "selections.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(selections)


if __name__ == "__main__":
    main()
