"""Write a notebook from the cell sources below, chosen by the output file name.
Run: python src/build_notebook.py notebooks/explore_entsoe.ipynb (or notebooks/combiner_diagnostics.ipynb)"""

import sys
from pathlib import Path

import nbformat

EXPLORE_CELLS = [
    ("markdown", """# Explore the clean ENTSO-E data (Germany)

Built on `data/clean/de_hourly.parquet` and `data/clean/day_flags.parquet` (run `src/clean_entsoe.py` and `src/anomalies.py` first).

**Everything here stops at 2023-12-31.** The loaders in `src/data.py` cut off the 2024+ holdout, which stays unseen until the config is frozen."""),
    ("code", """import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, str(Path("..") / "src"))
from data import load_dev_day_flags, load_dev_hourly

CLEAN_DIR = Path("..") / "data" / "clean"
HOLDOUT_START_YEAR = 2024
NUM_CONTEXT_DAYS = 7
NUM_EXAMPLE_FLAGS = 4

plt.rcParams["figure.figsize"] = (12, 4)
pl.Config.set_tbl_rows(30)"""),
    ("code", """def add_timestamp(hourly):
    \"\"\"Add a naive local timestamp (date + hour) for plotting.\"\"\"
    timestamp = pl.col("date").cast(pl.Datetime) + pl.duration(hours=pl.col("hour"))
    return hourly.with_columns(timestamp.alias("timestamp"))


hourly = add_timestamp(load_dev_hourly())
flags = load_dev_day_flags()
hourly.head()"""),
    ("markdown", """## Data quality

How each hourly value was obtained (`observed`, `interpolated`, `previous_week`, `missing`), and how many values broke an impossible-value rule. The only gaps are early 2015: prices start on 2015-01-05, and one day of offshore wind has no previous week to copy from."""),
    ("code", """fill_report = pl.read_csv(CLEAN_DIR / "fill_report.csv").filter(pl.col("year") < HOLDOUT_START_YEAR)
fill_report"""),
    ("code", """# Explicit types: with no flags the file is header-only and every column would be read as text.
flag_schema = {"year": pl.Int32, "column": pl.String, "rule": pl.String, "len": pl.UInt32}
flag_counts = pl.read_csv(CLEAN_DIR / "flag_counts.csv", schema=flag_schema)
flag_counts = flag_counts.filter(pl.col("year") < HOLDOUT_START_YEAR)
print(f"impossible-value flags 2015-2023: {flag_counts['len'].sum()}")"""),
    ("markdown", "## Prices over time"),
    ("code", """daily_prices = hourly.group_by("date").agg(pl.col("price").mean()).sort("date")

fig, ax = plt.subplots()
ax.plot(daily_prices["date"], daily_prices["price"], linewidth=0.7)
ax.set_ylabel("EUR/MWh")
ax.set_title("Daily mean day-ahead price, Germany")
plt.show()"""),
    ("code", """def yearly_price_stats(hourly):
    \"\"\"Return per-year price statistics, including the number of negative-price hours.\"\"\"
    return (
        hourly.group_by(pl.col("date").dt.year().alias("year"))
        .agg(
            pl.col("price").mean().alias("mean"),
            pl.col("price").std().alias("std"),
            pl.col("price").quantile(0.01).alias("p01"),
            pl.col("price").median().alias("median"),
            pl.col("price").quantile(0.99).alias("p99"),
            pl.col("price").max().alias("max"),
            (pl.col("price") < 0).sum().alias("negative_hours"),
        )
        .sort("year")
    )


yearly_price_stats(hourly).with_columns(pl.selectors.float().round(1))"""),
    ("markdown", """The 2021-22 crisis makes errors roughly 10x larger than in 2019. This is why the combiner losses are divided by the seasonal naive's trailing MAE. The average change from the same hour yesterday shows the scale shift:"""),
    ("code", """# |price(h) - price(h, day before)| is the seasonal naive's error on Tuesday-Friday.
day_over_day = hourly.sort("date", "hour").with_columns(
    (pl.col("price") - pl.col("price").shift(1).over("hour")).abs().alias("change")
)
monthly_change = day_over_day.group_by_dynamic("timestamp", every="1mo").agg(pl.col("change").mean())

fig, ax = plt.subplots()
ax.plot(monthly_change["timestamp"], monthly_change["change"], marker=".")
ax.set_yscale("log")
ax.set_ylabel("EUR/MWh (log scale)")
ax.set_title("Monthly mean |price(h) - price(h, day before)|")
plt.show()"""),
    ("markdown", "## Daily shape"),
    ("code", """def mean_profile_by_day_type(hourly):
    \"\"\"Return mean price per hour of day, split into weekday, Saturday and Sunday.\"\"\"
    weekday_number = pl.col("date").dt.weekday()
    day_type = (
        pl.when(weekday_number == 6).then(pl.lit("saturday"))
        .when(weekday_number == 7).then(pl.lit("sunday"))
        .otherwise(pl.lit("weekday"))
    )
    profile = hourly.group_by("hour", day_type.alias("day_type")).agg(pl.col("price").mean())
    return profile.pivot(on="day_type", index="hour", values="price").sort("hour")


periods = {"2017-2019": (2017, 2019), "2020": (2020, 2020), "2022": (2022, 2022)}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (label, (first_year, last_year)) in zip(axes, periods.items()):
    in_period = pl.col("date").dt.year().is_between(first_year, last_year)
    profile = mean_profile_by_day_type(hourly.filter(in_period))
    for day_type in ["weekday", "saturday", "sunday"]:
        ax.plot(profile["hour"], profile[day_type], label=day_type)
    ax.set_title(label)
    ax.set_xlabel("hour of day")
axes[0].set_ylabel("EUR/MWh")
axes[0].legend()
plt.show()"""),
    ("markdown", """## Prices against fundamentals

Residual load = load forecast - wind - solar: the demand left for thermal plants, which set the price."""),
    ("code", """with_residual = hourly.with_columns(
    (pl.col("load_forecast_winsorized") - pl.col("renewables_total")).alias("residual_load")
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, year in zip(axes, [2019, 2021, 2023]):
    year_data = with_residual.filter(pl.col("date").dt.year() == year)
    ax.scatter(year_data["residual_load"] / 1000, year_data["price"], s=1, alpha=0.2)
    ax.set_title(str(year))
    ax.set_xlabel("residual load forecast (GW)")
axes[0].set_ylabel("price (EUR/MWh)")
plt.show()"""),
    ("code", """with_residual.group_by(pl.col("date").dt.year().alias("year")).agg(
    pl.corr("price", "residual_load").round(2).alias("corr(price, residual_load)")
).sort("year")"""),
    ("markdown", """## Load winsorizing

Each hour is clipped to the 0.5% and 99.5% quantiles of the same hour over the previous 364 days. It clips 2-4% of hours (more than the nominal 1%, because load drifts between years); kept as is by choice."""),
    ("code", """clipped = hourly.with_columns((pl.col("load_forecast") != pl.col("load_forecast_winsorized")).alias("clipped"))
clipped.group_by(pl.col("date").dt.year().alias("year")).agg(
    (pl.col("clipped").mean() * 100).round(1).alias("percent_clipped")
).sort("year")"""),
    ("code", """winter_week = hourly.filter(pl.col("date").is_between(pl.date(2022, 12, 18), pl.date(2023, 1, 8)))

fig, ax = plt.subplots()
ax.plot(winter_week["timestamp"], winter_week["load_forecast"] / 1000, label="raw", linewidth=1)
ax.plot(winter_week["timestamp"], winter_week["load_forecast_winsorized"] / 1000, label="winsorized", linewidth=1)
ax.set_ylabel("GW")
ax.set_title("Load forecast over Christmas 2022: holiday lows get clipped")
ax.legend()
plt.show()"""),
    ("markdown", """## Outlier days

A day is an outlier if either detector fires:
- **z-score**: mean |robust z| over all hours of load, wind and solar, above the 99th percentile of the previous 364 days. The MAD is floored at 2% of the series' mean level and each |z| is capped at 10, so near-zero solar hours at dawn and dusk can't dominate.
- **isolation forest**: daily mean/min/max of load, wind, solar and residual load, contamination 0.5%, refit yearly. It also flags record wind/solar days from capacity growth, kept on purpose for diversity.

Outlier days are dropped as training targets but stay in evaluation."""),
    ("code", """flags.group_by(pl.col("date").dt.year().alias("year")).agg(
    pl.col("z_flag").sum(),
    pl.col("forest_flag").sum(),
    (pl.col("z_flag") & pl.col("forest_flag")).sum().alias("both"),
    pl.col("is_outlier").sum(),
).sort("year")"""),
    ("code", """flagged_by_month = flags.filter(pl.col("is_outlier")).group_by(
    pl.col("date").dt.month().alias("month")
).agg(pl.col("z_flag").sum(), pl.col("forest_flag").sum()).sort("month")

fig, ax = plt.subplots()
ax.bar(flagged_by_month["month"] - 0.2, flagged_by_month["z_flag"], width=0.4, label="z-score")
ax.bar(flagged_by_month["month"] + 0.2, flagged_by_month["forest_flag"], width=0.4, label="isolation forest")
ax.set_xticks(range(1, 13))
ax.set_xlabel("month")
ax.set_ylabel("flagged days, 2015-2023")
ax.legend()
plt.show()"""),
    ("code", """daily_fundamentals = hourly.group_by("date").agg(
    (pl.col("load_forecast").mean() / 1000).alias("load"),
    ((pl.col("wind_onshore") + pl.col("wind_offshore")).mean() / 1000).alias("wind"),
    (pl.col("solar").mean() / 1000).alias("solar"),
).sort("date").join(flags.select("date", "z_flag", "forest_flag"), on="date")

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, column in zip(axes, ["load", "wind", "solar"]):
    ax.plot(daily_fundamentals["date"], daily_fundamentals[column], linewidth=0.5, color="grey")
    z_days = daily_fundamentals.filter(pl.col("z_flag"))
    forest_days = daily_fundamentals.filter(pl.col("forest_flag"))
    ax.scatter(z_days["date"], z_days[column], s=12, label="z-score", zorder=3)
    ax.scatter(forest_days["date"], forest_days[column], s=12, marker="x", label="isolation forest", zorder=3)
    ax.set_ylabel(f"{column} (GW, daily mean)")
axes[0].legend()
plt.show()"""),
    ("markdown", "### Flagged days in context (two weeks either side, flagged day shaded)"),
    ("code", """def plot_flag_context(hourly, flagged_date, title):
    \"\"\"Plot hourly load, wind and solar around `flagged_date`, shading the flagged day.\"\"\"
    window = hourly.filter(
        pl.col("date").is_between(
            flagged_date - timedelta(days=NUM_CONTEXT_DAYS),
            flagged_date + timedelta(days=NUM_CONTEXT_DAYS),
        )
    )
    wind = (window["wind_onshore"] + window["wind_offshore"]) / 1000

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(window["timestamp"], window["load_forecast"] / 1000, label="load", linewidth=0.8)
    ax.plot(window["timestamp"], wind, label="wind", linewidth=0.8)
    ax.plot(window["timestamp"], window["solar"] / 1000, label="solar", linewidth=0.8)
    day_start = window.filter(pl.col("date") == flagged_date)["timestamp"].min()
    ax.axvspan(day_start, day_start + timedelta(days=1), color="red", alpha=0.15)
    ax.set_ylabel("GW")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.show()


strongest_z = flags.filter(pl.col("z_flag")).sort("z_score", descending=True).head(NUM_EXAMPLE_FLAGS)
for row in strongest_z.iter_rows(named=True):
    plot_flag_context(hourly, row["date"], f"z-score flag {row['date']} (score {row['z_score']:.1f})")"""),
    ("code", """# Lower isolation-forest scores mean more anomalous.
strongest_forest = flags.filter(pl.col("forest_flag")).sort("forest_score").head(NUM_EXAMPLE_FLAGS)
for row in strongest_forest.iter_rows(named=True):
    plot_flag_context(hourly, row["date"], f"isolation forest flag {row['date']} (score {row['forest_score']:.3f})")"""),
]

DIAGNOSTIC_CELLS = [
    ("markdown", """# Combiner diagnostics (Germany, development test years 2020-2023)

Weights over time and EWA's learning-rate grid. Built on `data/results/weights.parquet`, `metrics.csv`, `selections.csv` and `grid_scores.csv` (run `src/walk_forward.py` and `src/evaluate.py` first).

EWA's weights come from a separate run per fold: each starts uniform 3 years before its test year and runs through it. The dashed lines mark where one fold's run hands over to the next. BOA is one continuous run from 2017-01-01."""),
    ("code", """from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

RESULTS_DIR = Path("..") / "data" / "results"
# The palette has 8 colour-blind-safe hues, so the stacked plots show expert families, not single experts.
OTHER_BAND = "naive, arx, rf, knn"
GERMAN_LEAR_BAND = "lear (5 German windows)"
FRENCH_LEAR_BAND = "lear_fr (112, 728)"
EXPERT_BANDS = {
    "naive": OTHER_BAND,
    "arx": OTHER_BAND,
    "rf": OTHER_BAND,
    "knn": OTHER_BAND,
    "lear_56": GERMAN_LEAR_BAND,
    "lear_84": GERMAN_LEAR_BAND,
    "lear_112": GERMAN_LEAR_BAND,
    "lear_364": GERMAN_LEAR_BAND,
    "lear_728": GERMAN_LEAR_BAND,
    "lear_fr_112": FRENCH_LEAR_BAND,
    "lear_fr_728": FRENCH_LEAR_BAND,
    "lgbm": "lgbm",
    "lgbm_change": "lgbm_change",
    "lgbm_fr": "lgbm_fr",
    "lgbm_change_fr": "lgbm_change_fr",
}
# The fixed order sets both the stacking order and each band's colour, identical in every chart.
EXPERT_ORDER = [OTHER_BAND, GERMAN_LEAR_BAND, FRENCH_LEAR_BAND, "lgbm", "lgbm_change", "lgbm_fr", "lgbm_change_fr"]
# Grey for the folded band, then the categorical palette checked for separation between neighbouring bands.
EXPERT_COLOURS = ["#9e9d96", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
METHODS = ["ewa", "boa"]
METHOD_COLOURS = {"ewa": "#2a78d6", "boa": "#eb6834"}
FOLD_STARTS = [date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)]
TRAINING_YEARS = 3
HALF_LIVES = [1.0, 3.0, 7.0, 30.0, 365.0, float("inf")]
HALF_LIFE_LABELS = {
    1.0: "1 day",
    3.0: "3 days",
    7.0: "7 days",
    30.0: "30 days",
    365.0: "365 days",
    float("inf"): "no forgetting",
}
# One blue ramp, light to dark, because half-life is ordered (short to long memory).
HALF_LIFE_COLOURS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams["figure.figsize"] = (14, 4)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
pl.Config.set_tbl_rows(40)

weights = pl.read_parquet(RESULTS_DIR / "weights.parquet")
metrics = pl.read_csv(RESULTS_DIR / "metrics.csv")
selections = pl.read_csv(RESULTS_DIR / "selections.csv")
selections.filter(pl.col("method") == "ewa").select("test_year", "eta", "half_life_days", "score")"""),
    ("code", """def weight_matrix(weights, method):
    \"\"\"Return dates and a (days, bands) array of one method's daily weights, columns in EXPERT_ORDER.\"\"\"
    banded = weights.filter(pl.col("method") == method).with_columns(
        pl.col("expert").replace_strict(EXPERT_BANDS).alias("band")
    )
    banded = banded.group_by("date", "band").agg(pl.col("weight").sum())
    wide = banded.pivot(on="band", index="date", values="weight")
    wide = wide.sort("date")
    return wide["date"], wide.select(EXPERT_ORDER).to_numpy()


def mark_fold_starts(ax):
    \"\"\"Draw a dashed line where each new fold's run begins.\"\"\"
    for fold_start in FOLD_STARTS:
        ax.axvline(fold_start, color="black", linewidth=0.8, linestyle="--")


def plot_weight_stack(ax, dates, daily_weights, title):
    \"\"\"Draw daily weights as a stacked area summing to 1, one band per expert.\"\"\"
    ax.stackplot(
        dates,
        daily_weights.T,
        colors=EXPERT_COLOURS,
        labels=EXPERT_ORDER,
        edgecolor="white",
        linewidth=0.2,
    )
    mark_fold_starts(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("weight")
    ax.set_title(title, loc="left")"""),
    ("markdown", """## Daily weights

Each band is an expert family (summed weights), stacked bottom to top in the legend's order. Weights of single experts are in the table at the end."""),
    ("code", """fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, method in zip(axes, METHODS):
    dates, daily_weights = weight_matrix(weights, method)
    plot_weight_stack(ax, dates, daily_weights, method.upper())
handles, labels = axes[0].get_legend_handles_labels()
# Reversed so the legend reads top to bottom in the same order as the bands.
fig.legend(handles[::-1], labels[::-1], loc="center right", frameon=False)
fig.subplots_adjust(right=0.82)
plt.show()"""),
    ("markdown", """## How concentrated are the weights?

Effective number of experts = 1 / sum of squared weights: the number of experts (15) means equal weights (1/N), 1 means all weight on a single expert."""),
    ("code", """def effective_experts(weights):
    \"\"\"Return one row per (date, method) with the effective number of experts.\"\"\"
    concentration = weights.group_by("date", "method").agg((1 / (pl.col("weight") ** 2).sum()).alias("effective"))
    return concentration.sort("date")


concentration = effective_experts(weights)
fig, ax = plt.subplots()
for method in METHODS:
    series = concentration.filter(pl.col("method") == method)
    ax.plot(series["date"], series["effective"], linewidth=0.8, color=METHOD_COLOURS[method], label=method.upper())
mark_fold_starts(ax)
ax.set_ylim(1, weights["expert"].n_unique())
ax.set_ylabel("effective number of experts")
ax.legend(frameon=False)
plt.show()

concentration.group_by(pl.col("date").dt.year().alias("year"), "method").agg(
    pl.col("effective").mean().alias("mean"),
    pl.col("effective").min().alias("min"),
).sort("year", "method")"""),
    ("markdown", """## Mean weight vs. how good each expert was that year

Each expert's mean weight per test year, next to its test-year MAE and its MAE rank among the experts (1 = best)."""),
    ("code", """def mean_weights_by_year(weights):
    \"\"\"Return one row per (year, expert) with each method's mean daily weight.\"\"\"
    yearly = weights.group_by(pl.col("date").dt.year().alias("year"), "expert", "method").agg(pl.col("weight").mean())
    return yearly.pivot(on="method", index=["year", "expert"], values="weight")


def expert_mae_ranks(metrics):
    \"\"\"Return one row per (year, expert) with the test-year MAE and its rank among the experts.\"\"\"
    experts = metrics.filter(pl.col("method").str.starts_with("expert:"))
    experts = experts.with_columns(pl.col("method").str.replace("expert:", "").alias("expert"))
    ranked = experts.with_columns(pl.col("mae").rank("ordinal").over("year").cast(pl.Int32).alias("mae_rank"))
    return ranked.select("year", "expert", "mae", "mae_rank")


weight_vs_skill = mean_weights_by_year(weights).join(expert_mae_ranks(metrics), on=["year", "expert"])
weight_vs_skill.sort("year", "mae_rank").with_columns(pl.selectors.float().round(3))"""),
    ("markdown", """## EWA learning rate vs. training score

Selection score (mean scaled daily MAE over the 3 training years, outlier days excluded; lower is better) for every (eta, half-life) on the grid. The dashed grey line is eta = 0, which is 1/N. The circled point is the selected setting."""),
    ("code", """def plot_eta_scores(ax, fold_scores, test_year):
    \"\"\"Draw training score vs. eta for one fold, one line per half-life, with 1/N as a reference.\"\"\"
    equal_weight_score = fold_scores.filter(pl.col("eta") == 0)["score"][0]
    learning = fold_scores.filter(pl.col("eta") > 0)
    for half_life, colour in zip(HALF_LIVES, HALF_LIFE_COLOURS):
        line = learning.filter(pl.col("half_life_days") == half_life).sort("eta")
        ax.plot(line["eta"], line["score"], marker="o", markersize=4, linewidth=1.5, color=colour,
                label=HALF_LIFE_LABELS[half_life])
    best = fold_scores.sort("score").row(0, named=True)
    ax.scatter(best["eta"], best["score"], s=120, facecolors="none", edgecolors="black", zorder=3)
    ax.axhline(equal_weight_score, color="grey", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("eta")
    ax.set_title(f"train {test_year - TRAINING_YEARS}-{test_year - 1} (test {test_year})", loc="left")


grid_scores = pl.read_csv(RESULTS_DIR / "grid_scores.csv").filter(pl.col("method") == "ewa")
test_years = grid_scores["test_year"].unique().sort().to_list()
fig, axes = plt.subplots(1, len(test_years), figsize=(16, 4))
for ax, test_year in zip(axes, test_years):
    plot_eta_scores(ax, grid_scores.filter(pl.col("test_year") == test_year), test_year)
axes[0].set_ylabel("training score (scaled MAE)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="half-life", loc="center right", frameon=False)
fig.tight_layout(rect=(0, 0, 0.9, 1))
plt.show()"""),
    ("markdown", """The best eta shrinks as memory grows: the curves trace a ridge along which eta x (effective memory) stays roughly constant, and the short-memory end of that ridge scores best."""),
    ("code", """best_per_half_life = grid_scores.sort("score").group_by("test_year", "half_life_days", maintain_order=True).first()
best_per_half_life.pivot(on="test_year", index="half_life_days", values="eta").sort("half_life_days")"""),
]

CELLS_BY_NOTEBOOK = {
    "explore_entsoe.ipynb": EXPLORE_CELLS,
    "combiner_diagnostics.ipynb": DIAGNOSTIC_CELLS,
}


def build_notebook(cells):
    """Return a notebook object built from a list of (cell type, source) pairs."""
    notebook = nbformat.v4.new_notebook()
    for cell_type, source in cells:
        if cell_type == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(source))
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return notebook


def main():
    output_path = Path(sys.argv[1])
    if output_path.name not in CELLS_BY_NOTEBOOK:
        raise SystemExit(f"unknown notebook {output_path.name}; expected one of {list(CELLS_BY_NOTEBOOK)}")
    nbformat.write(build_notebook(CELLS_BY_NOTEBOOK[output_path.name]), output_path)


if __name__ == "__main__":
    main()
