"""Combine expert forecasts: equal weights (1/N), EWA, gradient BOA and a shallow tree that corrects 1/N.
Online methods update daily once prices are known; hyperparameters are chosen in walk_forward.py."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeRegressor

from experts import naive_forecasts
from features import HOURS_PER_DAY, load_day_data

PROJECT_DIR = Path(__file__).resolve().parent.parent
FORECASTS_PATH = PROJECT_DIR / "data" / "experts" / "forecasts.parquet"
HOLDOUT_START = date(2024, 1, 1)

SCALE_WINDOW_DAYS = 28
# The loss scale needs at least a week of naive errors before it is trusted.
SCALE_MIN_DAYS = 7

TREE_HISTORY_DAYS = 3 * 365
TREE_REFIT_DAYS = 7
TREE_MIN_TRAINING_DAYS = 28
RANDOM_SEED = 0


class CombinerInputs(NamedTuple):
    """Expert forecasts and everything the combiners need, aligned by day."""

    dates: list
    weekdays: np.ndarray
    expert_names: list
    forecasts: np.ndarray
    actual: np.ndarray
    scale: np.ndarray
    can_update: np.ndarray


# ---------- inputs ----------


def naive_loss_scale(data):
    """Return, per day, the seasonal naive's mean daily MAE over the previous 28 days."""
    first_index = int(np.argmax(data.has_features))
    naive = naive_forecasts(data, first_index, len(data.dates))
    # A plain mean leaves days with any missing hour as NaN; the rolling mean then skips them.
    daily_naive_mae = np.mean(np.abs(naive - data.targets), axis=1)
    errors = pl.Series("naive_mae", daily_naive_mae).fill_nan(None)
    trailing = errors.rolling_mean(window_size=SCALE_WINDOW_DAYS, min_samples=SCALE_MIN_DAYS).shift(1)
    return trailing.fill_null(np.nan).to_numpy()


def expert_forecast_array(first_date, end_date):
    """Return expert names and a (days, experts, 24) array of forecasts for dates in [first_date, end_date)."""
    forecasts = pl.read_parquet(FORECASTS_PATH)
    forecasts = forecasts.filter(pl.col("date").is_between(first_date, end_date, closed="left"))
    forecasts = forecasts.sort("date", "expert", "hour")
    expert_names = forecasts["expert"].unique().sort().to_list()
    num_days = forecasts["date"].n_unique()
    values = forecasts["forecast"].to_numpy()
    return expert_names, values.reshape(num_days, len(expert_names), HOURS_PER_DAY)


def load_combiner_inputs(include_holdout=False):
    """Return combiner inputs from the first expert forecast up to the holdout (or to the end)."""
    data = load_day_data()
    scale = naive_loss_scale(data)
    first_date = pl.read_parquet(FORECASTS_PATH)["date"].min()
    if include_holdout:
        end_date = date.max
    else:
        end_date = HOLDOUT_START
    expert_names, forecasts = expert_forecast_array(first_date, end_date)

    first_index = data.dates.index(first_date)
    days = np.arange(first_index, first_index + forecasts.shape[0])
    can_update = data.has_target[days] & ~data.is_outlier[days] & ~np.isnan(scale[days])
    return CombinerInputs(
        dates=[data.dates[day] for day in days],
        weekdays=data.weekdays[days],
        expert_names=expert_names,
        forecasts=forecasts,
        actual=data.targets[days],
        scale=scale[days],
        can_update=can_update,
    )


# ---------- 1/N ----------


def equal_weights(inputs, start, end):
    """Return the plain average of all experts for days [start, end)."""
    return inputs.forecasts[start:end].mean(axis=1)


# ---------- EWA ----------


def scaled_expert_losses(inputs, day, per_hour):
    """Return each expert's absolute error divided by the day's scale: (experts,) averaged, or (24, experts)."""
    errors = np.abs(inputs.forecasts[day] - inputs.actual[day]) / inputs.scale[day]
    if per_hour:
        return errors.T
    return errors.mean(axis=1)


def exponential_weights(cumulative_losses, eta):
    """Return weights proportional to exp(-eta * loss) over the last axis; uniform when eta is 0."""
    logits = -eta * cumulative_losses
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=-1, keepdims=True)


def discount_factor(half_life_days):
    """Return the daily discount for a half-life in days; 1 (no discounting) for an infinite half-life."""
    if np.isinf(half_life_days):
        return 1.0
    return 0.5 ** (1.0 / half_life_days)


def combine_with_weights(day_forecasts, weights):
    """Return the weighted average of (experts, 24) forecasts; weights are (experts,) or per hour (24, experts)."""
    if weights.ndim == 1:
        return weights @ day_forecasts
    return (weights * day_forecasts.T).sum(axis=1)


def run_ewa(inputs, start, end, eta, half_life_days, per_hour=False):
    """Run EWA from uniform weights over days [start, end); return combined forecasts and daily weights."""
    num_experts = len(inputs.expert_names)
    if per_hour:
        cumulative_losses = np.zeros((HOURS_PER_DAY, num_experts))
    else:
        cumulative_losses = np.zeros(num_experts)
    discount = discount_factor(half_life_days)

    combined = np.empty((end - start, HOURS_PER_DAY))
    weight_history = np.empty((end - start,) + cumulative_losses.shape)
    for offset, day in enumerate(range(start, end)):
        weights = exponential_weights(cumulative_losses, eta)
        weight_history[offset] = weights
        combined[offset] = combine_with_weights(inputs.forecasts[day], weights)
        if inputs.can_update[day]:
            cumulative_losses = discount * cumulative_losses + scaled_expert_losses(inputs, day, per_hour)
    return combined, weight_history


# ---------- gradient BOA (Wintenberger 2017, adaptive learning rates) ----------


@dataclass
class BoaState:
    """Per-expert running quantities of adaptive BOA."""

    max_abs_loss: np.ndarray
    sum_squared_loss: np.ndarray
    cumulative_loss: np.ndarray
    learning_rates: np.ndarray


def initial_boa_state(num_experts):
    """Return the state before any data: all zeros, so the first day's second-order term is zero."""
    zeros = np.zeros(num_experts)
    return BoaState(zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())


def linearized_excess_losses(inputs, day, combined):
    """Return each expert's excess loss over the combination via the gradient trick, averaged over hours."""
    # Gradient of |combined - actual| / scale with respect to the combined forecast, hour by hour.
    gradients = np.sign(combined - inputs.actual[day]) / inputs.scale[day]
    return (inputs.forecasts[day] - combined) @ gradients / gradients.size


def boa_update(state, excess_losses, prior):
    """Return the state after one day: accumulate l + eta*l^2, then reset each expert's learning rate."""
    cumulative_loss = state.cumulative_loss + excess_losses + state.learning_rates * excess_losses**2
    max_abs_loss = np.maximum(state.max_abs_loss, np.abs(excess_losses))
    sum_squared_loss = state.sum_squared_loss + excess_losses**2

    learning_rates = np.zeros_like(prior)
    has_history = sum_squared_loss > 0
    rate_from_range = 1.0 / (2.0 * max_abs_loss[has_history])
    rate_from_variance = np.sqrt(np.log(1.0 / prior[has_history]) / sum_squared_loss[has_history])
    learning_rates[has_history] = np.minimum(rate_from_range, rate_from_variance)
    return BoaState(max_abs_loss, sum_squared_loss, cumulative_loss, learning_rates)


def boa_weights(state, prior):
    """Return weights proportional to eta * exp(-eta * cumulative loss) * prior; the prior until all rates are set."""
    if np.any(state.learning_rates == 0):
        return prior.copy()
    log_weights = np.log(state.learning_rates) - state.learning_rates * state.cumulative_loss + np.log(prior)
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def run_boa(inputs, start, end):
    """Run gradient BOA from uniform weights over days [start, end); return combined forecasts and daily weights."""
    num_experts = len(inputs.expert_names)
    prior = np.full(num_experts, 1.0 / num_experts)
    state = initial_boa_state(num_experts)

    combined = np.empty((end - start, HOURS_PER_DAY))
    weight_history = np.empty((end - start, num_experts))
    for offset, day in enumerate(range(start, end)):
        weights = boa_weights(state, prior)
        weight_history[offset] = weights
        combined[offset] = weights @ inputs.forecasts[day]
        if inputs.can_update[day]:
            excess_losses = linearized_excess_losses(inputs, day, combined[offset])
            state = boa_update(state, excess_losses, prior)
    return combined, weight_history


# ---------- shallow tree correcting 1/N ----------


def tree_inputs(inputs, days):
    """Return one row per (day, hour): each expert's deviation from 1/N in scaled units, hour, weekday."""
    baseline = inputs.forecasts[days].mean(axis=1)
    scale = inputs.scale[days][:, np.newaxis, np.newaxis]
    deviations = (inputs.forecasts[days] - baseline[:, np.newaxis, :]) / scale
    expert_columns = deviations.transpose(0, 2, 1).reshape(-1, len(inputs.expert_names))
    hours = np.tile(np.arange(HOURS_PER_DAY), len(days))
    weekdays = np.repeat(inputs.weekdays[days], HOURS_PER_DAY)
    return np.column_stack([expert_columns, hours, weekdays])


def tree_targets(inputs, days):
    """Return (actual - 1/N) / scale, one value per (day, hour)."""
    baseline = inputs.forecasts[days].mean(axis=1)
    return ((inputs.actual[days] - baseline) / inputs.scale[days][:, np.newaxis]).reshape(-1)


def run_tree(inputs, start, end, max_depth, min_leaf_hours):
    """Correct 1/N with a shallow tree refit weekly on the previous 3 years; plain 1/N until history exists."""
    combined = equal_weights(inputs, start, end)
    for block_start in range(start, end, TREE_REFIT_DAYS):
        candidates = np.arange(max(0, block_start - TREE_HISTORY_DAYS), block_start)
        history = candidates[inputs.can_update[candidates]]
        if len(history) < TREE_MIN_TRAINING_DAYS:
            continue
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf_hours, random_state=RANDOM_SEED)
        tree.fit(tree_inputs(inputs, history), tree_targets(inputs, history))

        block = np.arange(block_start, min(block_start + TREE_REFIT_DAYS, end))
        corrections = tree.predict(tree_inputs(inputs, block)).reshape(len(block), HOURS_PER_DAY)
        combined[block - start] += corrections * inputs.scale[block][:, np.newaxis]
    return combined
