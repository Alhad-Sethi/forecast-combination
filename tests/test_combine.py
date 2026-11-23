"""Hand-worked checks of the combiners on tiny synthetic inputs. Run: python tests/test_combine.py
Expected numbers below were derived by hand from the update equations, not by running the code."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from combine import CombinerInputs, equal_weights, run_boa, run_ewa, run_tree  # noqa: E402

HOURS_PER_DAY = 24
TOLERANCE = 1e-4


def constant_inputs(expert_values, actual_values, can_update=None):
    """Build inputs where every hour of a day has the same value; expert_values is (days, experts)."""
    expert_values = np.asarray(expert_values, dtype=float)
    num_days, num_experts = expert_values.shape
    forecasts = np.repeat(expert_values[:, :, np.newaxis], HOURS_PER_DAY, axis=2)
    actual = np.repeat(np.asarray(actual_values, dtype=float)[:, np.newaxis], HOURS_PER_DAY, axis=1)
    if can_update is None:
        can_update = np.ones(num_days, dtype=bool)
    expert_names = []
    for expert in range(num_experts):
        expert_names.append(f"expert_{expert}")
    return CombinerInputs(
        dates=list(range(num_days)),
        weekdays=np.ones(num_days, dtype=int),
        expert_names=expert_names,
        forecasts=forecasts,
        actual=actual,
        scale=np.ones(num_days),
        can_update=np.asarray(can_update),
    )


def assert_close(actual, expected, label):
    """Fail with a readable message if two numbers differ by more than TOLERANCE."""
    if abs(actual - expected) > TOLERANCE:
        raise AssertionError(f"{label}: got {actual:.6f}, expected {expected:.6f}")


def test_ewa_with_zero_eta_is_equal_weights():
    inputs = constant_inputs([[1, 3], [2, 6], [5, 1]], [0, 0, 0])
    combined, _ = run_ewa(inputs, 0, 3, eta=0.0, half_life_days=7)
    if not np.allclose(combined, equal_weights(inputs, 0, 3)):
        raise AssertionError("EWA with eta = 0 must equal 1/N")


def test_ewa_hand_worked():
    # Experts forecast 1 and 3, the price is 0, scale 1: daily losses are 1 and 3.
    # Half-life 1 day -> discount 0.5. eta = 0.1.
    # Day 1: uniform. After day 1: L = (1, 3) -> day 2 weight on expert 0 = 1 / (1 + e^-0.2) = 0.549834.
    # After day 2: L = 0.5 * (1, 3) + (1, 3) = (1.5, 4.5) -> day 3 weight = 1 / (1 + e^-0.3) = 0.574443.
    inputs = constant_inputs([[1, 3], [1, 3], [1, 3]], [0, 0, 0])
    _, weights = run_ewa(inputs, 0, 3, eta=0.1, half_life_days=1)
    assert_close(weights[0, 0], 0.5, "EWA day 1")
    assert_close(weights[1, 0], 0.549834, "EWA day 2")
    assert_close(weights[2, 0], 0.574443, "EWA day 3")


def test_ewa_skips_outlier_days():
    inputs = constant_inputs([[1, 3], [1, 3], [1, 3]], [0, 0, 0], can_update=[True, False, True])
    _, weights = run_ewa(inputs, 0, 3, eta=0.1, half_life_days=1)
    assert_close(weights[2, 0], weights[1, 0], "EWA weights must not move after a skipped day")


def test_boa_hand_worked():
    # Wintenberger (2017), Figure 2. Experts forecast 1 and 3, the price is 0, scale 1, prior (0.5, 0.5).
    # Day 1: weights = prior (rates start at 0); combined 2, gradient sign(2 - 0) = 1, l = f - combined = (-1, +1).
    #   max |l| = (1, 1) -> k = 0 -> E = 2^1 = (2, 2). No range break (previous rates are 0).
    #   L = 0.5 * l = (-0.5, 0.5), V = (1, 1)
    #   eta = min(1 / 2, sqrt(ln 2 / 1) = 0.832555) = 0.5 for both
    #   day 2 weight on expert 0 = 1 / (1 + e^(-0.5 * (0.5 + 0.5))) = 1 / (1 + e^-0.5) = 0.622459
    # Day 2: combined = 0.622459 + 0.377541 * 3 = 1.755082, gradient 1, l = (-0.755082, 1.244918)
    #   max |l| = (1, 1.244918) -> E = (2, 4)
    #   range break: eta * l = (-0.377541, 0.622459) -> only expert 1 (> 0.5), so it pays E = 4
    #   L0 = -0.5 + 0.5 * (-0.755082 * (1 - 0.377541)) = -0.5 - 0.235004 = -0.735004
    #   L1 = 0.5 + 0.5 * (1.244918 * 1.622459 + 4) = 0.5 + 0.5 * 6.019829 = 3.509914
    #   V = (1.570149, 2.549821)
    #   eta_0 = min(1 / 2, sqrt(0.693147 / 1.570149) = 0.664420) = 0.5
    #   eta_1 = min(1 / 4, sqrt(0.693147 / 2.549821) = 0.521385) = 0.25
    #   log weights (prior cancels): ln 0.5 + 0.5 * 0.735004 = -0.325645
    #                                ln 0.25 - 0.25 * 3.509914 = -2.263773
    #   day 3 weight on expert 0 = 1 / (1 + e^(-1.938128)) = 0.874146
    inputs = constant_inputs([[1, 3], [1, 3], [1, 3]], [0, 0, 0])
    combined, weights = run_boa(inputs, 0, 3)
    assert_close(weights[0, 0], 0.5, "BOA day 1")
    assert_close(weights[1, 0], 0.622459, "BOA day 2")
    assert_close(combined[1, 0], 1.755082, "BOA day 2 combined forecast")
    assert_close(weights[2, 0], 0.874146, "BOA day 3")


def test_boa_learns_a_perfect_expert():
    rng = np.random.default_rng(0)
    num_days = 300
    actual = rng.normal(50, 10, num_days)
    noisy = actual + rng.normal(0, 5, num_days)
    biased = actual + 8
    inputs = constant_inputs(np.column_stack([actual, noisy, biased]), actual)
    _, weights = run_boa(inputs, 0, num_days)
    if weights[-1, 0] < 0.9:
        raise AssertionError(f"BOA should put most weight on the perfect expert, got {weights[-1, 0]:.3f}")
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise AssertionError("BOA weights must sum to 1 every day")


def test_tree_is_equal_weights_when_nothing_to_learn():
    rng = np.random.default_rng(1)
    num_days = 120
    experts = rng.normal(50, 10, (num_days, 3))
    inputs = constant_inputs(experts, experts.mean(axis=1))
    combined = run_tree(inputs, 60, num_days, max_depth=2, min_leaf_hours=50)
    if not np.allclose(combined, equal_weights(inputs, 60, num_days)):
        raise AssertionError("tree must reproduce 1/N when the actual equals 1/N")


def main():
    tests = [
        test_ewa_with_zero_eta_is_equal_weights,
        test_ewa_hand_worked,
        test_ewa_skips_outlier_days,
        test_boa_hand_worked,
        test_boa_learns_a_perfect_expert,
        test_tree_is_equal_weights_when_nothing_to_learn,
    ]
    for test in tests:
        test()
        print(f"passed  {test.__name__}")


if __name__ == "__main__":
    main()
