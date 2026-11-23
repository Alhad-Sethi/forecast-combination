"""Judge experts on 2017-2019 only, the years before the first test fold: the agreed criterion for the expert pool.
Prints each expert's leave-one-out contribution to 1/N and 1/N's MAE for the full pool and every defined subset."""

import numpy as np
import polars as pl

from combine import load_combiner_inputs
from walk_forward import EXPERT_SUBSETS

PRE_TEST_YEARS = [2017, 2018, 2019]


def year_errors(inputs, year):
    """Return (experts, hours) forecast errors for one year, dropping hours without a price."""
    days = []
    for index, day in enumerate(inputs.dates):
        if day.year == year:
            days.append(index)
    errors = inputs.forecasts[days] - inputs.actual[days][:, np.newaxis, :]
    errors = errors.transpose(1, 0, 2).reshape(len(inputs.expert_names), -1)
    has_price = ~np.isnan(errors).any(axis=0)
    return errors[:, has_price]


def leave_one_out_contributions(inputs):
    """Return one row per (year, expert): 1/N's MAE without the expert minus with it (positive = it helps 1/N)."""
    rows = []
    for year in PRE_TEST_YEARS:
        errors = year_errors(inputs, year)
        # The error of an average forecast is the average of the experts' errors.
        equal_weight_mae = np.abs(errors.mean(axis=0)).mean()
        for index, name in enumerate(inputs.expert_names):
            others = np.delete(np.arange(len(inputs.expert_names)), index)
            mae_without = np.abs(errors[others].mean(axis=0)).mean()
            rows.append({"year": year, "expert": name, "contribution": float(mae_without - equal_weight_mae)})
    return pl.DataFrame(rows).pivot(on="year", index="expert", values="contribution")


def subset_equal_weight_mae(inputs):
    """Return 1/N's MAE per pre-test year for the full pool and every subset in EXPERT_SUBSETS."""
    pools = {"all experts": inputs.expert_names}
    pools.update(EXPERT_SUBSETS)
    rows = []
    for pool_name, names in pools.items():
        columns = []
        for name in names:
            columns.append(inputs.expert_names.index(name))
        row = {"pool": pool_name, "experts": len(columns)}
        for year in PRE_TEST_YEARS:
            errors = year_errors(inputs, year)
            row[str(year)] = float(np.abs(errors[columns].mean(axis=0)).mean())
        rows.append(row)
    return pl.DataFrame(rows)


def main():
    inputs = load_combiner_inputs()
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(leave_one_out_contributions(inputs).sort("2017", descending=True))
        print(subset_equal_weight_mae(inputs))


if __name__ == "__main__":
    main()
