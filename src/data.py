"""Load the clean tables for development work, cut off before the 2024 holdout."""

from datetime import date
from pathlib import Path

import polars as pl

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
HOLDOUT_START = date(2024, 1, 1)


def load_dev_hourly():
    """Return the clean hourly table for dates before the holdout."""
    hourly = pl.read_parquet(CLEAN_DIR / "de_hourly.parquet")
    return hourly.filter(pl.col("date") < HOLDOUT_START)


def load_all_hourly():
    """Return the clean hourly table including the holdout (only for the single run of the frozen config)."""
    return pl.read_parquet(CLEAN_DIR / "de_hourly.parquet")


def load_dev_day_flags():
    """Return the outlier-day flags for dates before the holdout."""
    flags = pl.read_parquet(CLEAN_DIR / "day_flags.parquet")
    return flags.filter(pl.col("date") < HOLDOUT_START)
