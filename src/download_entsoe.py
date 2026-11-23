"""Download German day-ahead prices, load and wind/solar forecasts from ENTSO-E as yearly parquet files.
Finished years are skipped on rerun; the current year is always re-downloaded."""

import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

FIRST_YEAR = 2015
TIMEZONE = "Europe/Brussels"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "entsoe"

# The German-Austrian bidding zone split on this date; prices before it belong to DE_AT_LU.
ZONE_SPLIT = pd.Timestamp("2018-10-01", tz=TIMEZONE)
PRICE_AREA_BEFORE_SPLIT = "DE_AT_LU"
PRICE_AREA_AFTER_SPLIT = "DE_LU"
FORECAST_AREA = "DE"

# Day-ahead data for tomorrow is published today, so ask for a little past now.
DAYS_AHEAD_TO_REQUEST = 2

RETRY_COUNT = 3
RETRY_DELAY_SECONDS = 30


def build_client():
    """Return an ENTSO-E client authenticated with the ENTSOE_API_KEY environment variable."""
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise SystemExit("ENTSOE_API_KEY is not set. Run: set -a; source .env; set +a")
    return EntsoePandasClient(
        api_key=api_key,
        retry_count=RETRY_COUNT,
        retry_delay=RETRY_DELAY_SECONDS,
    )


def latest_available_time():
    """Return the latest timestamp worth requesting, a couple of days past today."""
    today = pd.Timestamp.now(tz=TIMEZONE).normalize()
    return today + pd.Timedelta(days=DAYS_AHEAD_TO_REQUEST)


def year_bounds(year):
    """Return the start and end timestamps of `year`, with the end capped at the latest available time."""
    start = pd.Timestamp(f"{year}-01-01", tz=TIMEZONE)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=TIMEZONE)
    end = min(end, latest_available_time())
    return start, end


def fetch_prices(client, start, end):
    """Return day-ahead prices for Germany, querying the right bidding zone on each side of the split."""
    pieces = []
    if start < ZONE_SPLIT:
        piece_end = min(end, ZONE_SPLIT)
        pieces.append(client.query_day_ahead_prices(PRICE_AREA_BEFORE_SPLIT, start=start, end=piece_end))
    if end > ZONE_SPLIT:
        piece_start = max(start, ZONE_SPLIT)
        pieces.append(client.query_day_ahead_prices(PRICE_AREA_AFTER_SPLIT, start=piece_start, end=end))
    prices = pd.concat(pieces)
    return prices.to_frame(name="price_eur_mwh")


def fetch_load_forecast(client, start, end):
    """Return the day-ahead total load forecast for Germany."""
    return client.query_load_forecast(FORECAST_AREA, start=start, end=end)


def fetch_renewables_forecast(client, start, end):
    """Return the day-ahead wind and solar generation forecasts for Germany."""
    return client.query_wind_and_solar_forecast(FORECAST_AREA, start=start, end=end)


SERIES_FETCHERS = {
    "day_ahead_prices": fetch_prices,
    "load_forecast": fetch_load_forecast,
    "wind_solar_forecast": fetch_renewables_forecast,
}


def raw_path(series_name, year):
    """Return the parquet path for one series and year."""
    return RAW_DIR / series_name / f"{year}.parquet"


def save_raw(frame, path):
    """Write `frame` to `path` via a temporary file, so an interrupted run leaves no half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    frame.to_parquet(temporary_path)
    temporary_path.replace(path)


def download_year(client, series_name, year, current_year):
    """Download one series for one year unless a finished file already exists."""
    path = raw_path(series_name, year)
    if path.exists() and year < current_year:
        print(f"skip  {series_name} {year} (exists)")
        return

    start, end = year_bounds(year)
    fetch = SERIES_FETCHERS[series_name]
    try:
        frame = fetch(client, start, end)
    except NoMatchingDataError:
        print(f"empty {series_name} {year} (no data returned)")
        return

    save_raw(frame, path)
    print(f"saved {series_name} {year}: {len(frame)} rows")


def main():
    client = build_client()
    current_year = pd.Timestamp.now(tz=TIMEZONE).year

    for series_name in SERIES_FETCHERS:
        for year in range(FIRST_YEAR, current_year + 1):
            download_year(client, series_name, year, current_year)


if __name__ == "__main__":
    main()
