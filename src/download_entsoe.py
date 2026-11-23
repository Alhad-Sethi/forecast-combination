"""Download day-ahead prices, load and wind/solar forecasts from ENTSO-E as yearly parquet files (--market DE or FR).
Finished years are skipped on rerun; the current year is always re-downloaded."""

import os
import sys
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

FIRST_YEAR = 2015
TIMEZONE = "Europe/Brussels"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "entsoe"

DEFAULT_MARKET = "DE"
# Per market: ENTSO-E area codes and the raw folder. Germany's prices came from the German-Austrian zone
# before the zone split on 2018-10-01; Germany predates the per-market layout, so its files sit in RAW_DIR.
MARKETS = {
    "DE": {
        "raw_dir": RAW_DIR,
        "price_area_before_split": "DE_AT_LU",
        "price_area": "DE_LU",
        "zone_split": pd.Timestamp("2018-10-01", tz=TIMEZONE),
        "forecast_area": "DE",
    },
    "FR": {
        "raw_dir": RAW_DIR / "FR",
        "price_area": "FR",
        "forecast_area": "FR",
    },
}

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


def selected_market():
    """Return the market code given after --market on the command line (Germany by default)."""
    if "--market" not in sys.argv:
        return DEFAULT_MARKET
    market = sys.argv[sys.argv.index("--market") + 1]
    if market not in MARKETS:
        raise SystemExit(f"--market {market}: unknown; known markets are {list(MARKETS)}")
    return market


def fetch_prices(client, config, start, end):
    """Return day-ahead prices, querying the right bidding zone on each side of a zone split if the market had one."""
    zone_split = config.get("zone_split")
    if zone_split is None:
        prices = client.query_day_ahead_prices(config["price_area"], start=start, end=end)
        return prices.to_frame(name="price_eur_mwh")
    pieces = []
    if start < zone_split:
        piece_end = min(end, zone_split)
        pieces.append(client.query_day_ahead_prices(config["price_area_before_split"], start=start, end=piece_end))
    if end > zone_split:
        piece_start = max(start, zone_split)
        pieces.append(client.query_day_ahead_prices(config["price_area"], start=piece_start, end=end))
    prices = pd.concat(pieces)
    return prices.to_frame(name="price_eur_mwh")


def fetch_load_forecast(client, config, start, end):
    """Return the day-ahead total load forecast."""
    return client.query_load_forecast(config["forecast_area"], start=start, end=end)


def fetch_renewables_forecast(client, config, start, end):
    """Return the day-ahead wind and solar generation forecasts."""
    return client.query_wind_and_solar_forecast(config["forecast_area"], start=start, end=end)


SERIES_FETCHERS = {
    "day_ahead_prices": fetch_prices,
    "load_forecast": fetch_load_forecast,
    "wind_solar_forecast": fetch_renewables_forecast,
}


def raw_path(config, series_name, year):
    """Return the parquet path for one market, series and year."""
    return config["raw_dir"] / series_name / f"{year}.parquet"


def save_raw(frame, path):
    """Write `frame` to `path` via a temporary file, so an interrupted run leaves no half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    frame.to_parquet(temporary_path)
    temporary_path.replace(path)


def download_year(client, config, series_name, year, current_year):
    """Download one series for one year unless a finished file already exists."""
    path = raw_path(config, series_name, year)
    if path.exists() and year < current_year:
        print(f"skip  {series_name} {year} (exists)")
        return

    start, end = year_bounds(year)
    fetch = SERIES_FETCHERS[series_name]
    try:
        frame = fetch(client, config, start, end)
    except NoMatchingDataError:
        print(f"empty {series_name} {year} (no data returned)")
        return

    save_raw(frame, path)
    print(f"saved {series_name} {year}: {len(frame)} rows")


def main():
    market = selected_market()
    config = MARKETS[market]
    client = build_client()
    current_year = pd.Timestamp.now(tz=TIMEZONE).year

    print(f"downloading {market}")
    for series_name in SERIES_FETCHERS:
        for year in range(FIRST_YEAR, current_year + 1):
            download_year(client, config, series_name, year, current_year)


if __name__ == "__main__":
    main()
