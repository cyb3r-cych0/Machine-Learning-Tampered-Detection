#!/usr/bin/env python3
"""
fetch_country_pm25.py

Research-grade OpenAQ v3 PM2.5 collector
with:

- country-level ingestion
- streaming page processing
- immediate checkpointing
- graceful CTRL+C handling
- rate-limit awareness
- fault tolerance
- raw JSON preservation
- resumable-safe architecture

ARCHITECTURE
------------
Country
    ↓
Stream location pages
    ↓
Process locations immediately
    ↓
Discover PM2.5 sensors
    ↓
Fetch measurements
    ↓
Immediate checkpoint save

RECOMMENDED USAGE
-----------------

export OPENAQ_API_KEY="your_api_key"

python fetch_country_pm25.py \
    --country ET \
    --start-date 2025-01-01 \
    --end-date 2025-01-07
"""

import os
import json
import time
import argparse
import logging

from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "OPENAQ_API_KEY environment variable not found."
    )

# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://api.openaq.org/v3"

RAW_DIR = Path("data/raw/openaq")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "country_pm25_fetch.log"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
)

console.setFormatter(formatter)

logging.getLogger("").addHandler(console)

# ============================================================
# API SESSION
# ============================================================

HEADERS = {
    "X-API-Key": API_KEY
}

session = requests.Session()
session.headers.update(HEADERS)

# ============================================================
# RATE LIMIT TRACKER
# ============================================================

RATE_LIMIT = None
RATE_REMAINING = None
RATE_RESET = None

# ============================================================
# REQUEST HANDLER
# ============================================================

def request_with_retries(
        url,
        params=None,
        retries=3,
        backoff=2
):
    """
    Robust request handler with:
        - retry logic
        - rate-limit awareness
        - graceful pausing
        - fault tolerance
    """

    global RATE_LIMIT
    global RATE_REMAINING
    global RATE_RESET

    for attempt in range(1, retries + 1):

        try:

            response = session.get(
                url,
                params=params,
                timeout=30
            )

            # ====================================================
            # RATE LIMIT HEADERS
            # ====================================================

            RATE_LIMIT = int(
                response.headers.get(
                    "x-ratelimit-limit",
                    0
                )
            )

            RATE_REMAINING = int(
                response.headers.get(
                    "x-ratelimit-remaining",
                    0
                )
            )

            RATE_RESET = int(
                response.headers.get(
                    "x-ratelimit-reset",
                    60
                )
            )

            logging.info(
                f"Rate Limit: "
                f"{RATE_REMAINING}/{RATE_LIMIT} remaining"
            )

            # ====================================================
            # SUCCESS
            # ====================================================

            if response.status_code == 200:
                return response.json()

            # ====================================================
            # RATE LIMIT EXCEEDED
            # ====================================================

            if response.status_code == 429:

                logging.warning(
                    "API rate limit exceeded."
                )

                print("\n")
                print("=" * 60)
                print("OPENAQ API RATE LIMIT EXCEEDED")
                print("=" * 60)

                print(
                    f"Requests Remaining : "
                    f"{RATE_REMAINING}"
                )

                print(
                    f"Reset Timer (sec)  : "
                    f"{RATE_RESET}"
                )

                print("=" * 60)

                print("\nOptions:")
                print("1 - Wait and continue")
                print("2 - Exit safely")

                user_choice = input(
                    "\nEnter choice: "
                ).strip()

                # ================================================
                # WAIT + CONTINUE
                # ================================================

                if user_choice == "1":

                    wait_time = RATE_RESET + 1

                    print(
                        f"\nWaiting "
                        f"{wait_time} seconds "
                        f"for rate limit reset...\n"
                    )

                    logging.warning(
                        f"Sleeping for "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                # ================================================
                # SAFE EXIT
                # ================================================

                else:

                    logging.warning(
                        "User terminated collection "
                        "after rate limit hit."
                    )

                    return None

            # ====================================================
            # OTHER HTTP ERRORS
            # ====================================================

            logging.warning(
                f"HTTP {response.status_code} "
                f"| URL: {url}"
            )

        except requests.RequestException as e:

            logging.warning(
                f"Request failed: {e}"
            )

        # ========================================================
        # RETRY BACKOFF
        # ========================================================

        sleep_time = backoff ** attempt

        logging.info(
            f"Retrying in {sleep_time}s..."
        )

        time.sleep(sleep_time)

    return None


def get_country_id(country_code):
    """
    Resolve ISO country code (e.g. ET)
    to OpenAQ country ID.
    """

    url = f"{BASE_URL}/countries"

    page = 1

    while True:

        data = request_with_retries(
            url,
            params={
                "limit": 100,
                "page": page
            }
        )

        if not data:
            break

        results = data.get("results", [])

        if not results:
            break

        for country in results:

            if (
                country.get("code", "").upper()
                == country_code.upper()
            ):

                logging.info(
                    f"Resolved {country_code} "
                    f"to country_id="
                    f"{country['id']}"
                )

                return country["id"]

        page += 1

    raise ValueError(
        f"Country code not found: "
        f"{country_code}"
    )

# ============================================================
# STREAM COUNTRY LOCATIONS
# ============================================================

def stream_country_locations(country_id):
    """
    Stream locations page-by-page
    instead of loading all into memory.
    """

    page = 1

    while True:

        logging.info(
            f"Fetching location page {page}"
        )

        url = f"{BASE_URL}/locations"

        params = {
            "countries_id": country_id,
            "limit": 100,
            "page": page
        }

        data = request_with_retries(
            url,
            params=params
        )

        print("\nREQUEST PARAMS:")
        print(params)

        print("\nMETA:")
        print(data.get("meta", {}))

        if data is None:

            logging.warning(
                "Location request returned None."
            )

            break

        results = data.get("results", [])


        # ====================================================
        # COUNTRY FILTER VALIDATION
        # ====================================================

        if page == 1 and results:

            print("\nCOUNTRY FILTER VALIDATION")
            print("=" * 50)

            for r in results[:5]:
                country = (
                    r.get("country", {})
                    .get("name")
                )

                print(
                    f"Location ID: {r.get('id')}"
                )

                print(
                    f"Location   : {r.get('name')}"
                )

                print(
                    f"Country    : {country}"
                )

                print("-" * 50)

                logging.info(
                    f"First returned country: "
                    f"{country}"
                )

            # break
        # location_id = location.get("id")
        #
        # print(
        #     f"\nPROCESSING LOCATION: "
        #     f"{location_id} | "
        #     f"{location.get('name')}"
        # )

        # ====================================================

        if not results:

            logging.info(
                "No more location pages."
            )

            break

        logging.info(
            f"Fetched page {page} "
            f"({len(results)} locations)"
        )

        logging.info(
            f"Rate Status -> "
            f"{RATE_REMAINING}/{RATE_LIMIT} remaining"
        )

        # ====================================================
        # STREAM IMMEDIATELY
        # ====================================================

        for location in results:
            yield location

        page += 1

        time.sleep(1)

# ============================================================
# LOCATION METADATA
# ============================================================

def fetch_location_metadata(location_id):

    url = f"{BASE_URL}/locations/{location_id}"

    logging.info(
        f"Fetching location metadata: "
        f"{location_id}"
    )

    return request_with_retries(url)

# ============================================================
# SENSOR DISCOVERY
# ============================================================

def extract_pm25_sensors(location_data):

    sensors = []

    print(
        f"PM25 Sensors Found: "
        f"{len(sensors)}"
    )

    results = location_data.get("results", [])

    if not results:
        return sensors

    location = results[0]

    for sensor in location.get("sensors", []):

        parameter = sensor.get("parameter", {})

        parameter_name = parameter.get(
            "name",
            ""
        ).lower()

        sensor_name = sensor.get(
            "name",
            ""
        ).lower()

        if (
            parameter_name == "pm25"
            or "pm25" in sensor_name
        ):

            sensors.append({

                "sensor_id":
                    sensor.get("id"),

                "sensor_name":
                    sensor.get("name"),

                "parameter":
                    parameter_name,

                "units":
                    parameter.get("units")
            })

    return sensors

# ============================================================
# FETCH SENSOR MEASUREMENTS
# ============================================================

def fetch_measurements(
        sensor_id,
        start_date,
        end_date,
        limit=1000
):

    logging.info(
        f"Fetching measurements "
        f"for sensor {sensor_id}"
    )

    all_results = []

    page = 1

    while True:

        url = (
            f"{BASE_URL}/sensors/"
            f"{sensor_id}/measurements"
        )

        params = {

            "date_from":
                f"{start_date}T00:00:00Z",

            "date_to":
                f"{end_date}T23:59:59Z",

            "limit":
                limit,

            "page":
                page
        }

        data = request_with_retries(
            url,
            params=params
        )

        if data is None:
            break

        results = data.get("results", [])

        if not results:
            break

        all_results.extend(results)

        logging.info(
            f"Sensor {sensor_id} | "
            f"Page {page} | "
            f"Rows: {len(results)}"
        )

        page += 1

        time.sleep(1)

    logging.info(
        f"Total measurements collected: "
        f"{len(all_results)}"
    )

    return all_results

# ============================================================
# SAVE RAW JSON
# ============================================================

def save_raw_json(data, filename):

    output_path = RAW_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logging.info(
        f"Saved raw JSON: {output_path}"
    )

# ============================================================
# NORMALIZE DATAFRAME
# ============================================================

def normalize_measurements(
        measurements,
        metadata,
        sensor_info
):

    records = []

    location = metadata["results"][0]

    coordinates = location.get(
        "coordinates",
        {}
    )

    for row in measurements:

        records.append({

            "timestamp_utc":
                row.get("period", {})
                   .get("datetimeFrom", {})
                   .get("utc"),

            "timestamp_local":
                row.get("period", {})
                   .get("datetimeFrom", {})
                   .get("local"),

            "value":
                row.get("value"),

            "parameter":
                sensor_info["parameter"],

            "units":
                sensor_info["units"],

            "sensor_id":
                sensor_info["sensor_id"],

            "sensor_name":
                sensor_info["sensor_name"],

            "location_id":
                location.get("id"),

            "location_name":
                location.get("name"),

            "country":
                location.get("country", {})
                        .get("name"),

            "provider":
                location.get("provider", {})
                        .get("name"),

            "owner":
                location.get("owner", {})
                        .get("name"),

            "latitude":
                coordinates.get("latitude"),

            "longitude":
                coordinates.get("longitude")
        })

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # ========================================================
    # CLEANING
    # ========================================================

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp_utc", "value"]
    )

    df = df[df["value"] >= 0]

    df = df.drop_duplicates()

    df = df.sort_values("timestamp_utc")

    return df

# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
        all_dataframes,
        country_code
):

    if not all_dataframes:

        logging.warning(
            "No dataframe data available "
            "for checkpoint save."
        )

        return

    try:

        checkpoint_df = pd.concat(
            all_dataframes,
            ignore_index=True
        )

        checkpoint_df = (
            checkpoint_df
            .drop_duplicates()
            .sort_values("timestamp_utc")
        )

        checkpoint_filename = (
            f"{country_code.lower()}_"
            f"pm25_checkpoint.csv"
        )

        checkpoint_path = (
            PROCESSED_DIR /
            checkpoint_filename
        )

        checkpoint_df.to_csv(
            checkpoint_path,
            index=False
        )

        logging.info(
            f"Checkpoint saved successfully: "
            f"{checkpoint_path}"
        )

    except Exception as e:

        logging.error(
            f"Checkpoint save failed: {e}"
        )

# ============================================================
# FINAL EXPORT
# ============================================================

def export_csv(df, filename):

    output_path = PROCESSED_DIR / filename

    df.to_csv(output_path, index=False)

    logging.info(
        f"Saved processed CSV: {output_path}"
    )

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Country-level OpenAQ PM2.5 collector"
    )

    parser.add_argument(
        "--country",
        default=86,
        help="Country ID"
    )

    parser.add_argument(
        "--start-date",
        default="2016-01-01",
        help="Start date YYYY-MM-DD"
    )

    parser.add_argument(
        "--end-date",
        default="2016-12-31",
        help="End date YYYY-MM-DD"
    )

    args = parser.parse_args()

    country_code = args.country.upper()

    start_date = args.start_date
    end_date = args.end_date

    logging.info("===================================")
    logging.info("OpenAQ Country PM2.5 Collector")
    logging.info("===================================")

    logging.info(
        f"Country     : {country_code}"
    )

    logging.info(
        f"Start Date  : {start_date}"
    )

    logging.info(
        f"End Date    : {end_date}"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    all_dataframes = []

    processed_locations = 0
    processed_sensors = 0
    total_rows_collected = 0

    # ========================================================
    # STREAMING EXECUTION
    # ========================================================

    try:

        country_id = get_country_id(
            country_code
        )

        logging.info(
            f"Using OpenAQ country ID "
            f"{country_id}"
        )

        for location in stream_country_locations(
            country_id
        ):

            location_id = location.get("id")

            processed_locations += 1

            logging.info(
                f"Processing location "
                f"{location_id}"
            )

            # ------------------------------------------------
            # FETCH LOCATION METADATA
            # ------------------------------------------------

            metadata = fetch_location_metadata(
                location_id
            )

            if metadata is None:
                continue

            # ------------------------------------------------
            # SENSOR DISCOVERY
            # ------------------------------------------------

            sensors = extract_pm25_sensors(
                metadata
            )

            if not sensors:

                logging.info(
                    f"No PM2.5 sensors found "
                    f"for location {location_id}"
                )

                continue

            logging.info(
                f"Discovered {len(sensors)} "
                f"PM2.5 sensor(s)"
            )

            # ------------------------------------------------
            # SENSOR LOOP
            # ------------------------------------------------

            for sensor in sensors:

                sensor_id = sensor["sensor_id"]

                processed_sensors += 1

                measurements = fetch_measurements(
                    sensor_id,
                    start_date,
                    end_date
                )

                if not measurements:

                    logging.warning(
                        f"No measurements returned "
                        f"for sensor {sensor_id}"
                    )

                    continue

                # --------------------------------------------
                # SAVE RAW JSON
                # --------------------------------------------

                raw_filename = (
                    f"{country_code}_"
                    f"sensor_{sensor_id}_"
                    f"{timestamp}.json"
                )

                save_raw_json(
                    measurements,
                    raw_filename
                )

                # --------------------------------------------
                # NORMALIZE
                # --------------------------------------------

                df = normalize_measurements(
                    measurements,
                    metadata,
                    sensor
                )

                if df.empty:

                    logging.warning(
                        f"Empty dataframe for "
                        f"sensor {sensor_id}"
                    )

                    continue

                rows_added = len(df)

                total_rows_collected += rows_added

                all_dataframes.append(df)

                # --------------------------------------------
                # IMMEDIATE CHECKPOINT SAVE
                # --------------------------------------------

                save_checkpoint(
                    all_dataframes,
                    country_code
                )

                logging.info(
                    f"Rows added: {rows_added}"
                )

                logging.info(
                    f"Total rows collected: "
                    f"{total_rows_collected}"
                )

                logging.info(
                    f"Processed locations: "
                    f"{processed_locations}"
                )

                logging.info(
                    f"Processed sensors: "
                    f"{processed_sensors}"
                )

    # ========================================================
    # CTRL+C SAFE EXIT
    # ========================================================

    except KeyboardInterrupt:

        logging.warning(
            "Keyboard interrupt detected."
        )

        print("\n")
        print("=" * 60)
        print("INTERRUPTION DETECTED")
        print("=" * 60)

        print(
            "Saving checkpoint before exit..."
        )

        save_checkpoint(
            all_dataframes,
            country_code
        )

        print("\nSafe exit completed.")

        return

    # ========================================================
    # FINAL EXPORT
    # ========================================================

    if not all_dataframes:

        logging.warning(
            "No PM2.5 data collected."
        )

        return

    final_df = pd.concat(
        all_dataframes,
        ignore_index=True
    )

    final_df = (
        final_df
        .drop_duplicates()
        .sort_values("timestamp_utc")
    )

    csv_filename = (
        f"{country_code.lower()}_pm25_"
        f"{start_date}_to_{end_date}.csv"
    )

    export_csv(
        final_df,
        csv_filename
    )

    logging.info("===================================")
    logging.info("Collection completed successfully")
    logging.info("===================================")

    print("\n")
    print("=" * 60)
    print("COLLECTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Total rows collected: "
        f"{len(final_df)}"
    )

    print(
        f"Processed locations: "
        f"{processed_locations}"
    )

    print(
        f"Processed sensors: "
        f"{processed_sensors}"
    )

    print("=" * 60)

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()