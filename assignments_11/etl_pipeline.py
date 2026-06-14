# video link: https://youtu.be/DTfBegCamIE

from prefect import flow, task
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import json
from datetime import date
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from prefect.logging import get_run_logger


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())

ACCOUNT_URL = "https://fanyctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
MAX_RECORDS = 24  # process one day of hourly data

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


# Extract task
# Decorated with @task(retries=2, retry_delay_seconds=10)
# Calls the Open-Meteo API for 7 days of hourly temperature_2m and precipitation data for a city of your choosing
# Uses raise_for_status()
# Returns the raw JSON response as a dict
# Prints a confirmation message

@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    print(f"Extracted forecast data for ({latitude}, {longitude})")
    return response.json()

# Transform task
# Decorated with @task
# Reshapes the "hourly" parallel lists into individual per-hour records
# Classifies the first 24 records (one day) using the OpenAI API with this system prompt:
# You are classifying hourly weather conditions for outdoor running.
# Given a temperature in Celsius and a precipitation amount in mm,
# classify the conditions as exactly one of: good, marginal, or bad.
# Reply with that one word only -- no punctuation, no explanation.
# Falls back to "unknown" if the model returns an unexpected response
# Prints a progress message every 6 records
# Returns the list of enriched records
@task(retries=2, retry_delay_seconds=10)
def transform(data: dict, max_records: int):
    logger = get_run_logger()
    hourly = data["hourly"]
    
    records = []
    for i in range(min(max_records, len(hourly["time"]))):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })

    enriched = []

    for i, record in enumerate(records):
        user_msg = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )

        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"
        if raw_label == "unknown": logger.warning( "label is not in VALID LABELS" )
        enriched.append({**record, "conditions": label})

        if (i + 1) % 6 == 0:
            print(f"  Classified {i + 1}/{len(records)} records")

    print(f"Transform complete: {len(enriched)} records enriched")
    return enriched
# Load task
# Decorated with @task
# Uploads the enriched records as JSON to final/<today>/weather_etl.json in your pipeline-data container
# Uses overwrite=True
# Prints a confirmation with the blob path and byte count
@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    credential = DefaultAzureCredential()
    
    container = ContainerClient(
        ACCOUNT_URL,
        CONTAINER,
        credential=credential
    )

    payload = json.dumps(records).encode("utf-8")

    container.upload_blob(
        blob_path,
        payload,
        overwrite=True
    )
    logger.info(
        f"Loaded {len(records)} records to {blob_path}"
    )


# Flow
# Decorated with @flow(log_prints=True)
# Calls the three tasks in order
# Prints a completion message with the final blob path    
@flow(log_prints=True)
def etl_pipeline(
        latitude: float = 35.2271,
        longitude: float = -80.8431
    ):
        today = date.today().isoformat()

        blob_path = f"final/{today}/weather_etl.json"

        data = extract(latitude, longitude)

        enriched = transform(
            data,
            max_records=MAX_RECORDS
        )

        load(enriched, blob_path)

        print(f"Pipeline complete. Results at {blob_path}")

if __name__ == "__main__":
    etl_pipeline()
