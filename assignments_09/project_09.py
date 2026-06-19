# video link https://youtu.be/ickD5xGperU
print("----Part 2: Project -- Extract + Load Pipeline----")

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

ACCOUNT_URL = "https://fanyctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
OUTPUT_JSON = Path("outputs/weather_raw.json")

LATITUDE = 35.2271
LONGITUDE = -80.8431
FORECAST_DAYS = 7
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    "&hourly=temperature_2m,precipitation"
    f"&forecast_days={FORECAST_DAYS}"
)


def fetch_weather_data():
    response = requests.get(OPEN_METEO_URL)
    response.raise_for_status()
    return response.json()



# API response  →  Python dict  →  JSON string  →  UTF-8 bytes  →  upload to blob
#    (Extract)      (in memory)     (Serialize)     (ready for Load)
def serialize_weather_data(weather_data):
    # json.dumps(weather_data) — turns the dict into a JSON string, a portable text format you can save and read back later.
    # .encode("utf-8") — turns that string into bytes, which is what blob upload APIs expect (same idea as writing a .json file to disk).
    return json.dumps(weather_data).encode("utf-8")


def get_weather_blob_path():
    today = date.today().isoformat()
    return f"raw/{today}/weather.json"


def load_weather_data(weather_bytes):
    blob_path = get_weather_blob_path()

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(weather_bytes, overwrite=True)

    print(f"Uploaded {len(weather_bytes)} bytes to {CONTAINER}/{blob_path}")
    return blob_path


def verify_blobs():
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)

    for blob in container_client.list_blobs():
        print(f"{blob.name}  {blob.size} bytes")


def read_weather_data(blob_path):
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)
    blob_client = container_client.get_blob_client(blob_path)
    downloaded = blob_client.download_blob().readall()

    weather_data = json.loads(downloaded.decode("utf-8"))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_bytes(downloaded)

    hourly_df = pd.DataFrame(weather_data["hourly"])
    print(hourly_df.head())
    return hourly_df


if __name__ == "__main__":
    
    print("----Step 1: Extract-----")
    weather = fetch_weather_data()
    hourly = weather["hourly"]
    print(f"Retrieved {len(hourly['time'])} hourly records for Charlotte, NC")

    print("----Step 2: Serialize-----")
    weather_bytes = serialize_weather_data(weather)
    print(f"Serialized to {len(weather_bytes)} bytes")

    print("----Step 3: Load-----")
    blob_path = load_weather_data(weather_bytes)

    print("----Step 4: Verify-----")
    verify_blobs()

    print("----Step 5: Read Back-----")
    read_weather_data(blob_path)

