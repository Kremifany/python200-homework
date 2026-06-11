
# Video link: https://youtu.be/w4_9a3hM3zQ

# Step 6: Reflect
# I think the LLM was ok for this assignment because we needed to practice the pipeline,
# but honestly rule-based code could have done the job better. We only have temperature
# and precipitation, so I could write something like if temp > 10 and precip < 1 then
# good — no API calls and same result every run. If I switched to rules I would save
# money and time and not worry about weird model answers, but I would lose the model
# guessing "marginal" for in-between weather that is hard to put in one simple if statement.

print("-----Project -- LLM Transform Pipeline-----")

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ACCOUNT_URL = "https://fanyctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
VALID_LABELS = {"good", "marginal", "bad"}
MAX_RECORDS = 24  # one day of hourly data — keeps API cost/runtime manageable
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)
print("------Step 1: Read------")
# Download the raw weather data you uploaded in Week 9 from raw/<today>/weather.json. 
# Parse the JSON and reshape the "hourly" parallel lists into a list of per-hour record dictionaries
# (each with "time", "temperature_2m", and "precipitation"). 
# If today's date doesn't match when you uploaded in Week 9, use the fallback dataset.
# If you did not complete Week 9, a fallback dataset is available at assignments/resources/weather_raw.json -- load it with json.load() and reshape it the same way.


today = date.today().isoformat()
credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

raw = container.download_blob(f"raw/{today}/weather.json").readall()
data = json.loads(raw.decode("utf-8"))
hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    records.append(
        {
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        }
    )

print(f"Loaded {len(records)} records")

print("-----Step 2: Transform-----")
# For each record, call the OpenAI API to classify the conditions as good, marginal, or bad for outdoor running, based on temperature and precipitation. Use this system prompt exactly (so your mentor can compare results):
# The user message for each record should be: "Temperature: <value>C, Precipitation: <value>mm".
# To keep costs and runtime manageable, process only the first 24 records (one day of hourly data).
#  Add a fallback: if the model's response is not one of the three valid labels, store "unknown" instead.
# Print a progress message every 6 records so you can see it running.
def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = []
for i, record in enumerate(records[:MAX_RECORDS]):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ],
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})
    if (i + 1) % 6 == 0:
        print(f"  Processed {i + 1} records...")

print("----Step 3: Write----")
# Upload the enriched records (with the new "conditions" field) 
# to processed/<today>/weather_classified.json in Blob Storage. Use overwrite=True.

processed_path = f"processed/{today}/weather_classified.json"
container.upload_blob(
    processed_path, json.dumps(enriched).encode("utf-8"), overwrite=True
)
print(f"Uploaded to {processed_path}")

print("----Step 4: Spot-Check----")
# Download the processed blob, load it into a pandas DataFrame, and print:
# df["conditions"].value_counts()
# The first 5 rows of the DataFrame

processed_raw = container.download_blob(processed_path).readall()
processed_data = json.loads(processed_raw.decode("utf-8"))
df = pd.DataFrame(processed_data)

print("\nconditions: ")
print(df["conditions"].value_counts())
print("\nFirst 5 rows:")
print(df.head())

# Save the first 10 enriched records to outputs/first_10_records.json so your mentor can inspect the results without running the script.
print("----Step 5: Save Output----")
output_path = Path("outputs/first_10_records.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(enriched[:10], indent=2), encoding="utf-8")
print(f"Saved first 10 records to {output_path}")