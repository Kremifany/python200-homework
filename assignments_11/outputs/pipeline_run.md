# Weather Running Conditions ETL Pipeline

**Course:** Python 200 — Machine Learning and Cloud Computing (final project)  
**Project type:** End-to-end data pipeline (Extract → Transform → Load)  
**Code:** `assignments_11/etl_pipeline.py`  
**Demo video:** https://youtu.be/DTfBegCamIE  

---

## 1. Business / analysis question

Outdoor runners care about hourly weather, not just a daily summary. This pipeline answers:

> For each hour today, are outdoor running conditions good, marginal, or bad — given temperature and precipitation?

It does that by:

1. Pulling a multi-day hourly weather forecast for a city (Charlotte, NC by default)
2. Classifying each hour as **good / marginal / bad** for running with an LLM
3. Storing enriched, analysis-ready records in Azure Blob Storage

From the output, you can analyze:

- How many “good” running hours are available tomorrow
- Which hours look marginal because of rain or temperature
- How conditions change hour by hour or day over day

Pattern: **raw API data → enrichment → durable storage for analysis**.

---

## 2. Architecture

```text
Open-Meteo API ──► extract() ──► transform() ──► load()
   (weather)         Prefect        OpenAI         Azure Blob
                     task           gpt-4o-mini    pipeline-data/
                                                   final/<date>/weather_etl.json
```

| Step | Responsibility | Tools |
|------|----------------|--------|
| **Extract** | Call weather API, validate HTTP response, return raw JSON | `requests`, Open-Meteo |
| **Transform** | Reshape hourly lists into records; classify first 24 hours | OpenAI Chat Completions |
| **Load** | Upload enriched JSON to cloud storage | Azure Blob Storage |
| **Orchestration** | Run steps in order with logging and retries | Prefect `@flow` / `@task` |

Extract retrieves the forecast. Transform adds the running-condition label. Load writes temperature, precipitation, time, and that label into Azure.

---

## 3. Data flow

### Extract
- Source: Open-Meteo forecast API
- Location: latitude / longitude (default Charlotte: `35.2271, -80.8431`)
- Features: hourly `temperature_2m` (°C), `precipitation` (mm)
- Window: 7 days of hourly data
- Failure handling: `raise_for_status()` so bad HTTP responses fail the task instead of continuing with bad data
- Retries: up to 2, with a 10-second delay

### Transform
- Reshapes parallel API arrays (`time`, `temperature_2m`, `precipitation`) into one record per hour
- Scope: **24 records** (one day) via `MAX_RECORDS = 24`
- Classifier: `gpt-4o-mini` with a constrained system prompt
- Labels: `good`, `marginal`, `bad`
- Fallback: unexpected model output → `unknown` (warning logged)
- Progress: log every 6 records (`Classified 6/24`, `12/24`, …)

### Load
- Container: `pipeline-data` (storage account `fanyctd2026sa`)
- Path: `final/<YYYY-MM-DD>/weather_etl.json`
- Auth: `DefaultAzureCredential`
- Write mode: `overwrite=True` so same-day re-runs replace the prior file cleanly

---

## 4. Output data model

Each record in `weather_etl.json`:

```json
{
  "time": "2026-08-20T14:00",
  "temperature_2m": 28.4,
  "precipitation": 0.2,
  "conditions": "marginal"
}
```

| Field | Description |
|-------|-------------|
| `time` | Hour timestamp from the forecast |
| `temperature_2m` | Air temperature (°C) |
| `precipitation` | Precipitation (mm) |
| `conditions` | Running-condition class: `good`, `marginal`, `bad`, or `unknown` |

Example analyses on this table:

- Daily count of good vs bad hours
- Distribution of conditions by hour of day
- Filter “good” windows for a running schedule

---

## 5. Orchestration and reliability

- Prefect `@flow` (`etl_pipeline`) owns the full run
- Prefect `@task` isolates extract / transform / load (separate status + logs)
- Retries on extract and transform cover temporary API failures
- Logging via Prefect run logger and progress prints
- Date-partitioned blob path + overwrite for repeatable daily loads
- Secrets loaded from `.env` (OpenAI key not hard-coded)

### How to run

```bash
python assignments_11/etl_pipeline.py
```

Prefect starts a temporary local server, runs the flow, and records task states in the UI.

---

## 6. End-to-end verification (demo)

Verified full path: code → Prefect UI → Azure Storage.

1. **Prefect + pipeline**  
   Local Prefect server runs while `etl_pipeline.py` executes extract → transform → load.

2. **Task outcomes**  
   - Extract completes first (forecast retrieved)  
   - Transform runs next (AI labeling) — longest step because of repeated model calls  
   - Load uploads the final blob; temporary server then stops

3. **Azure Storage Browser**  
   - Containers → `pipeline-data` → `final` → today’s date → `weather_etl.json`  
   - File contains temperature, precipitation, time, and the running-condition label

4. **Prefect UI**  
   - Latest flow run: **Completed / Succeeded**  
   - extract, transform, and load all completed  
   - Logs show progress and info messages for each step

---

## 7. Debugging notes

### First run (failed)
- Extract succeeded
- Transform failed with OpenAI `APIStatusError` (431 — request headers too large)
- Also hit `MissingContextError` from calling `get_run_logger()` at module level instead of inside a task

### Fixes
1. Create the logger inside transform and load tasks
2. Create the OpenAI client once at module level after loading `.env`
3. Keep request payloads small and consistent

### Second run (successful)
- extract, transform, and load all completed
- All 24 hourly records classified
- Uploaded to `final/<today>/weather_etl.json`
- Transform progress logs visible (`Classified 6/24` …)
- No retries on the successful run

### Production hardening (if scheduled daily)
- Prefect deployment with a schedule
- Failure alerts (email / Slack) for transform or load failures
- Monitor OpenAI quota and Azure storage growth

---

## 8. Methods and tools used

| Area | What was applied in this project |
|------|----------------------------------|
| **ETL design** | Separate extract, transform, and load stages |
| **API work** | Weather API + OpenAI API + Azure Blob |
| **Data shaping** | Nested JSON arrays → tidy per-hour records |
| **Feature enrichment** | LLM classifier with fixed labels and `unknown` fallback |
| **Cloud storage** | Date-partitioned blobs under `final/<date>/` |
| **Orchestration** | Prefect flows/tasks, retries, UI inspection |
| **Reliability** | `raise_for_status()`, retries, overwrite, logging |
| **Validation of results** | Prefect task status + Storage Browser confirmation of the output file |

---

## 9. Possible extensions

- Multi-city / multi-location parameter
- Dashboard on top of the JSON (Streamlit / Power BI)
- Persist history in DuckDB or Postgres instead of blobs only
- Batch classification or rules baseline to reduce per-hour LLM cost
- Data quality checks before load (null temps, row-count = 24, unexpected `unknown` rate)
