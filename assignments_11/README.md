# Assignment 11 — Weather Running Conditions ETL

Final project for Python 200: a Prefect-orchestrated ETL pipeline that classifies hourly weather for outdoor running and stores the results in Azure.

## Start here

**Full project write-up:** [outputs/pipeline_run.md](outputs/pipeline_run.md)

That document covers the business question, architecture, data flow, output schema, Prefect/Azure verification, debugging notes, and tools used.

**Demo video:** https://youtu.be/DTfBegCamIE

## Files

| File | Role |
|------|------|
| [`etl_pipeline.py`](etl_pipeline.py) | Extract → Transform → Load flow |
| [`warmup_11.py`](warmup_11.py) | Prefect / production-pattern warmup notes |
| [`outputs/pipeline_run.md`](outputs/pipeline_run.md) | Project documentation (featured) |

## Pipeline in one line

Open-Meteo forecast → reshape hourly records → OpenAI labels (`good` / `marginal` / `bad`) → Azure Blob `pipeline-data/final/<date>/weather_etl.json`

## How to run

```bash
# From repo root, with .env containing OPENAI_API_KEY and Azure auth available
python assignments_11/etl_pipeline.py
```

Requires: OpenAI API access and Azure credentials (`DefaultAzureCredential`).
