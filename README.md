# python200-homework

Coursework from **Python 200** (Code the Dream) — Python, data analysis, ML, agents, and cloud pipelines.

---

## Featured project — Assignment 11: Weather Running Conditions ETL

**End-to-end data pipeline:** extract hourly weather → enrich with AI running labels (`good` / `marginal` / `bad`) → load analysis-ready JSON to Azure Blob Storage, orchestrated with Prefect.

| | |
|---|---|
| **Project write-up** | [`assignments_11/outputs/pipeline_run.md`](assignments_11/outputs/pipeline_run.md) |
| **Pipeline code** | [`assignments_11/etl_pipeline.py`](assignments_11/etl_pipeline.py) |
| **Demo video** | [Watch on YouTube](https://youtu.be/DTfBegCamIE) |

**What it demonstrates:** ETL design, API integration (Open-Meteo + OpenAI), data shaping, cloud storage (Azure), orchestration (Prefect), retries/logging, and end-to-end verification in Prefect UI and Azure Storage Browser.

→ Start here: **[pipeline_run.md](assignments_11/outputs/pipeline_run.md)**

```bash
# Optional: run the pipeline (needs OpenAI API key + Azure credentials)
python assignments_11/etl_pipeline.py
```

---

## Stack

- **Language:** Python 3.x  
- **Data / ML:** pandas, numpy, matplotlib, scikit-learn, Jupyter  
- **Pipelines / cloud:** Prefect, Azure Blob Storage, OpenAI API  
- **Other:** agents / RAG tools in later assignments  

See [`requirements.txt`](requirements.txt) for dependencies.

---

## Course assignments

| Week | Focus | Folder |
|------|--------|--------|
| 01 | Python / Prefect warmups | [`assignments_01/`](assignments_01/) |
| 02 | Data analysis | [`assignments_02/`](assignments_02/) |
| 03 | ML / spam classification | [`assignments_03/`](assignments_03/) |
| 04 | Notebooks | [`assignments_04/`](assignments_04/) |
| 05 | Prompting / OpenAI | [`assignments_05/`](assignments_05/) |
| 06 | Retrieval / RAG | [`assignments_06/`](assignments_06/) |
| 07 | Agents | [`assignments_07/`](assignments_07/) |
| 08–09 | Cloud / Azure setup | [`assignments_08/`](assignments_08/), [`assignments_09/`](assignments_09/) |
| 10 | Blob + AI transforms | [`assignments_10/`](assignments_10/) |
| **11** | **Prefect ETL + Azure (featured)** | [`assignments_11/`](assignments_11/) |

Each folder usually has a warmup, a project, and an `outputs/` directory. See the README inside each assignment for a short overview.

---

## How to run (general)

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Examples:

```bash
python assignments_02/project_02.py
jupyter notebook assignments_04/project_04.ipynb
python assignments_11/etl_pipeline.py
```

---

## Highlight again

If you only open one file in this repo, open:

**[assignments_11/outputs/pipeline_run.md](assignments_11/outputs/pipeline_run.md)**  
— full description of the weather ETL pipeline, architecture, output schema, verification, and methods.
