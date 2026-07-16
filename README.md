# python200-homework

A collection of Python coursework and exercises from the Python200 program at Code the Dream. This repository contains weekly warmups, project assignments, datasets, example scripts, and Jupyter notebooks used to practice Python fundamentals and introductory data analysis.

## Stack
- **Language(s):** Python
- **Framework / runtime:** CPython 3.x, Jupyter Notebook
- **Notable libraries:** pandas, numpy, matplotlib (typical data-science stack used in the notebooks and scripts)

## Repository layout
```
README.md
liniar_regression.py
linear_regression.py
outputs/
assignments_01/
  warmups_01.py
  project_01.py
  hello_prefect.py
  prefect_warmup.py
  data/
assignments_02/
  warmup_02.py
  project_02.py
  student_performance_math.csv
  outputs/
assignments_03/
  warmup_03.py
  project_03.py
  outputs/
assignments_04/
  warmup_04.ipynb
  project_04.ipynb
  outputs/
assignments_05/
  warmup_05.py
  project_05.py
assignments_06/
  warmup_06.py
  project_06.py
  resources/
assignments_07/
  warmup_07.py
  project_07.py
  scatter_plot.png
  outputs/
assignments_08/
assignments_09/
assignments_10/
assignments_11/
```

Each `assignments_N` directory typically includes:
- warmup exercises (warmup_N.py or notebook)
- a larger project script or notebook (project_N.py / .ipynb)
- local datasets and an `outputs/` folder for generated artifacts

## How to run
1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install common data-science libraries (or create a requirements file from your environment):
```bash
pip install pandas numpy matplotlib scikit-learn jupyter seaborn prefect
```

3. Run a script:
```bash
python assignments_02/project_02.py
```

4. Open notebooks:
```bash
jupyter notebook assignments_04/project_04.ipynb
```

## Notes & recommendations
- Add a `requirements.txt` or `environment.yml` to make installs reproducible.
- Consider adding short READMEs inside each `assignments_N/` directory describing the learning goals and expected inputs/outputs for that week's project.
- Rename `liniar_regression.py` to `linear_regression.py` if that file contains a linear regression example.

## Questions you can ask me
- "Can you create a requirements.txt for this repo by scanning the notebooks and scripts?"
- "Add short READMEs to each assignments directory summarizing the learning objectives."
- "Run a quick scan of project_04.ipynb and extract the main analysis steps into a README summary."
