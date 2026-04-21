import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prefect import flow, task
import scipy
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns   
# Rebuild the pipeline from Q1 using Prefect. Copy your three functions from Pipeline Question 1 (create_series, clean_data, summarize_data)
# into this file and turn them into Prefect tasks using @task.

@task
def create_series(arr):
    series = pd.Series(arr, name="values")
    return series

@task
def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series

@task
def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary

@flow
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

if __name__ == "__main__":
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    print(f"Original array:\n{arr}\n")
    summarized_data = data_pipeline(arr)
    print("Summary of the data:")   
    for key, value in summarized_data.items():
        print(f"{key}: {value}")

# Q: Why might Prefect be more overhead than it is worth here?
# A: because it just made it working longer for the same results, and it is not necessary for such a simple pipeline.
# Q: Describe some realistic scenarios where a framework like Prefect could still be useful,
#   even if the pipeline logic itself stays simple like in this case.
# A: Prefect could be helpful for tasks that hae to be executed with schedule and repeted number of times to benefit from prefect