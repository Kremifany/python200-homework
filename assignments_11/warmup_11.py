# Prefect Orchestration


# Prefect Question 1

# Q: What is the difference between a @task and a @flow in Prefect?
#
# From what I got from the docs and etl_pipeline.py, a @flow is the main pipeline function.
# It calls the tasks in order and passes data between them. In my project that's etl_pipeline()
# -- it runs extract, then transform, then load.
#
# A @task is one piece of that pipeline. Each task shows up as its own run in Prefect, and
# you can set retries on it (I used retries=2 on extract and transform). So the flow is the
# overall job, and tasks are the individual steps.
#
# Q: You have a helper that converts Celsius to Fahrenheit -- pure, in-memory, no I/O.
# Would you decorate it with @task? Why or why not?
#
# I wouldn't. It's just math in memory -- c * 9/5 + 32 -- so there's nothing to retry and
# nothing that can really fail. Making it a @task would just add extra Prefect overhead for
# no reason. I'd leave it as a normal helper function and call it from inside a task if I
# needed it, like how I kept small stuff inline in transform instead of making everything a task.

# Prefect Question 2
# Write the decorator for a task named call_api that retries up to 3 times
# with a 30-second delay between attempts.

# @task(retries=3, retry_delay_seconds=30)

# Prefect Question 3
# You run your pipeline and the Prefect UI shows: extract is Completed, transform is Failed, load never ran.
# In a comment block, describe: where in the UI do you look to understand what went wrong,
# and what specific information would you expect to find there?

# I would look into Flow Runs -> open that run -> click the transform task run (the one marked Failed so RED) -> Logs tab
#
# What specific info I'd expect to find there:
# - the error message -what actually went wrong
# - the exception type like APIStatusError, KeyError
# - the traceback - which file and line number in transform() caused the failure
# - any log/print output from transform before it failed
#
# Load never ran, so there's nothing useful to check on load -- the transform task logs
# are where the failure message would be.


# Production Patterns
# Production Question 1
# Explain what raise_for_status() does and why it's better than
# if response.status_code != 200: print("error") in a pipeline task.
# What happens to downstream tasks in each case when the API returns a 500?

# raise_for_status() checks the HTTP response code and raises an exception (HTTPError)
# if the request failed like a 500 server error. I use it in extract() in etl_pipeline.py
# right after requests.get().
#
# It's better than just printing "error" because printing doesn't stop the task. If you only
# print and keep going, Prefect thinks extract succeeded and passes bad/missing data to
# transform, which could fail in a confusing way later or even load garbage to the blob.
#
# When the API returns 500:
# - with raise_for_status(): extract fails immediately, Prefect marks it Failed and can retry
#   it (I have retries=2 on extract). If retries run out, transform and load never run.
# - with print("error") only: extract probably finishes as Completed, transform and load still
#   run, but they're working with a broken response -- so the failure is harder to catch and
#   debug.

# Production Question 2
# Pipeline uploads to final/{today}/weather_etl.json with overwrite=True.
# It crashes halfway through transform. You fix the bug and re-run from the beginning.
# What does overwrite=True protect you from, and what would happen without it?

# Since the crash happened during transform, load never ran on that failed attempt -- so no
# file was uploaded from the broken run. But when I fix the bug and re-run the whole pipeline,
# it still writes to the same path: final/{today}/weather_etl.json.
#
# overwrite=True protects me if that blob already exists, like if I had a successful run
# earlier today, or I'm re-running after a previous attempt that did make it to load. With
# overwrite=True, the new run just replaces the old file with the fresh complete results.
#
# Without overwrite=True, Azure would throw a "blob already exists" error when load tries to
# upload to a path that's already there. The pipeline would fail at the last step even though
# extract and transform worked, and the old file would still be sitting in the blob unchanged.

# Production Question 3
# Task stub: decorator, signature, one INFO log line with get_run_logger().

from prefect import task
from prefect.logging import get_run_logger


@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")

