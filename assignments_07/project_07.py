# Mini-Project — World Happiness Agent
import matplotlib
from pathlib import Path
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.stats import pearsonr
import os

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool, CodeAgent

BASE_DIR = Path(__file__).resolve().parent.parent  # here python homework 200 folder
DATA_PATH = BASE_DIR / "assignments_01" / "outputs" / "merged_happiness.csv"
PLOT_DIR = Path("outputs")

# The active dataset. load_happiness_data() fills this in.
df = None


# ----- Task 1: Define Your Tools -----
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness data so the other tools have something to work with.

    Use this first, before any other tool. It tries to read the merged CSV, and
    if that file isn't there it loops over the yearly files and joins them
    together instead. The result is saved in the global df.

    Returns:
        A dict with the "shape" and "columns" of the data I loaded, or a dict
        with an "error" message if I couldn't find any files.
    """
    global df

    merged_path = Path(DATA_PATH)
    yearly_dir = Path(__file__).resolve().parent.parent / "assignments_01" / "data"

    if merged_path.exists():
        df = pd.read_csv(merged_path)
    else:
        standard_columns = [
            'Ranking', 'Country', 'Regional indicator', 'Happiness score',
            'GDP per capita', 'Social support', 'Healthy life expectancy',
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
        ]
        frames = []
        for file_path in sorted(yearly_dir.glob("world_happiness_*.csv")):
            year_df = pd.read_csv(file_path, sep=';', decimal=',')
            year_df.columns = standard_columns[:len(year_df.columns)]
            year_df['Year'] = file_path.stem.split('_')[-1]
            frames.append(year_df)
        if not frames:
            return {"error": f"No data at {merged_path} or in {yearly_dir}"}
        df = pd.concat(frames, ignore_index=True)

    return {"shape": df.shape, "columns": df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Give quick stats for one column, like its average, min and max.

    Use this when someone wants a summary of a single column, for example
    "describe the Happiness score" or "what's the average GDP per capita".
    Make sure the data is loaded first.

    Args:
        column: The name of the column I should summarize, e.g. "GDP per capita".

    Returns:
        A dict from pandas describe() with count, mean, std, min, the quartiles
        and max. If nothing is loaded or the column name is wrong, I return a
        dict with an "error" message instead.
    """
    global df

    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Check how strongly two number columns are related using Pearson correlation.

    Use this when someone asks if two columns move together, like "is GDP per
    capita related to the Happiness score". Both columns have to be numbers.

    Args:
        col1: The first number column to compare.
        col2: The second number column to compare.

    Returns:
        A dict with "col1", "col2", "pearson_r" (between -1 and 1) and
        "p_value", both rounded to 4 decimals. If the data isn't loaded, a
        column is missing or not numeric, or there aren't enough rows, I return
        a dict with an "error" message.
    """
    global df

    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    for col in (col1, col2):
        if col not in df.columns:
            return {"error": f"Column '{col}' not found. Available: {df.columns.tolist()}"}
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Column '{col}' is not numeric."}

    pair = df[[col1, col2]].dropna()
    if len(pair) < 2:
        return {"error": "Not enough data points to compute correlation."}

    r, p = pearsonr(pair[col1], pair[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(r, 4),
        "p_value": round(p, 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Find the best countries for one year, ranked by a column.

    Use this for "top" questions, like "the 5 happiest countries in 2020" or
    "which countries had the highest GDP per capita in 2019". I keep only the
    rows for that year, sort them from highest to lowest, and take the top few.

    Args:
        column: The column to rank by, e.g. "Happiness score".
        year: The year I should look at, e.g. 2020.
        n: How many countries to return. Defaults to 5.

    Returns:
        A dict with a "top" list, where each item is a small dict holding the
        "country" and its value for that column. If the data isn't loaded, a
        column is missing, or that year has no rows, I return a dict with an
        "error" message.
    """
    global df

    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    for required in ("Country", "Year", column):
        if required not in df.columns:
            return {"error": f"Column '{required}' not found. Available: {df.columns.tolist()}"}

    year_df = df[df["Year"].astype(str) == str(year)]
    if year_df.empty:
        return {"error": f"No rows found for year {year}."}

    top = year_df.sort_values(column, ascending=False).head(n)
    records = [
        {"country": row["Country"], column: row[column]}
        for _, row in top.iterrows()
    ]
    return {"top": records}


TOOLS = [
    load_happiness_data,
    summarize_column,
    compute_correlation,
    get_top_n_countries,
]

# ----- Prompts
TOOL_AGENT_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

CODE_AGENT_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).

A pandas DataFrame named `df` is already available in your code with the data loaded.
Use it directly for custom analysis or plots (do NOT try to read a CSV from disk).
Its exact column names are:
'Ranking', 'Country', 'Regional indicator', 'Happiness score', 'GDP per capita',
'Social support', 'Healthy life expectancy', 'Freedom to make life choices',
'Generosity', 'Perceptions of corruption', 'Year'.
Always use these exact column names (e.g. 'Regional indicator', not 'Region';
'Happiness score', not 'happiness_score').

Be concise and student-friendly in your responses.
"""

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]


if __name__ == "__main__":
    # --- Setup ---
    if load_dotenv():
        print("Successfully loaded environment variables from .env")
    else:
        print("Warning: could not load environment variables from .env")
    api_key = os.getenv("OPENAI_API_KEY")
    print(repr(DATA_PATH))
    PLOT_DIR.mkdir(exist_ok=True)

    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

    # --- Tool-calling agent, tests ---
    print("-------Create and test tool calling agent------")
    tool_agent = ToolCallingAgent(
        tools=TOOLS,
        model=model,
        instructions=TOOL_AGENT_PROMPT,
    )

    print("-------------Test 1:---------------")
    tool_agent.run("load world hapiness df")
    print("-------------Test 2:---------------")
    tool_agent.run("summarize column GDP per capita")
    print("-------------Test 3:---------------")
    tool_agent.run("What is the correlation between GDP per capita and Happiness score?")
    print("-------------Test 4:---------------")
    tool_agent.run("What are the top 5 happiest countries in 2020?")

    # --- Task 2: Build the code agent ---
    print("-----Task 2: Build the Agent----")
    agent = CodeAgent(
        tools=TOOLS,
        model=model,
        instructions=CODE_AGENT_PROMPT,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
        max_steps=8,
    )

    # --- Task 3: Run guided queries ---
    print("-----Task 3: Run Guided Queries----")
    # Pre-load so the DataFrame can be handed into the code agent's sandbox.
    load_happiness_data()

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)

    # --- Task 4: My own questions ---
    print("-----Task 4: Your Own Questions----")

    # My query 1
    my_query_1 = "What is the correlation between Social support and Happiness score?"
    response_1 = agent.run(my_query_1, reset=False, additional_args={"df": df})
    print(response_1)
    # Comment: This one just used a tool. My compute_correlation tool already does
    # exactly this, so the agent called it and gave back the answer. No code needed.

    # My query 2
    my_query_2 = "What is the average Happiness score for each Regional indicator? Sort them from highest to lowest."
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # Code generation only. None of my tools do a groupby average, so the agent had to write Python 

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#
#    My tool gave back r = 0.6313 and a p_value of 0.0, and the agent pretty much just
#    repeated those numbers. The p-value being basically 0 means it's significant, so that
#    part was fine. But it never actually said "p < 0.05" or explained the cutoff - it just
#    showed the tiny p-value and said significant. So it used the p-value right but kind of
#    skipped explaining the threshold.
#
# 2. Did any of the agent's responses surprise you - either by being more capable than
#    you expected, or less? Describe one specific example.
#
#    It surprised me by struggling on the easiest question. When I asked it to load the
#    data and tell me the shape and columns, it kept trying df.shape even though df wasn't there
#    yet and just retried until it hit the step limit - even though the load tool had already
#    returned the shape and columns. I expected plotting to be the hard part, but this trivial
#    question was the one it couldn't handle.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#
#    I'd add an average_by(group_col, value_col) tool that gives the average of one column for
#    each group of another, already sorted. Then stuff like "average happiness per region" or
#    "average GDP per year" would just be one tool call instead of the agent writing its own
#    groupby code every time.