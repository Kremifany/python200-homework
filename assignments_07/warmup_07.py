from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

print("-----Initial Setup------")
if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
print('OpenAI client created.')


print("-----Baseline query-----")
baseline_messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'What time is it right now?'},
]

baseline_response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages=baseline_messages,
)
print(baseline_response.choices[0].message.content)


print("----Tool Definition----")
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


print("----------Passing the tool description to the model---------")

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Convert a Celsius temperature to Fahrenheit.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'celsius': {
                        'type': 'number',
                        'description': 'The temperature in Celsius to convert.',
                    }
                },
                'required': ['celsius'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]
print('Tools list defined')

print("----------Implementing the ReAct Agent---------")

def run_agent1(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can Convert a Celsius temperature to Fahrenheit.
                     Use the tool celsius_to_fahrenheit whenever a user asks to convert a temperature from Celsius temperature to Fahrenheit .'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or '{}')
            if function_name == 'celsius_to_fahrenheit':
                tool_result = celsius_to_fahrenheit(args['celsius'])
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''


print("---Lesson 02: Tool Definitions and the ReAct Loop---\n")
print("----Q1----call function directly----")
for temp in [0, 100, -40]:
    print(f"for temperature in celcius the temperature in Fahrenheit is {celsius_to_fahrenheit(temp)}\n")


print("----Q2----")

print("----Tool Definition----")
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("----------Passing the tool description to the model---------")


# Q1: Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
# A1: The agent will not trigger the tool because he needs to convert temperature but the tool is not doing that.
#     The dicision of triggering the tool is after reading tool description.
#     Also the model can  already have that knowledge of the conversion temp and not trigger tool from that reason.
# Q2: How many API calls will be made to answer this query?
# A2: 2 if agent got triggered and 1 if the model kwons the answer

# Q3: Then call run_agent("Convert 100 degrees Celsius to Fahrenheit") and print the result. Was your prediction correct?

print(run_agent1("Convert 100 degrees Celsius to Fahrenheit"))

# the output is:
# No tools needed....
# To convert Celsius to Fahrenheit, you can use the formula:
# F = (C × 9/5) + 32
# For 100 degrees Celsius:
# F = (100 × 9/5) + 32
# F = 180 + 32
# F = 212
# So, 100 degrees Celsius is equal to 212 degrees Fahrenheit.

# A3: Model was not needed of tool the knowledge was there, my prediction was right

print("----Q3----")
# Now extend the agent to support both tools. 
# Update your tools list to include celsius_to_fahrenheit (using the schema from Q1), and update run_agent to dispatch it when the model requests it.
# Test the extended agent on both of these queries:

# Q: Add a comment after each print() explaining whether a tool was called and why.
response_a = run_agent1("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# A: was called a tool converting c to f because the model decided so
response_b = run_agent1("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# A: the tool was not called becase the model had a knowledge and tool was not appropriate for that question

print("---------Lesson 03: Multi-Tool Agent----------")
# For Q4-Q6, use the full CsvManager class and run_agent_cycle setup from the lesson (copy them into your file).
# You will extend them.


# 

print("-----Setting up-----")
if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

client = OpenAI()
print("OpenAI client created.")

RESOURCES_DIR = Path("resources")
print(repr(RESOURCES_DIR))

print("---------Defining the external tools---------")

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        
        """
       
        error = self._ensure_loaded()
        if error:
            return error

        for col in [col1, col2]:
            if col not in self.df.columns:
                return {"error": f"'{col}' is not a column."}

        a = self.df[col1]
        b = self.df[col2]

        if not pd.api.types.is_numeric_dtype(a) or not pd.api.types.is_numeric_dtype(b):
            return {"error": "Both columns must be numeric to compute correlation."}

        paired = self.df[[col1, col2]].dropna()
        if len(paired) < 2:
            return {"error": "Could not compute correlation (not enough valid data)."}

        pearson_r, p_value = pearsonr(paired[col1], paired[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

print("Class defined")

print("---------Defining the tool schema-----------")
csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "compute_correlation": csv_backend.compute_correlation,
    "plot_data": csv_backend.plot_data,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First column name.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second column name.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
]

print("--------Defining a response cycle--------")
def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

# name = tool_call.function.name
# tool_args = json.loads(tool_call.function.arguments or "{}")
# fn = node_tools.get(name)

print("----------Setting up agent chat-----------")
SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)
def run_agent():
    """
    Simple command-line chat loop so it feels like a chatbot.

    We keep a single 'messages' list for the whole session so the model
    sees the conversation history each turn.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    print("CSV data agent at your service. Here to help look at your CSV data!")
    print("Type a question. Type 'exit' to quit.\n")
    print("To start, try 'list csv files' or 'load bike_commute.csv'\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break
        
        print(f"User query: {user_text}")
        assistant_text = run_agent_cycle(messages, user_text)
        print(f"\nAssistant: {assistant_text}\n")

print("----------Running the agent--------")
run_agent()

print("------Q4-----")
# The lesson ended with the agent hitting the tool-round limit when asked to compute a correlation, 
# because no tool existed for it. Fix that.
# Add a compute_correlation method to CsvManager:done

# OUTPUT last agent response:

# User query: any too numeric
# ACT: compute_correlation({'col1': 'distance_km', 'col2': 'duration_min'})
# ACT: compute_correlation({'col1': 'distance_km', 'col2': 'avg_speed_kmh'})
# ACT: compute_correlation({'col1': 'distance_km', 'col2': 'avg_heart_rate'})
# ACT: compute_correlation({'col1': 'distance_km', 'col2': 'avg_traffic_density'})
# ACT: compute_correlation({'col1': 'duration_min', 'col2': 'avg_speed_kmh'})
# ACT: compute_correlation({'col1': 'duration_min', 'col2': 'avg_heart_rate'})
# ACT: compute_correlation({'col1': 'duration_min', 'col2': 'avg_traffic_density'})
# ACT: compute_correlation({'col1': 'avg_speed_kmh', 'col2': 'avg_heart_rate'})
# ACT: compute_correlation({'col1': 'avg_speed_kmh', 'col2': 'avg_traffic_density'})

# Assistant: Here are some correlations between numeric columns:

# - distance_km and duration_min: 0.43 (moderate positive)
# - distance_km and avg_speed_kmh: 0.13 (weak positive)
# - duration_min and avg_speed_kmh: -0.78 (strong negative)
# - duration_min and avg_traffic_density: 0.42 (moderate positive)
# - avg_speed_kmh and avg_traffic_density: -0.53 (moderate negative)
# - avg_heart_rate and avg_traffic_density: 0.24 (weak positive)

# Other pairs have weaker or insignificant correlations. Let me know if you want details on any specific pair or further analysis.  

print("------Q5-------")
# Recreate the scenario from the lesson that hit the tool-round limit.
# Set up the agent with the system prompt from the lesson, then run:

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

# With the new tool in place, the agent should now succeed. Print the agent's final response.
# ------Q5-------
# ACT: load_csv({'filename': 'bike_commute.csv'})
# ACT: compute_correlation({'col1': 'avg_traffic_density', 'col2': 'avg_speed_kmh'})
# The correlation between avg_traffic_density and avg_speed_kmh is approximately -0.53, indicating a moderate negative relationship.
#  This means that as traffic density increases, average speed tends to decrease.

print("------Q6-------")
# After Q5 runs, print the full messages list. 
# Each item in the list is a dictionary with a "role" key. 
# Add a comment above the print that identifies what each role (system, user, assistant, tool)
# represents in the ReAct loop.
# Hint:

import json
print(json.dumps(messages, indent=2, default=str))

# OUTPUT:
# ------Q6-------
# [
#   {
#     "role": "system",
#     "content": "You are a small data assistant for CSV files stored in resources/. Use the available tools to do any data work (do not guess). If no CSV is loaded yet, load one first (or list available CSV files). Keep answers short and student-friendly."     
#   },
#   {
#     "role": "user",
#     "content": "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
#   },
#   {
#     "role": "assistant",
#     "content": null,
#     "tool_calls": [
#       {
#         "id": "call_Ajqzce6PtYnVmCGsQouB9IMX",
#         "function": {
#           "arguments": "{\"filename\":\"bike_commute.csv\"}",
#           "name": "load_csv"
#         },
#         "type": "function"
#       }
#     ]
#   },
#   {
#     "role": "tool",
#     "tool_call_id": "call_Ajqzce6PtYnVmCGsQouB9IMX",
#     "content": "{\"message\": \"Loaded bike_commute.csv with shape (160, 6).\", \"columns\": [\"distance_km\", \"duration_min\", \"avg_speed_kmh\", \"avg_heart_rate\", \"avg_traffic_density\", \"rain\"]}"
#   },
#   {
#     "role": "assistant",
#     "content": null,
#     "tool_calls": [
#       {
#         "id": "call_5eYatUrbZT3KumfKhqdo2G9e",
#         "function": {
#           "arguments": "{\"col1\":\"avg_traffic_density\",\"col2\":\"avg_speed_kmh\"}",
#           "name": "compute_correlation"
#         },
#         "type": "function"
#       }
#     ]
#   },
#   {
#     "role": "tool",
#     "tool_call_id": "call_5eYatUrbZT3KumfKhqdo2G9e",
#     "content": "{\"col1\": \"avg_traffic_density\", \"col2\": \"avg_speed_kmh\", \"pearson_r\": -0.5321, \"p_value\": 0.0}"       
#   },
#   {
#     "role": "assistant",
#     "content": "The correlation between average traffic density and average speed (km/h) is approximately -0.53, indicating a moderate negative correlation. This means as traffic density increases, average speed tends to decrease."
#   }
# ]

print("----Lesson 04: smolagents-----")

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_backend.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_backend.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_backend.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_backend.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_backend.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Retirms a short success message string, or an error dict/string.
    """
    return csv_backend.plot_data(y=y, x=x, plot_type=plot_type)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two numeric columns.

    Args:
        col1: First column name.
        col2: Second column name.

    Returns:
        A dict with col1, col2, pearson_r, and p_value, or an error dict.
    """
    return csv_backend.compute_correlation(col1, col2)

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation
]


# Re-wrap compute_correlation as a smolagents tool using the @tool decorator. 
# The decorated function should call csv_manager.compute_correlation(col1, col2) under the hood.
# After defining it, run:

print("----Q7-----")
print(compute_correlation.description)
# Add a comment comparing what smolagents generates automatically to the JSON schema you wrote manually in Q4.
# Smolagents framework mantaing the schema for us and we no need to specify anything we did in
# Q4 by hand here only description and tools and the rest done from the defined function automatically

# What information does smolagents need from you (the developer) in order to produce a good description?
#  function name, type hints on parameters, docstring: with summary , Args block describing each parameter and returns

print("----Q8-----")
# Create both a ToolCallingAgent and a CodeAgent using the same TOOLS list from the lesson
# (including your new compute_correlation tool) and the same OpenAIServerModel. Run the following prompt through both:


print("-------Create and test tool calling agent------")
model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

tool_agent = ToolCallingAgent(tools=TOOLS,
                         model=model,
                         instructions=SYSTEM_PROMPT,)

print("-------Create and test code agent--------\n-------Coding Prompt:-------")
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- ALWAYS put every response inside a <code>...</code> block. Never reply with plain text.
- When the task is done (even for simple questions like listing files), end with:
  final_answer("your reply to the user")
  inside the <code> block. Never use a stray </code> tag without an opening <code>.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color). 
  Explain what failed and what you would need to proceed.
- For custom matplotlib plots, use plt.show() to display the plot window.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

print("-------Initializing the code Agent:------")

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_backend})
print("-------------------response_tool------------\n", response_tool)
print("-------------------response_code------------\n", response_code)

# Print both responses, then add a comment block answering:
# What did each agent actually produce? Did the ToolCallingAgent change the dot color? Did the CodeAgent?
# A: Each agent produces a scattered plot of avg_heart_rate vs duration_minbut tooCallingAgent did it with the blue dots and claimed it was sucessfull greeen dots and 
# codingAgent did produce green dots
# What does this reveal about when each type of agent is more useful?
# for more specific tasks coding agent and for more general toolCallingAgent

# Q9
# Add a comment block at the bottom of your warmup file answering both questions:

# Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent.
# A: the task where the toolAlling agent will feet the best is a task where no need for somthing different then the tools can provide
#  For instance compute the corelation.
#  What property of the task makes it a good fit for a tool-based approach?
# A: generic task that is well defined and fully covered by the tool 
# What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent? (T
# A: It could generate wrong, unsafe code and run it itself. 

