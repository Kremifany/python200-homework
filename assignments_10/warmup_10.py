# --- LLMs as Transform ---

# Q1
# For each task below, write a one-sentence comment saying whether you would use an LLM
# or deterministic code, and why.
#
# Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
# A: I would use deterministic code because the output is always the same for the given input.
#
# Classify a customer support ticket -- "my card was charged twice" -- into one of:
# billing, technical, or general.
# A: I would use an LLM because it needs to understand what the customer meant, not just match keywords.
#
# Calculate the average of a list of numbers.
# A: I would use deterministic code because it is basic math and code always gives the exact answer.
#
# Extract the company name from a freeform job title like
# "Sr. Data Eng @ Acme Corp (contract)".
# A: I would use an LLM because job titles are written in many different ways and the company name is not always in the same spot.
#
# Determine whether a product review is more than 100 words long.
# A: I would use deterministic code because counting words is a simple rule that code can check exactly every time.


# Q2
# Your colleague has written the following pipeline prompt:
# system = "Summarize this product review in a few sentences."
# In a comment block, explain what problem this creates downstream in a pipeline, and
# rewrite the prompt so it produces output that is easy to parse and store reliably.
#
# A: The problem is the output is just free text — "a few sentences" is vague, so each run
# might be a different length or format. Downstream code cannot reliably parse it into a
# database or compare results across reviews because there is no fixed structure.
#
# Rewritten prompt:
# system = (
#     "Summarize this product review in 2-3 sentences. "
#     'Please output a JSON object with a single key "summary" containing the text. '
#     "Return only valid JSON — no markdown, no extra explanation."
# )
# Example output: {"summary": "The customer loved the fit but found the soles wore out quickly."}


# Q3
# Your dataset has 50,000 records and you need to run a classification call for each one
# using gpt-4o-mini. In a comment block, answer:
# If each call takes 1 second on average, how long would sequential processing take?
# What is one practical strategy to handle this more efficiently at scale, without changing models?
#
# A: Sequential processing would take 50,000 seconds total, which is about 833 minutes or 14 hours.
# One practical strategy is to run many API calls in parallel with async code or a worker pool,
# so multiple records are running at the same time instead of one after another.


# --- Azure OpenAI ---

# Q1
# In a comment block, name two reasons an organization might use Azure OpenAI instead of
# calling the OpenAI API directly. Be specific -- "it's better" is not an answer.
#
# A: Azure OpenAI keeps data inside the organization's Azure cloud, which helps with security
# and compliance rules. Organizations can also use Azure identity (like managed identity)
# instead of managing separate OpenAI API keys on their own.


# Q2
# When you switch from OpenAI to AzureOpenAI, the client initialization takes three
# Azure-specific parameters. Name them and describe what each one is.
# (Do not include the standard api_key -- describe the Azure-specific ones.)
#
# A:
# 1. azure_endpoint — the URL of your Azure OpenAI resource in the portal. It tells the
#    client which Azure resource to send requests to.
# 2. api_version — a dated version string that tells Azure which REST API release to use.
#    OpenAI's direct API does not need this.
# 3. azure_ad_token_provider — a function that gets an Azure Entra ID token for
#    authentication instead of using a plain API key. This hooks into Azure identity
#    (like DefaultAzureCredential) when connecting to Azure OpenAI.


# Q3
# In a comment block, answer: when using AzureOpenAI, the model parameter in
# chat.completions.create() does not take a value like "gpt-4o-mini". What does it
# take instead, and where do you find the right value to use?
#
# A: It takes the deployment name — the name you gave the model when you deployed it in
# Azure (for example "my-gpt4o-mini"), not the public OpenAI model name. You find the
# right value in the Azure portal under your Azure OpenAI resource, in the Model
# deployments section, listed as the deployment name.
