print("----Azure Authentication----")
print("----Azure Authentication Question 1----")

# In a comment block, answer: when you run a Python script locally that uses DefaultAzureCredential,
#  what does it rely on to authenticate? What command
#  must you have run first, and how does DefaultAzureCredential know to use it?
# A: when I ran az login command it opened a browser and authenticated me
#   Asure CLI which I've installed on my machine cached the login
#  When authentication needed DefaultAzureCredentialdoes a credential chain which is tries all the methods to authenticate and stops wjhen one of them succeed
#  or if the previous not configured on dev machine it fallsback to az login session and obtains tokens from it.

print("----Azure Authentication Question 2----")

# In a comment block, answer: why can't a deployed pipeline (running on an Azure VM or
# container) use az login for authentication? What does it use instead, and why does
# the same Python code work without changes?

# A: Deployed pipeline cannot use command az login because the process of login is interactive and requires human to sign in
#  Instead it uses Managed identity or similar processes. 
# Azure assigns the VM/container a built-in identity, and DefaultAzureCredential
# obtains tokens from the local metadata endpoint (IMDS) without any secrets in code.
# Same python code works seemlesly because the authentication not requres changes in code it separate process that 
# DefaultAzureCredential - credential chain will falls through AzureCliCredential after az login.

print("----Azure Authentication Question 3----")

# In a comment block, answer: You run a script that creates a DefaultAzureCredential and
# immediately gets an AuthenticationError. Describe the two most likely causes and how
# you would diagnose each.

# A:
# Cause 1 i would say could be a az login didnt wenrt through
# Cause 2 Managed Identity is not enabled or not reachable

print("----Blob Storage----\n-----Blob Storage Question 1----")
# In a comment block, describe the three-level hierarchy of Azure Blob Storage in your own words. 
# Give a concrete analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.).

# A:
# Azure Blob Storage has three nested levels:
#   1. Storage account — its a top level the name that holds all the storage like a pantry
#   2. Container — a grouping inside a storage account like a bucket. 
#   3. Blob — it's  the content of a container, one blob per container like apple or carrots

print("----Blob Storage Question 2----")

# A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
# A: for the JSON I would use Blob container because it is unstuctured and not any format
# Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
# A: here I would use the SQL table because it's stuctured and queries easily
# A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.
# A: I would store the arrays in blob container because it does not have table stucture

print("----Blob Storage Question 4----")


def upload_text(container_client, blob_name, text):
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(text.encode("utf-8"), overwrite=True)

    
