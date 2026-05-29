
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
# print("---API Question 1---")
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
# )
# # Print just the text of the response (not the whole object). 
# # Then print the name of the model that responded and the total number of tokens used. Label each output.

# reply = response.choices[0].message.content
# model_name = response.model
# total_tokens = response.usage.total_tokens

# print(f"Answer------------- {reply}")
# print(f"Model-------------- {model_name}")
# print(f"Total tokens------- {total_tokens}")

# print("---API Question 2---")
# prompt = "Suggest a creative name for a data engineering consultancy."
# temperatures = [0, 0.7, 1.5]
# for temp in temperatures:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=temp,
#     )
#     reply = response.choices[0].message.content
#     print(f"Answer------------- {reply}")
#     print(f"Model-------------- {model_name}")
#     print(f"Total tokens------- {total_tokens}")
#     print(f"Temperature------------- {temp}")

# # I would use lower temperature if the need was the more reproducible output
# # but in this case I would go with more creative one, so the highest tempreture

# print("---API Question 3---")
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
#     n=3,
#     temperature=1.0
# )
# for i, choice in enumerate(response.choices, start=1):
#     print(f"Completion {i}------ {choice.message.content}")

# print("---API Question 4---")
# # Set max_tokens=15 and send a prompt that would normally produce a long response
# #  (for example, "Explain how neural networks work.").
# #  Print the result. Add a comment: What happened, and why might you want to use max_tokens in a real application?
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Explain how neural networks work."}],
#     max_completion_tokens=15,
# )
# print("Limited output:", response.choices[0].message.content)

# # With max_completion_tokens=15, the response is cut short because generation stops
# # once the token cap is reached. This helps control cost/latency and
# # prevents very long outputs.

# print("---System Question 1---")
# # Use a system message to give the model a personality, then ask it a question. Print the response.
# messages = [
#     {"role": "system", "content": "You are a patient, encouraging Python tutor."
#     "You always explain things simply and end with a word of encouragement."},
#     {"role": "user", "content": "I don't understand what a list comprehension is."}
# ]
# response_tutor = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=messages,
# )
# print("\n Tutor personality response:", response_tutor.choices[0].message.content)

# # Now change the system message to give the model a completely different personality (your choice)
# # and ask the same question. Print that response too. Add a comment noting what changed.
# messages_alt = [
#     {"role": "system", "content": "You are a strict, no-nonsense technical reviewer. Be concise and direct."},
#     {"role": "user", "content": "I don't understand what a list comprehension is."}
# ]
# response_reviewer = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=messages_alt,
# )
# print("\n Technical reviewer response:", response_reviewer.choices[0].message.content)

# # The tutor response is warmer and more supportive, while the reviewer response is shorter and more concise.
# # changing the system message changes tone, style, and how the same content is explained.

# print("---System Question 2---")
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "My name is Jordan and I'm learning Python."},
#     {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
#     {"role": "user", "content": "Can you remind me what my name is?"}
# ]

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=messages,
# )
# print("\n Technical reviewer response:", response.choices[0].message.content)
# # Print the model's response. Add a comment: Why does the model know Jordan's name, even though it's stateless?
# # Model knows Jordans name because the messages list was sent to the model in the completions.create


reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

# print("Prompt Question 1 — Zero-Shot")
# Ask the model to classify the sentiment of each review below as positive, negative, or mixed.
# Give it no examples — just the task description and the reviews. 
# Print each result labeled with the review number.
# prompt = f"""
# Classify the sentiment of each review as positive, negative, or mixed.
# Return exactly one label per line in the same order as the reviews.
# {reviews}
# """
# responses = get_completion(prompt, model="gpt-4o-mini")
# labels = [line.strip().lower().strip(".") for line in responses.splitlines() if line.strip()]

# print(f"Sentiments by review:\n")
# for i, _review in enumerate(reviews, start=1):
#     label = labels[i - 1] if i - 1 < len(labels) else "unknown"
#     print(f"Review {i}: {label}")

# print("---Prompt Question 2 — One-Shot---")

# # Repeat the same task, but this time add one example before the reviews to show the model the format you want
# prompt = f"""
# Classify the sentiment of each review as positive, negative, or mixed.
# Return exactly 6 lines, 2 per review, in this exact format:
# Example:
# Review: "Fast shipping but the item arrived damaged."
# Sentiment: mixed

# Reviews:
# 1. {reviews[0]}
# 2. {reviews[1]}
# 3. {reviews[2]}
# """
# responses = get_completion(prompt, model="gpt-4o-mini")

# print(f"Sentiments by review:\n")
# print(responses)

# Adding one example  made output format more consistent than Q1,
# because the model has a concrete pattern to imitate.

# print("Prompt Question 3 — Few-Shot")
# # Repeat the task again, this time with three examples. At least one example should be positive,
# # one negative, and one mixed. Print the results.


# prompt = f"""
# Classify each review as positive, negative, or mixed.
# Use the examples below as the required format:
# Review: "The update is fantastic and made everything easier to use."
# Sentiment: positive
# Review: "The app freezes every time I try to save and support is unhelpful."
# Sentiment: negative
# Review: "Great features, but setup was confusing and took too long."
# Sentiment: mixed

# # Reviews:
# # 1. {reviews[0]}
# # 2. {reviews[1]}
# # 3. {reviews[2]}
# # """
# responses = get_completion(prompt, model="gpt-4o-mini")

# print(f"Sentiments by review:\n")
# print(responses)

# # Add a comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?
# I would use zero-shot for simplier tasks that the formatting not needed, for more complicated and format needed tasks would use one-shot
# For tasks that one-shot is not enough will use few-shot. Here it was unnessesary to use few-shot. One shot was more than enough.

# print("---Prompt Question 4 — Chain of Thought---")
# # Ask the model to solve the following problem, but instruct it to show its reasoning step by step before giving 
# # a final answer. Label the final answer clearly.
# problem = "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later"
# "takes a new job that pays $7,500 more per year than her post-raise salary."
# "What is her final annual salary?"

# prompt = f"""
# Solve the following problem, show the reasoning in format 
# step 1 :  put here the resoning
# step 2 : put here the resoning 
# step by step before giving 
# a final answer display without any **. Label the final answer clearly
# {problem}
# """
# responses = get_completion(prompt, model="gpt-4o-mini")
# print(responses)

# # Print the full response including the reasoning.
# # Add a comment: Why does asking the model to reason step by step tend to improve accuracy on problems like this?
# # The accuracy is improved with the steps because of the complicated problem devided by our prompt to many little problems.
# # This way the model makes her steps towards the solutions in more "secure" way less confusion and errors  

# print("---Prompt Question 5 — Structured Output---")
# # Ask the model to analyze the review below and return the result only as valid JSON with keys sentiment,
# # confidence (a float from 0 to 1), and reason (one sentence).
# # Print the raw response, then parse it
# # with json.loads() and print each field separately, labeled.


# import json

# review = "I've been using this tool for three months. It handles large datasets well, \
# but the UI is clunky and the export options are limited."

# prompt = f"""
# analyze the review below and return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1),
# and reason (one sentence).
# {review}
# """
# responses = get_completion(prompt, model="gpt-4o-mini")
# print(responses)

# try: 
#     result = json.loads(responses)
#     # result is dictionary

#     for key, value in result.items():
#         print(f"{key}: {value}")
# except json.JSONDecodeError: 
#     print(f"The response is not valid JSON.{responses}")

# print("---Prompt Question 6 — Delimiters---")
# # Use triple backticks as delimiters to clearly separate the user's text from your instructions.
# # Send the prompt below and print the result.
# user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
# pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

# prompt = f"""
# You will be given text inside triple backticks.
# If it contains step-by-step instructions, rewrite them as a numbered list.
# If it does not contain instructions, respond with exactly: "No steps provided."

# ```{user_text}```
# """

# responses = get_completion(prompt, model="gpt-4o-mini")
# print(responses)

# Then send a second prompt using a passage that is not a set of instructions
# (any sentence or two of regular prose). Confirm that the model returns "No steps provided." 
# Add a comment: What problem do delimiters help prevent?
# Delimeters ment to prevent confusin text for the model to work on with actual instructions to the model

print("---Ollama Question 1---")
# In your terminal, run the following prompt using Ollama (you installed it during the lesson):
prompt = f"""
Explain what a large language model is in two sentences.
"""

responses = get_completion(prompt, model="gpt-4o-mini")
print(responses)

# OpenAi:
# A large language model is an artificial intelligence system designed to understand
# and generate human-like text by analyzing vast amounts of written data.
# It uses deep learning techniques, particularly neural networks,
# to predict and produce coherent and contextually relevant language based on the input it receives.

# Ollama: not enough space on disk


