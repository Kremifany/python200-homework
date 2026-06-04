# =====================================================================
# Mini-Project: Groundwork Coffee Co. Q&A Assistant
# =====================================================================

from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

print("Step 1: Setup")
docs_dir = Path(__file__).parent / "resources" / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print("Step 2: Load the Documents")
docs = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Loaded {len(docs)} document(s).")
for doc in docs:
    print(doc.metadata["file_name"])

print("Step 3: Build the Index and Query Engine")
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")

print("Step 4: Query the Assistant")


def ask(question):
    print(f"\nQ: {question}")
    response = query_engine.query(question)
    print("A:", response)
    if response.source_nodes:
        top = response.source_nodes[0]
        print(f"Source document: {top.node.metadata['file_name']}")
        print(f"Similarity Score: {top.score:.4f}")
        print(f"Text: {top.node.get_content()[:200]}")
    return response


questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for q in questions:
    ask(q)


# Reflection after running all five queries:
# Did the assistant sound confident and accurate? Did any of the answers surprise you?
# A: Assistant sounded confident and accurate with the answers but not the sources
# that was suprising that the file name of the source was wrong

# Q1: What are Groundwork's hours on weekends?

# A1: Groundwork's hours on weekends are 8:00 AM to 5:00 PM.
# Source document: our_story.txt
# Similarity Score: 0.8102
# Text: Our Story

# Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala. S

# Reflection1: The answer is right but the source shown mistakenly our_story.txt it should be faq.txt
# ---------
# Q2: Do you offer any dairy-free milk options?
# A2: All dairy-free options are available at no extra charge, including oat milk, almond milk, and soy milk.
# Source document: seasonal_specials.txt
# Similarity Score: 0.7843
# Text: Seasonal Specials — Current Menu

# These drinks are available for a limited time only.

# Iced Lavender Lemonade — $5.00
# Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dairy-fre

# Reflection2: The fact retrieved in answer is accurate but taken froom different file, not from 
# seasonal_specials.txt but from menu.txt
#----------

# Q3: How does the loyalty program work?
# A3: The loyalty program is free to join and allows customers to earn one point for every dollar spent. Once a customer accumulates 100 points, they can redeem them for any free drink on the menu. Customers can sign up for the loyalty program either at the register or on the company's website.
# Source document: faq.txt
# Similarity Score: 0.7625
# Text: Frequently Asked Questions

# Hours
# - Monday through Friday: 7:00 AM to 7:00 PM
# - Saturday and Sunday: 8:00 AM to 5:00 PM
# - We are closed on Thanksgiving Day and Christmas Day.

# Locations
# - Downtown: 42

# Reflection3: Very accurate answer and true to the source 
# -----------

# Q4: How did Groundwork Coffee get started?
# A4: Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala, while Sam had managed a community center in his hometown. They believed in the connection between good coffee and strong communities, leading them to establish Groundwork with a commitment to sourcing only fair-trade, sustainably grown beans directly from small farms.
# Source document: our_story.txt
# Similarity Score: 0.9004
# Text: Our Story

# Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala. S

# Reflection4: Very accurate answer and true to the source 
# -----------

# Q5: Do you offer catering or wholesale orders?
# A5: Yes, we offer both catering and wholesale orders.
# Source document: wholesale_catering.txt
# Similarity Score: 0.8578
# Text: Wholesale and Catering

# Wholesale Coffee
# We sell our house blends and single-origin beans in bulk to local restaurants, offices, and retailers. 
# Wholesale pricing is available for orders of 5 pounds or

# Reflection5: Very accurate answer and true to the source but not complete 100%
# -----------

print("Step 5: Find a Failure")

def ask_all_sources(question):
    """Print full answer and all retrieved source nodes (similarity_top_k=3)."""
    print(f"\nQ: {question}")
    response = query_engine.query(question)
    print("A:", response)
    for node_with_score in response.source_nodes:
        print(f"Source document: {node_with_score.node.metadata['file_name']}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text: {node_with_score.node.get_content()[:200]}")
        print("-" * 30)
    return response

failure_question = (
    "Can I redeem loyalty points toward a catering order for 50 people, "
    "and what advance notice and email should I use to book it?"
)
ask_all_sources(failure_question)


# OUTPUT:
# Q: Can I redeem loyalty points toward a catering order for 50 people, and what advance notice and email should I use to book it?  
# A: You cannot redeem loyalty points toward a catering order for 50 people. To book a catering order for 50 people, you should email hello@groundworkcoffee.com with your event date, location, estimated guest count, and preferred package. 
# Catering orders require at least 72 hours of advance notice.
# Source document: wholesale_catering.txt
# Similarity Score: 0.7864
# Text: Wholesale and Catering

# Wholesale Coffee
# We sell our house blends and single-origin beans in bulk to local restaurants, offices, and retailers. Wholesale pricing is available for orders of 5 pounds or
# ------------------------------
# Source document: faq.txt
# Similarity Score: 0.7493
# Text: Frequently Asked Questions

# Hours
# - Monday through Friday: 7:00 AM to 7:00 PM
# - Saturday and Sunday: 8:00 AM to 5:00 PM
# - We are closed on Thanksgiving Day and Christmas Day.

# Locations
# - Downtown: 42
# ------------------------------
# Source document: seasonal_specials.txt
# Similarity Score: 0.7177
# Text: Seasonal Specials — Current Menu

# These drinks are available for a limited time only.

# Iced Lavender Lemonade — $5.00
# Freshly squeezed lemonade with lavender syrup and a splash of cold brew. Dairy-fre
# ------------------------------




# Step 5 analysis:
# What I asked and why I expected it to be hard:
# A: I asked whether loyalty points work on catering for 50 guests plus booking details.
#    This requires combining faq.txt and wholesale_catering.txt
#    The docs describe each topic separately but never say whether points apply to
#    catering orders.
#
# What went wrong:
# A: Model partially answwerd wrong but confident that the loyalty points cannot be applied towards the catering but the documents does not sdupport that.
#
# Tone when retrieval was weak:
# A: The model sounded fully confident when the information was not so right and mostly not true and not supported by any document
#   I suggest always double check the answers with the docs

# What I would change to improve the system:
# A: I would instruct the model with the system prompt to answer only when it supported by context, or say that the answer for the question not specified in docs.
#   Also I would increase similarity_top_k, I woud run FaithfulnessEvaluator before sowing the answers. 


print("Step 6: Reflection\n")
# The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code.
# How many lines did the equivalent LlamaIndex implementation take in your project?
# What does that tell you about the value of using a framework?
# A: The manual pipeline (read PDFs, chunk, embed, build FAISS, retrieve) is roughly
#    100+ lines. In this project the same core work is about 4 lines: load_data(),
#    VectorStoreIndex.from_documents(docs), and as_query_engine(). LlamaIndex handles chunking,
#    embedding, and the in-memory vector store out of the box. That shows a framework saves time,
#    reduces bugs, and lets you focus on the use case (questions, evaluation) instead of plumbing.
#
# Describe a different use case — not a coffee shop — where this approach would add genuine value.
# What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?
# A: A similar system for immigration law would add real value: many long forms, policy memos, and
#    FAQs that clients and staff need to search without reading every file. Staff could ask plain-
#    language questions and get answers grounded in the current document set.
#
#    One failure mode RAG cannot fully prevent: the model can still sound confident while being wrong
#    (as in Step 5 — it said loyalty points cannot be used for catering even though the docs are
#    silent on that). Good retrieval does not stop hallucination or overconfident guessing. Another
#    issue is stale knowledge when laws change often — unless you re-index regularly, answers can
#    reflect old documents even when retrieval returns the "best" matching chunks.
