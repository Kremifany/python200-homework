from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


print("------Concepts Question 1------")
# A: For Scenario A it's RAG because they have a lot of PDFs that get updated every
#    quarter, so we can refresh the document store without retraining the model.
# B: For Scenario B I would use the fine tuning approach, because the goal is a specific
#    brand voice (a style) that barely appears online, and they have 3,000 examples -
#    enough data to train the model to internalize that voice.
# C: For Scenario C the Prompt engineering approach is perfect, because the report is only
#    two pages and needed just once, so it fits in the prompt and RAG would be overkill.
print("-------Concepts Question 2-------")
# When model responses confidently and not saying "I am not sure" each time we tempting to believe the model responses 
# and not double check the facts that might be wrong and be harmful for the system that relies on those answers for 
# the further work and decision makes wrong.
#  For example some kind of facts from previous readings of sensor that the data about them is not present for 
# the model and it will hallucinates the response that we will take and make decisions using that fabricated data.
#  The tone totally affects our judging as humans and not machines. the more confident tone we tempt to take 
# the response of the model for granted and not double check.
print("------Concepts Question 3------")
steps = [
    "Generate a response from the LLM",
    
    "Inject retrieved chunks into the prompt",
   
]

# Correrect order and each step description:
# Indexing phase:
#  1. Extract text from source documents: 
#     Read the raw text out of the PDFs/files so it can be processed
#  2. Split text into chunks: 
#     Break each document into smaller passages so retrieval can return focused pieces
#  3. Convert text chunks into embeddings: 
#     Turn each chunk into a numeric vector that captures its meaning, and store it
# Query phase:
#  4. Receive the user's query:
#     Taking in the question from user
#  5. Embed the user's query: 
#     Once the FAISS index is created, the given query is converted into an embedding using the embedding model
# Retrieve phase:
#  6. Retrieve the most relevant chunks: 
#     and then input to the FAISS index's search function. 
#     The search function computes the cosine similarity between the query embedding and 
#     the chunk embeddings in the index and returns the k highest scoring (most relevant) chunks. 
# Augment phase:
#  7. Inject retrieved chunks into the prompt: 
#     Augment the context with the query and input the modified prompt to the LLM 
#  8. Generate a response from the LLM: 
#     The model aswer the question that was augmented with the context

print("-------Keyword RAG--------")
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

print("-------Keyword Question 1------")
# Run simple_keyword_retrieval with verbose=True on the query and documents below.
# Print the name of the selected document.

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

def simple_keyword_retrieval(query, documents, verbose=True):
    """
    Keyword retrieval using token overlap scoring.
    - Removes stopwords and punctuation for cleaner matching.
    - Returns the single best-matching document.
    - `documents`: dictionary with the document names as keys and the text as values, extracted using the `extract_text_from_pdf` function.
    """
    import string

    stopwords = [
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    ]

    # Translator to remove punctuation (so "Solar?" -> "Solar")
    translator = str.maketrans("", "", string.punctuation)

    # Tokenize query: lowercase, remove punctuation and stopwords
    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        # Tokenize document: lowercase, remove punctuation and stopwords
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }

        # Compute simple overlap score
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))

        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    # Sort by overlap score (descending)
    scores.sort(reverse=True)

    # Pick the single best match (if score > 0)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


simple_keyword_retrieval(query, documents, verbose=True)

# OUTPUT:
# API key loaded successfully.
# ------Concepts Question 1------
# -------Concepts Question 2-------
# ------Concepts Question 3------
# -------Keyword RAG--------
# -------Keyword Question 1------

# Query tokens (filtered): ['hours', 'weekends', 'what', 'your']
# [menu.txt] overlap=0 -> []
# [hours.txt] overlap=1 -> ['weekends']
# [hiring.txt] overlap=1 -> ['your']
# [loyalty.txt] overlap=1 -> ['your']

# Selected best match: loyalty.txt
# After running the function, add a comment explaining which document was selected and why.
# A: The document loyalty.txt was selected because the overlap=1 -> 'your' and the name
#    alphabetically was before the others in function sort reverse-alphabetical and not based
#    on true relevance


print("-------Keyword Question 2------")
query = "Do you have anything without caffeine?"

simple_keyword_retrieval(query, documents, verbose=True)
# OUTPUT:
# Which document was selected: no document was selected
# Whether keyword RAG got this right — and why or why not?
# A: Yes he got it right, there are no overlaps
# What kind of retrieval would do better here?
# A: I thinks semantic keyword will do better here

print("-------Keyword Question 3------")
query = "How do I sign up for rewards?"
# Before running any code, predict which document will be selected for the query below. 
# Write your prediction and your reasoning as a comment first, then run the code to check.
# A:Without even running the code there are no overlaps of words between query and documents
simple_keyword_retrieval(query, documents, verbose=True)
# Yes my prediction was correct there are no overlaps between the query and documents text
print("-------Semantic RAG Concepts-------\n")
print("-------Semantic Question 1-------\n")
# What is a vector embedding? (1-2 sentences)
# A: vector embedding it's a vector representation of data, the closer in meaning the words are - 
#   the numbers representing them in a vector are closer in high dimentional space 
# Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query.
#  Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
# A: the chunk with higher similarity score is closer in meaning with a given query then the chunk with the lower cosine number
# Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
# A: Because semantic search not looks for the similar words but for the closer meaning that the words create

print("-------Semantic Question 2-------\n")
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | meaning that words create , cosine of vectors, embedding of query and embedding of document|
# | What is retrieved?         | Full document                     | most relevant text chunks - that the vectors of those are the closest                      |
# | Can it handle synonyms?    | No                                | yes           |
# | Storage format             | Plain text dictionary             | structure to store vectors - FAISS  and chunk text         |
# | Relevance score            | Number of overlapping keywords    | how close cosine of embeddings are          |

print("---------LlamaIndex--------\n")
print("---------LlamaIndex Question 1--------")

# Build an in-memory LlamaIndex pipeline using the Brightleaf Solar PDFs and run the two queries below.
# For each query, print:
# The question
# The answer from the model
# For each of the 3 retrieved source nodes: the similarity score and the first 150 characters of the chunk text
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file import PDFReader

pdfs_dir = Path(__file__).parent / "resources" / "brightleaf_pdfs"
assert pdfs_dir.exists(), f"Pdf directory not found: {pdfs_dir}."



file_extractor = {".pdf": PDFReader()}
docs = SimpleDirectoryReader(str(pdfs_dir), file_extractor=file_extractor).load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
# Use similarity_top_k=3. After printing the results, add a comment for each query answering:
# Do the retrieved chunks look relevant to the question?
# Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context"
# or "I'm not sure"? Note what you observe about the tone.
# Did anything unexpected get retrieved?

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text: {node_with_score.node.get_content()[:150]}")
        print("-" * 30)

# Q1 observations:
#  A: BrightLeaf offers a comprehensive benefits program that includes health, vision, and wellness benefits such as medical insurance, vision benefits, wellness programs, and a Wellness Reimbursement Plan. Additionally, the company provides financial security and retirement benefits like life insurance, disability insurance, and a 401(k) retirement plan with a company match. BrightLeaf also offers parental leave, work flexibility, professional development opportunities, mentorship programs, and access to free online courses through the Learning Hub.
# Similarity Score: 0.9083
# Text: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m
# ------------------------------
# Similarity Score: 0.8152
# Text: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract
# ------------------------------
# Similarity Score: 0.8120
# Text: Network and Data Security
# BrightLeaf maintains layered defenses for all production and corporate networks. Access to critical
# systems requires multi■f
# ------------------------------
# A: the first chunk from document employee_benefits.pdf very specific about benefits
#  the other 2 are not so relevant to the query


# OUTPUT
# A: BrightLeaf maintains layered defenses for all production and corporate networks. Access to critical systems requires multi-factor authentication and VPN with device certificates. Credentials are rotated every 90 days and revoked immediately upon role change. Customer data is encrypted in transit (TLS 1.3) and at rest (AES-256) with keys stored in a managed HSM. Perimeter firewalls and cloud security groups enforce least privilege, and logs are centralized with anomaly detection tuned to privilege escalation and data-exfiltration signatures. The incident response plan follows NIST 800-61 guidance, including preparation, identification, containment, eradication, recovery, and post-incident review. Employee training includes onboarding security training, annual refreshers, phishing simulations, password hygiene, and safe data handling. Role-based access control is reviewed quarterly, and contractors use time-bound accounts. Vendors must pass a security questionnaire, and third-party integrations are isolated. Compliance aligns with ISO 2700e networks. Access to critical
# systems requires multi■f
# ------------------------------
# Similarity Score: 0.8384
# Text: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m
# ------------------------------
# Similarity Score: 0.8208
# Text: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract
# ------------------------------

# Q2 observations:
# First chunk is the most relevantand it from security_policy.pdf with strong similarity (0.88).
# the other 2 responses not so relevant to the question

# in two cases the model sounds confident


print("---------LlamaIndex Question 2--------")
# Re-run one of the queries from Q1 twice: once with similarity_top_k=1 and once with similarity_top_k=5.
# Print the response and source node scores for both runs.
# Add a comment explaining how the response changed (if at all) and whether more retrieved context is always better.

query = "What employee benefits does BrightLeaf offer?"

print(f"\n--- similarity_top_k=1---")
print(f"Q: {query}")
query_engine = index.as_query_engine(similarity_top_k=1)
response = query_engine.query(query)
print("A:", response)
for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text: {node_with_score.node.get_content()[:150]}")
    print("-" * 30)

print(f"\n--- similarity_top_k=5---")
print(f"Q: {query}")
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(query)
print("A:", response)
for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text: {node_with_score.node.get_content()[:150]}")
    print("-" * 30)

# k=1 returned a shorter but very accurate answer using only the best chunk (employee_benefits.pdf).
# OUTPUT:
# --- similarity_top_k=1---
# Q: What employee benefits does BrightLeaf offer?
# A: BrightLeaf Solar offers a comprehensive benefits package to its employees, including health benefits such as medical insurance, vision benefits, and wellness programs. Additionally, the company provides financial security through life, disability, and retirement benefits. BrightLeaf also supports work-life balance with parental leave, flexible scheduling, and hybrid work options. Furthermore, the company invests in employee learning and growth through professional development stipends, mentorship programs, and access to online courses.
# Similarity Score: 0.9083
# Text: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m
# ------------------------------


# k=5 returned a longer, more detailed answer, but also pulled in less relevant chunks
# (mission_statement, security_policy, partnerships, earnings_report).
# More retrieved context is not always better.
# OUTPUT
# --- similarity_top_k=5---
# Q: What employee benefits does BrightLeaf offer?
# A: BrightLeaf offers a comprehensive benefits program that includes health insurance with various services, vision benefits, wellness programs, a Wellness Reimbursement Plan, life, disability, and retirement benefits, a 401(k) plan with company match, parental leave, work flexibility options, professional development stipend, internal mentorship network, Diversity, Equity, and Inclusion Council, and free access to curated online courses for continuous education.
# Similarity Score: 0.9083
# Text: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m
# ------------------------------
# Similarity Score: 0.8152
# Text: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract
# ------------------------------
# Similarity Score: 0.8120
# Text: Network and Data Security
# BrightLeaf maintains layered defenses for all production and corporate networks. Access to critical
# systems requires multi■f
# ------------------------------
# Similarity Score: 0.8072
# Text: EcoVolt Energy (2022 Partnership)
# BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid
# solutions to ru
# ------------------------------
# Similarity Score: 0.7861
# Text: Overview
# This report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The period
# includes a growth phase, a temporary dip i
# ------------------------------

print("---------LlamaIndex Question 3--------")
# Try a query you think the pipeline might struggle with — something vague, something that spans multiple documents, 
# or something where the information might not be in the documents at all. Print the response and all retrieved chunks.
# Add a comment explaining what you expected, what actually happened,
# and what you would change about the system to handle this kind of query better.

query="What is the name of brightleaf means?"
print(f"\n---Vague question similarity_top_k=5---")
print(f"Q: {query}")
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(query)
print("A:", response)
for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text: {node_with_score.node.get_content()[:150]}")
    print("-" * 30)

# OUTPUT
# ---Vague question similarity_top_k=5---
# Q: What is the name of brightleaf means?
# 2026-06-03 21:12:34,307 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
# 2026-06-03 21:12:35,634 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
# A: BrightLeaf Solar.
# Similarity Score: 0.8156
# Text: Overview
# BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our
# mission is to make solar power pract
# ------------------------------
# Similarity Score: 0.7957
# Text: Introduction
# BrightLeaf Solar views employee well-being as inseparable from long-term innovation. Our benefits
# program is designed to help each team m
# ------------------------------
# Similarity Score: 0.7911
# Text: EcoVolt Energy (2022 Partnership)
# BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid
# solutions to ru
# ------------------------------
# Similarity Score: 0.7828
# Text: Network and Data Security
# BrightLeaf maintains layered defenses for all production and corporate networks. Access to critical
# systems requires multi■f
# ------------------------------
# Similarity Score: 0.7742
# Text: Overview
# This report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The period
# includes a growth phase, a temporary dip i
# ------------------------------

# A: I've expected the explanation about the name of the company but even with the high similarity score
#   of 0.8156 the answer from model  was not so related to the question

print("---------LlamaIndex Question 4--------")
# lamaIndex Question 4
# Using the same index and query engine you built in Q1, evaluate one response using LlamaIndex's built-in evaluators.

# Import and instantiate a FaithfulnessEvaluator and a RelevancyEvaluator, both using gpt-4o-mini as 
# the judge LLM (refer to the "RAG Evaluation using LlamaIndex" 
# section of lesson 4 for the exact import and setup pattern). Run them on this query:
# q = "What employee benefits does BrightLeaf offer?"
# Print both scores. Then run the evaluators again on a query you expect to produce a lower-quality
# response — for example, a question about something that is clearly not in the Brightleaf documents.

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini")
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

query_engine = index.as_query_engine(similarity_top_k=3)

def evaluate_query(q):
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    faithfulness_result = faithfulness_evaluator.evaluate_response(response=response)
    relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)

    print(f"Faithfulness passing: {faithfulness_result.passing} (score: {faithfulness_result.score})")
    print(f"Relevancy passing: {relevancy_result.passing} (score: {relevancy_result.score})")


# High-quality response: the answer IS in the BrightLeaf documents
evaluate_query("What employee benefits does BrightLeaf offer?")

# Lower-quality response: this is clearly NOT in the BrightLeaf documents
evaluate_query("What is the capital of France and who won the 2018 World Cup?")

# Q4 observations:
# A: Faithfulness measures whether the answer is supported by the retrieved chunks (no hallucination),
#    Relevancy measures whether the answer + retrieved chunks actually address the question.
#    For the benefits query both evaluators pass (the answer is grounded in and relevant to the docs).
#    For the off-topic query the relevancy/faithfulness scores drop because the retrieved chunks
#    don't contain the answer, so the model either hedges or pulls in unrelated context.  
#    we cant evaluate hallucinations with simple scalar accuracy but with llm judgimnet we can.
#    also traditional metrics miss meaning that semantic RAG can achieve
