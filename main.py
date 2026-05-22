"""
RAG STEP 2 — ASK QUESTIONS
===========================
This is the RAG (Retrieval Augmented Generation) chat loop.

The RAG pattern — for every question:
  1. EMBED   the question  (turn it into numbers)
  2. SEARCH  Azure AI Search for the most similar chunks
  3. PROMPT  send the chunks as context to the LLM
  4. ANSWER  return the LLM's grounded response

Run this after ingest.py has populated the index.
"""

import os
import re

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENAI_ENDPOINT  = re.sub(r"/api/projects/.*", "", os.environ["AZURE_FOUNDRY_ENDPOINT"])
OPENAI_API_KEY   = os.environ["AZURE_OPENAI_API_KEY"]
EMBED_MODEL      = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
CHAT_MODEL       = os.environ["AZURE_OPENAI_DEPLOYMENT"]

SEARCH_ENDPOINT  = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_INDEX     = os.environ["AZURE_SEARCH_INDEX"]
SEARCH_KEY       = os.environ["AZURE_SEARCH_API_KEY"]

# =============================================================================
# CREATE CLIENTS
# =============================================================================

openai_client = AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY, api_version="2024-10-21")
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX, credential=AzureKeyCredential(SEARCH_KEY))

# The system prompt tells the LLM how to behave
SYSTEM_PROMPT = """
You are a helpful HR assistant.
Answer questions using ONLY the context passages provided below.
If the answer is not in the context, say "I don't have that information."
Always mention which document your answer comes from.
"""


# =============================================================================
# RAG FUNCTIONS
# =============================================================================

def get_embedding(text: str) -> list[float]:
    """Step 1 — Convert text to a vector (list of numbers)."""
    result = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return result.data[0].embedding


def search_documents(question: str, top: int = 3) -> list[dict]:
    """Step 2 — Find the most relevant chunks in Azure AI Search."""
    question_vector = get_embedding(question)
    results = search_client.search(
        search_text=question,                                                    # keyword search
        vector_queries=[VectorizedQuery(vector=question_vector, fields="content_vector")],  # vector search
        select=["title", "category", "content"],
        top=top,
    )
    return [{"title": r["title"], "category": r["category"], "content": r["content"]} for r in results]


def ask(question: str) -> str:
    """Full RAG pipeline: search → build prompt → get answer."""

    # Step 2 — Retrieve relevant chunks
    chunks = search_documents(question)

    # Show what was retrieved (great for teaching — students see what the model will use)
    print("\n  [Retrieved chunks]")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk['title']}")
        print(f"     {chunk['content'][:100].strip()} ...")
    print()

    # Step 3 — Build the prompt: system message + retrieved context + question
    context = "\n\n".join(
        f"--- {c['title']} ---\n{c['content']}" for c in chunks
    )
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    # Step 4 — Ask the LLM
    response = openai_client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content


# =============================================================================
# MAIN LOOP
# =============================================================================

print("RAG Assistant ready. Type 'exit' to quit.")
print(f"  Model     : {CHAT_MODEL}")
print(f"  Embedding : {EMBED_MODEL}")
print(f"  Index     : {SEARCH_INDEX}")
print()

while True:
    question = input("You: ").strip()
    if not question or question.lower() in ("exit", "quit"):
        break

    answer = ask(question)
    print(f"Assistant: {answer}\n")
