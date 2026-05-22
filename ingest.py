"""
RAG STEP 1 — INGEST
====================
This script prepares your documents for RAG (Retrieval Augmented Generation).

What it does:
  1. Create an index in Azure AI Search  (defines the structure / schema)
  2. Read PDFs → split into small text chunks
  3. Convert each chunk into an embedding  (a list of numbers = meaning)
  4. Upload chunks + embeddings to Azure AI Search

Run once before using main.py.
"""

import os
import re
from pathlib import Path

import pypdf
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# =============================================================================
# CONFIGURATION  (values come from .env)
# =============================================================================

# Azure AI Foundry gives a project-level URL like:
#   https://myaccount.services.ai.azure.com/api/projects/myproject
# The OpenAI SDK needs only the account-level part, so we strip the rest.
OPENAI_ENDPOINT = re.sub(r"/api/projects/.*", "", os.environ["AZURE_FOUNDRY_ENDPOINT"])
OPENAI_API_KEY  = os.environ["AZURE_OPENAI_API_KEY"]
EMBED_MODEL     = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]  # e.g. text-embedding-ada-002

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_INDEX    = os.environ["AZURE_SEARCH_INDEX"]
SEARCH_KEY      = os.environ["AZURE_SEARCH_API_KEY"]

DOCS_FOLDER   = Path(__file__).parent / "benefitdocs"
CHUNK_SIZE    = 800   # max characters per chunk
CHUNK_OVERLAP = 150   # overlap keeps context at chunk boundaries

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_pdf(path: Path) -> str:
    """Extract all text from a PDF file."""
    reader = pypdf.PdfReader(str(path))
    return " ".join(page.extract_text() or "" for page in reader.pages)


def split_into_chunks(text: str) -> list[str]:
    """Split a long text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_embedding(text: str) -> list[float]:
    """Turn text into a list of numbers (embedding) using Azure OpenAI."""
    result = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return result.data[0].embedding


# =============================================================================
# CREATE CLIENTS
# =============================================================================

openai_client = AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY, api_version="2024-10-21")
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX, credential=AzureKeyCredential(SEARCH_KEY))
index_client  = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))


# =============================================================================
# STEP 1 — Create the Azure AI Search index
#
# CONCEPT: An index is like a database table.  We define:
#   - Regular text fields  (id, title, content, category)
#   - A vector field        (content_vector)  — stores the embedding numbers
#   - A vector profile      tells the index to use HNSW algorithm for fast search
# =============================================================================
print("\nSTEP 1: Creating Azure AI Search index...")

index = SearchIndex(
    name=SEARCH_INDEX,
    fields=[
        SimpleField(name="id",       type=SearchFieldDataType.String, key=True),
        SearchField(name="title",    type=SearchFieldDataType.String, searchable=True),
        SearchField(name="content",  type=SearchFieldDataType.String, searchable=True),
        SearchField(name="category", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,        # text-embedding-ada-002 produces 1536 numbers
            vector_search_profile_name="hnsw-profile",
        ),
    ],
    vector_search=VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-algo")],
    ),
)

index_client.create_or_update_index(index)
print(f"  Index '{SEARCH_INDEX}' is ready.")


# =============================================================================
# STEP 2 — Read PDFs and split into chunks
#
# CONCEPT: We can't send an entire document to the model — it's too long.
#          We split it into small overlapping pieces called "chunks".
#          Overlap ensures a sentence cut at a boundary isn't lost.
# =============================================================================
print("\nSTEP 2: Reading PDFs and splitting into chunks...")

all_chunks = []   # will hold every chunk across all PDFs

for pdf_file in sorted(DOCS_FOLDER.glob("*.pdf")):
    text   = read_pdf(pdf_file)
    chunks = split_into_chunks(text)
    source = pdf_file.stem.replace("_", " ")

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "id":       f"{pdf_file.stem}_{i}",
            "title":    f"{source} (chunk {i + 1})",
            "content":  chunk,
            "category": pdf_file.stem,
        })

    print(f"  {pdf_file.name}  →  {len(chunks)} chunks")

print(f"  Total: {len(all_chunks)} chunks across all PDFs")


# =============================================================================
# STEP 3 — Generate embeddings
#
# CONCEPT: An embedding converts text into a list of ~1536 numbers.
#          Similar meaning → similar numbers → close together in vector space.
#          This lets us find relevant chunks by MEANING, not just keywords.
# =============================================================================
print("\nSTEP 3: Generating embeddings (this takes a few minutes)...")

# Show what an embedding looks like — just once, so students can see it
sample_embedding = get_embedding(all_chunks[0]["content"])
print(f"  What an embedding looks like:")
print(f"    dimensions : {len(sample_embedding)}")
print(f"    first 5 values: {[round(v, 4) for v in sample_embedding[:5]]}")
print()

# Generate embeddings for every chunk
for i, chunk in enumerate(all_chunks):
    chunk["content_vector"] = get_embedding(chunk["content"])
    # Print progress every 50 chunks
    if (i + 1) % 50 == 0 or (i + 1) == len(all_chunks):
        print(f"  {i + 1}/{len(all_chunks)} embeddings generated")


# =============================================================================
# STEP 4 — Upload to Azure AI Search
#
# CONCEPT: We upload in batches of 50 because Azure Search has a per-request limit.
#          Each document contains both the text AND its embedding.
# =============================================================================
print("\nSTEP 4: Uploading chunks + embeddings to Azure AI Search...")

BATCH_SIZE = 50
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i : i + BATCH_SIZE]
    search_client.upload_documents(batch)
    print(f"  Uploaded {min(i + BATCH_SIZE, len(all_chunks))} / {len(all_chunks)}")

print(f"\nDone! {len(all_chunks)} chunks are now stored in the '{SEARCH_INDEX}' index.")
print("You can now run main.py to ask questions.")
