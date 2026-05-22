# RAG Agent — Microsoft Agent Framework + Azure OpenAI + Azure AI Search

A minimal RAG agent built with the [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/agents/rag?pivots=programming-language-python).

| Component | Service |
|-----------|---------|
| LLM | Azure OpenAI (via Azure AI Foundry) |
| Embeddings | Azure OpenAI (`text-embedding-ada-002`) |
| Knowledge base | Azure AI Search (hybrid vector search) |

## Project structure

```
RAG_Agent_Framework/
├── main.py          # RAG agent — run this
├── ingest.py        # optional: upload sample docs to the index
├── requirements.txt
├── .env.example     # copy to .env and fill in your values
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Copy `.env.example` to `.env` and fill in your values:

```bash
copy .env.example .env   # Windows
```

| Variable | Description |
|----------|-------------|
| `AZURE_FOUNDRY_ENDPOINT` | Your Azure AI Foundry project endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name (e.g. `gpt-4o`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name (e.g. `text-embedding-ada-002`) |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint |
| `AZURE_SEARCH_INDEX` | Name of your search index |
| `AZURE_OPENAI_API_KEY` | *(optional)* leave blank to use `DefaultAzureCredential` |
| `AZURE_SEARCH_API_KEY` | *(optional)* leave blank to use `DefaultAzureCredential` |

### 3. (Optional) Populate the index with sample data
```bash
python ingest.py
```
This creates the index and uploads 3 sample articles. Skip if your index already has data.

### 4. Run the agent
```bash
python main.py
```

Type a question and the agent will search the knowledge base before answering.

## How it works

1. `AzureAISearchCollection` (Semantic Kernel) connects to your Azure AI Search index.  
2. `create_search_function()` wraps the collection as a callable tool with hybrid keyword + vector search.  
3. `.as_agent_framework_tool()` bridges it into the Agent Framework tool protocol.  
4. `AzureOpenAIChatClient` connects to your Azure OpenAI deployment via Foundry.  
5. The agent automatically calls the search tool, retrieves relevant chunks, and includes them in its answer.
