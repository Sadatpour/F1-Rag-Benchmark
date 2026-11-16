# Apex F1 RAG Benchmark

A Streamlit dashboard for building a RAG application over independent Wikipedia and Jolpica corpora, comparing OpenRouter models, and benchmarking vector stores. The app reads only `knowledge_base/` and does not depend on the original `data/` or `pipeline/` folders.

## Project structure

```text
app.py                              # Streamlit dashboard entry point
src/f1_rag/                         # Ingestion, chunking, embeddings, retrieval, evaluation
config/models.toml                  # Central LLM and embedding model configuration
knowledge_base/
  corpora/wikipedia.jsonl           # Self-contained Wikipedia corpus
  corpora/jolpica.jsonl.gz          # Compressed Jolpica corpus (GitHub file-size safe)
  evaluation/qa.jsonl               # Reference QA evaluation set
  indices/                          # Generated local indexes (ignored by Git)
models/
  embedding/all-MiniLM-L6-v2/       # Downloaded local embedding model (ignored by Git)
  llm/                              # OpenRouter-hosted models; no local weights
.streamlit/secrets.toml             # Private OpenRouter key (never commit)
```

The original `data/` and `pipeline/` folders were used only to prepare the independent knowledge base. They have been removed from this workspace; the dashboard does not need them.

## Windows setup

Open PowerShell in the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-embedding.txt
```

The generated corpora and evaluation set are already bundled in `knowledge_base/`. To rebuild them, first restore the original `data/raw/wikipedia`, `data/raw/jolpica`, and `data/final/qa.jsonl` inputs, then run `python -m f1_rag.ingest`.

### OpenRouter API key

Copy the example secrets file and add your key locally:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Set the key under `[openrouter]`. Alternatively, define `OPENROUTER_API_KEY` in your environment. Both options are excluded from Git.

The answer model defaults to `cohere/north-mini-code:free`; the free judge defaults to `nvidia/nemotron-3.5-lightning:free`. Edit `config/models.toml` to change answer, judge, or query-rewrite models. Free model availability and context windows are checked against the OpenRouter catalog.

### Embedding model

The app loads the local `sentence-transformers/all-MiniLM-L6-v2` snapshot from `models/embedding/all-MiniLM-L6-v2/`. Download it with:

```powershell
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/embedding/all-MiniLM-L6-v2
```

The model files are ignored by Git. The model ID, snapshot path, and weights path are configured in `config/models.toml`.

### Run the dashboard

```powershell
streamlit run app.py
```

## Features

- Select Wikipedia or Jolpica as the knowledge source.
- Compare answers across multiple LLMs and three vector stores: exact NumPy cosine, FAISS, and ChromaDB.
- Configure embedding-token chunk sizes (128, 192, or 224), token overlap, and 1–10 retrieved chunks. The chunks use the embedding model tokenizer and stay below its 256-token input limit.
- Optionally rewrite the query before embedding.
- Toggle answer judging on or off, and choose a separate judge model when enabled.
- Measure reference-source hit@10 and source-level NDCG@10, and estimate QA accuracy on a stratified sample of 10–100 questions.
- Build indexes by streaming corpus records and embedding batches with visible progress. Wikipedia remains the default source for a smaller first index.
- View each configured model's context window when OpenRouter provides it.

### Jolpica attribution

Whenever Jolpica is selected, the dashboard identifies the data source as the [Jolpica F1 API (Ergast-compatible)](https://api.jolpi.ca/ergast/f1/). Retrieved Jolpica records include a direct API endpoint link; Wikipedia records include canonical article links. Source names and URLs are also supplied to the answer and judge models.

### Optional vector stores

NumPy exact cosine is the default and does not require an external service. Install FAISS and ChromaDB for comparisons:

```powershell
python -m pip install -r requirements-vectorstores.txt
```

If the FAISS wheel is incompatible with your Python/Windows setup, use NumPy or ChromaDB.

Generated indexes are rebuildable, stored under `knowledge_base/indices/`, and excluded from Git. The self-contained corpus and QA files remain in the repository so it can be published without the original raw data.

> **Evaluation note:** Each QA row contains one source label rather than a complete set of relevant document or chunk IDs. The reported reference-source hit rate checks whether that source appears in the first 10 results, and NDCG deduplicates results at the source level. Neither is standard document/chunk Recall@10 or NDCG@10. A judged qrels file with all relevant IDs per question is needed for strict information-retrieval metrics. QA accuracy is a stratified sample estimate, not a corpus-wide score; LLM judgments should be calibrated with human review.

> **Context and prompt safety:** Answer context is budgeted conservatively using UTF-8 byte counts, reserving room for system instructions, the question, and the maximum answer. Retrieved documents are explicitly treated as untrusted data, and citations are limited to references included in the prompt.

## Tests

```powershell
python -m unittest discover -s tests -v
```
