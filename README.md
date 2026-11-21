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

### Provider API keys

Copy `.env.example` to `.env` and set only the keys for providers you use. The app loads `.env` without replacing environment variables that are already set:

```powershell
Copy-Item .env.example .env
```

Supported keys include `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY`, and `GOOGLE_API_KEY`. Provider, model ID, endpoint, protocol, and capabilities are separate in `config/models.toml`. Groq provides `openai/gpt-oss-20b` and `openai/gpt-oss-120b`; free-tier limits and model access can change. Provider usage limits are not a promise of zero-cost service. Never place key values in `config/models.toml`.

The answer model defaults to `cohere/north-mini-code:free`; the free judge defaults to `nvidia/nemotron-3.5-lightning:free`. Edit `config/models.toml` to change answer, judge, or query-rewrite models. Free model availability and context windows are checked against the OpenRouter catalog.

### Embedding model

The sidebar lets you select the embedding model. The default is `sentence-transformers/all-MiniLM-L6-v2`, loaded from `models/embedding/all-MiniLM-L6-v2/`. Add future local models under `[embedding.models]` in `config/models.toml`, then set `embedding.default_model` to the desired profile. Each profile defines its Hugging Face repo ID, local path, and optional safetensors path. The selected profile is used for query vectors, corpus vectors, retrieval evaluation, and benchmarks. Its ID and paths are part of the index fingerprint, so changing models builds or loads a separate compatible index.

Download the default model with:

```powershell
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/embedding/all-MiniLM-L6-v2
```

The reranker files are loaded from `models/reranker/ms-marco-MiniLM-L6-v2/`. The sidebar can rerank a candidate pool (default 30) and pass the final Top K to the answer model. Add profiles under `[reranking.models]` and set `[reranking].default_model` in `config/models.toml`. Model weights stay local and are excluded from Git. The configured MS MARCO model is English-focused, and its raw scores are ranking logits, not confidence probabilities.

The model files are ignored by Git. The model ID, snapshot path, and weights path are configured in `config/models.toml`.

### Run the dashboard

```powershell
streamlit run app.py
```

## Features

- Select Wikipedia or Jolpica as the knowledge source.
- Compare answers across multiple LLMs and three vector stores: exact NumPy cosine, FAISS, and ChromaDB.
- Use the sidebar to switch between asking questions, evaluation/benchmarking, and the project guide. Each evaluation mode has its own settings.
- Configure embedding-token chunk sizes (128, 192, or 224), token overlap, and Top K values 1, 3, and 5. The chunks use the embedding model tokenizer and stay below its 256-token input limit.
- Optionally rewrite the query before embedding.
- Toggle answer judging on or off, and choose a separate judge model when enabled.
- Measure source-level Recall@K and NDCG@K at selected K values, and estimate generated-answer accuracy on a stratified sample of 10–100 questions.
- Build indexes by streaming corpus records and embedding batches with visible progress. Wikipedia remains the default source for a smaller first index.
- View each configured model's context window when OpenRouter provides it.
- Compare Dense, BM25, and Hybrid (BM25 + dense) retrieval. Hybrid ranking uses Reciprocal Rank Fusion.
- Optionally rerank a candidate pool with a local CrossEncoder before sending the final Top K chunks to answer models. The configured MS MARCO MiniLM model scores query/chunk pairs; it does not replace the embedding model or retrieve documents on its own.
- Select direct Groq, OpenAI, xAI, Mistral, and Google models alongside OpenRouter models. Provider rotation follows `answer_rotation` in `config/models.toml`; quota, access, billing, model, network, and server failures move to the next compatible model. Failed models stay in a persisted cooldown to avoid repeating known failures on each request.
- Compare vector-store index build time, search p50/p95, process RAM/CPU, Recall@K, and NDCG@K with the same corpus vectors and questions.
- Tune chunk size, overlap, top-k, and vector store on a deterministic tuning split. Run a selected configuration once on the separate holdout split.
- Record settings, metrics, run IDs, and hardware/software details in `knowledge_base/runs/runs.jsonl`.
- Export every evaluation table as CSV or PDF. Evaluation results and their settings are saved in `knowledge_base/runs/evaluations.jsonl` and can be reopened from the saved test history.
- Show progress while retrieval, answer-accuracy, benchmark, tuning, and final holdout tests run; failed backends remain visible in partial results and saved run records.

### Jolpica attribution

Whenever Jolpica is selected, the dashboard identifies the data source as the [Jolpica F1 API (Ergast-compatible)](https://api.jolpi.ca/ergast/f1/). Retrieved Jolpica records include a direct API endpoint link; Wikipedia records include canonical article links. Source names and URLs are also supplied to the answer and judge models.

### Optional vector stores

NumPy exact cosine is the default and does not require an external service. Install FAISS and ChromaDB for comparisons:

```powershell
python -m pip install -r requirements-vectorstores.txt
```

If the FAISS wheel is incompatible with your Python/Windows setup, use NumPy or ChromaDB.

Generated indexes are rebuildable, stored under `knowledge_base/indices/`, and excluded from Git. The self-contained corpus and QA files remain in the repository so it can be published without the original raw data.

Process resource measurements use `psutil`, which is installed with the project requirements. If an older environment does not have it, install project requirements again with `python -m pip install -r requirements.txt`.

> **Evaluation note:** Each QA row contains one source label rather than a complete set of relevant document or chunk IDs. Recall@K and NDCG@K therefore use source-level labels; NDCG deduplicates results by source. These are not document/chunk-level relevance metrics. A judged qrels file with relevant IDs per question is needed for strict document or chunk Recall/NDCG. QA accuracy is a stratified sample estimate, not a corpus-wide score; calibrate LLM judgments with human review.

> **Context and prompt safety:** Answer context is budgeted conservatively using UTF-8 byte counts, reserving room for system instructions, the question, and the maximum answer. Retrieved documents are explicitly treated as untrusted data, and citations are limited to references included in the prompt.

## Tests

```powershell
python -m unittest discover -s tests -v
```
