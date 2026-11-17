from __future__ import annotations

import hashlib
import json
import logging
import sys
from itertools import islice
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
LOGGER = logging.getLogger(__name__)
sys.path.insert(0, str(ROOT / "src"))

from f1_rag.embeddings import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_WEIGHTS_PATH,
    load_embedding_model,
)
from f1_rag.evaluation import ndcg_at_n, qa_accuracy, recall_at_n, stratified_sample
from f1_rag.benchmark import BenchmarkConfig, benchmark_store, run_hyperparameter_grid
from f1_rag.ingest import KNOWLEDGE_BASE, iter_documents, resolve_corpus_path
from f1_rag.openrouter import (
    ADDITIONAL_MODELS,
    CONTEXT_WINDOW_OVERRIDES,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODELS,
    DEFAULT_REWRITE_MODEL,
    PROVIDER_ROUTER,
    answer_question,
    get_api_key,
    judge_answer,
    list_models,
    rewrite_query,
)
from f1_rag.retrieval import embed_documents, encode_query, load_evaluation_rows, prepare_index
from f1_rag.sources import (
    JOLPICA_API_BASE,
    JOLPICA_ATTRIBUTION,
    WIKIPEDIA_ARTICLE_BASE,
    jolpica_source_url,
    wikipedia_source_url,
)
from f1_rag.sparse import BM25Index, hybrid_search


st.set_page_config(
    page_title="Apex RAG | F1 Knowledge Lab",
    page_icon=":material/speed:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html(
    """
    <style>
    .hero { border:1px solid #402529; border-radius:24px; padding:2rem 2.2rem; margin:.4rem 0 1.6rem; background:linear-gradient(112deg,rgba(239,68,68,.16),rgba(21,23,28,.76) 58%); }
    .eyebrow { color:#f87171; text-transform:uppercase; letter-spacing:.18em; font-weight:700; font-size:.72rem; }
    .hero-title { font-size:2.55rem; font-weight:700; margin:.3rem 0; letter-spacing:-.05em; }
    .hero-copy { color:#b8bac2; max-width:760px; margin:0; }
    </style>
    <div class="hero">
      <div class="eyebrow">Formula 1 · Retrieval augmented generation</div>
      <div class="hero-title">Apex RAG <span style="color:#ef4444">Knowledge Lab</span></div>
      <p class="hero-copy">Compare knowledge sources, embeddings, vector stores, and answer models
      using an independent, reproducible Formula 1 RAG benchmark.</p>
    </div>
    """,
)


@st.cache_resource(show_spinner=False, max_entries=2)
def embedding_model(
    model_id: str,
    model_path: str,
    weights_path: str,
    weights_size: int,
    weights_mtime: int,
) -> Any:
    del weights_size, weights_mtime
    return load_embedding_model(model_id, Path(model_path), Path(weights_path))


def embedding_signature() -> tuple[str, str, str, int, int]:
    try:
        stat = EMBEDDING_WEIGHTS_PATH.stat()
        size, modified = stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        size, modified = 0, 0
    return (
        EMBEDDING_MODEL_ID,
        str(EMBEDDING_MODEL_PATH),
        str(EMBEDDING_WEIGHTS_PATH),
        size,
        modified,
    )


@st.cache_data(ttl=3600, show_spinner=False, max_entries=1)
def model_catalog() -> tuple[dict[str, dict[str, Any]], str | None]:
    try:
        return list_models(), None
    except (RuntimeError, TimeoutError) as error:
        return {}, str(error)


@st.cache_data(show_spinner=False, max_entries=1)
def evaluation_rows() -> list[dict[str, Any]]:
    return load_evaluation_rows()


@st.cache_data(show_spinner=False, max_entries=2)
def corpus_preview(dataset: str) -> str:
    docs = islice(iter_documents(dataset), 12)
    return "\n".join(
        f"{doc.get('source', '')} ({doc.get('source_url', '')}): "
        f"{doc.get('title', doc.get('text', '')[:300])}"
        for doc in docs
    )


@st.cache_resource(show_spinner=False, max_entries=4)
def embedded_corpus(
    dataset: str,
    model_id: str,
    model_path: str,
    weights_path: str,
    weights_size: int,
    weights_mtime: int,
    chunk_size: int,
    overlap: int,
    corpus_size: int,
    corpus_mtime: int,
    corpus_document_count: int,
) -> tuple[list[dict[str, Any]], Any]:
    embedder = embedding_model(model_id, model_path, weights_path, weights_size, weights_mtime)
    progress = st.progress(0, text=f"Embedding {dataset} documents in batches…")

    try:
        chunks, vectors = embed_documents(
            iter_documents(dataset),
            embedder,
            chunk_size,
            overlap,
            on_progress=lambda document_index: progress.progress(
                min(document_index / max(corpus_document_count, 1), 1.0),
                text=f"Embedded {document_index:,}/{corpus_document_count:,} documents…",
            ),
        )
    finally:
        progress.empty()

    if not chunks:
        raise ValueError(f"The {dataset} corpus did not produce any chunks.")
    return chunks, vectors


@st.cache_resource(show_spinner=False, max_entries=6)
def indexed_corpus(
    dataset: str,
    store_name: str,
    model_id: str,
    model_path: str,
    weights_path: str,
    weights_size: int,
    weights_mtime: int,
    chunk_size: int,
    overlap: int,
    corpus_size: int,
    corpus_mtime: int,
    corpus_document_count: int,
) -> tuple[Any, int, bool]:
    return prepare_index(
        dataset,
        store_name,
        chunk_size,
        overlap,
        embedder=None,
        precompute=lambda: embedded_corpus(
            dataset,
            model_id,
            model_path,
            weights_path,
            weights_size,
            weights_mtime,
            chunk_size,
            overlap,
            corpus_size,
            corpus_mtime,
            corpus_document_count,
        ),
    )


def _dataset_questions(dataset: str) -> list[dict[str, Any]]:
    rows = evaluation_rows()
    if dataset == "wikipedia":
        candidates = [row for row in rows if str(row.get("source", "")).startswith("Wikipedia:")]
    else:
        candidates = [row for row in rows if not str(row.get("source", "")).startswith("Wikipedia:")]
    sources = _corpus_sources(dataset)
    return [row for row in candidates if row.get("source") in sources]


def _evaluation_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Create a deterministic 80/20 tuning and holdout split from question text."""
    if split not in {"tune", "test"}:
        raise ValueError("split must be 'tune' or 'test'.")
    selected = []
    for row in rows:
        key = f"{row.get('source', '')}\0{row.get('question', '')}".encode("utf-8")
        is_test = int(hashlib.sha256(key).hexdigest()[:8], 16) % 5 == 0
        if is_test == (split == "test"):
            selected.append(row)
    return selected


@st.cache_data(show_spinner=False, max_entries=2)
def _corpus_sources(dataset: str) -> set[str]:
    path = resolve_corpus_path(dataset)
    sources = set()
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                source = json.loads(line).get("source")
                if source:
                    sources.add(str(source))
    return sources


def _ready_index(dataset: str, store_name: str, chunk_size: int, overlap: int) -> tuple[Any, int, bool]:
    path = resolve_corpus_path(dataset)
    stat = path.stat()
    return indexed_corpus(
        dataset,
        store_name,
        *embedding_signature(),
        chunk_size,
        overlap,
        stat.st_size,
        stat.st_mtime_ns,
        corpus_total,
    )


@st.cache_resource(show_spinner=False, max_entries=4)
def sparse_index(
    dataset: str,
    model_id: str,
    model_path: str,
    weights_path: str,
    weights_size: int,
    weights_mtime: int,
    chunk_size: int,
    overlap: int,
    corpus_size: int,
    corpus_mtime: int,
    corpus_document_count: int,
) -> BM25Index:
    chunks, _ = embedded_corpus(
        dataset,
        model_id,
        model_path,
        weights_path,
        weights_size,
        weights_mtime,
        chunk_size,
        overlap,
        corpus_size,
        corpus_mtime,
        corpus_document_count,
    )
    return BM25Index(chunks)


def _get_hits(
    dataset: str,
    store_name: str,
    chunk_size: int,
    overlap: int,
    query_vector: Any,
    count: int,
    query_text: str = "",
) -> tuple[list[dict[str, Any]], int]:
    if store_name in {"BM25", "Hybrid (NumPy + BM25)"}:
        corpus_path = resolve_corpus_path(dataset)
        corpus_stat = corpus_path.stat()
        sparse = sparse_index(
            dataset,
            *embedding_signature(),
            chunk_size,
            overlap,
            corpus_stat.st_size,
            corpus_stat.st_mtime_ns,
            corpus_total,
        )
        candidate_count = min(max(count * 3, count), len(sparse.documents))
        sparse_hits = sparse.search(query_text, candidate_count)
        if store_name == "BM25":
            return sparse_hits, len(sparse.documents)
        dense_store, _, _ = _ready_index(dataset, "NumPy (exact)", chunk_size, overlap)
        dense_hits = dense_store.search(query_vector, candidate_count)
        return hybrid_search(dense_hits, sparse_hits, count), len(sparse.documents)
    store, chunk_count, _ = _ready_index(dataset, store_name, chunk_size, overlap)
    return store.search(query_vector, count), chunk_count


def _run_retrieval_evaluation(
    dataset: str,
    store_name: str,
    chunk_size: int,
    overlap: int,
    rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    embedder = embedding_model(*embedding_signature())
    recalls = []
    ndcgs = []
    for row in rows:
        query = encode_query(embedder, str(row["question"]))
        hits, _ = _get_hits(dataset, store_name, chunk_size, overlap, query, 10, str(row["question"]))
        relevant = {str(row["source"])} if row.get("source") else set()
        recalls.append(recall_at_n(hits, relevant, 10))
        ndcgs.append(ndcg_at_n(hits, relevant, 10))
    return {
        "questions": len(rows),
        "recall@10": sum(recalls) / len(recalls) if recalls else 0.0,
        "ndcg@10": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
    }


def _context_window(model: str) -> int | None:
    value = model_info.get(model, {}).get("context_length") or CONTEXT_WINDOW_OVERRIDES.get(model)
    return int(value) if isinstance(value, (int, float)) else None


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "not measured"
    sign = "−" if value < 0 else "+"
    size = abs(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{sign}{size:.1f} {unit}"
        size /= 1024
    return f"{sign}{size:.1f} GB"


def _format_seconds(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.3f} s"


def _requires_openrouter_key(models: list[str] | tuple[str, ...]) -> bool:
    return any("::" not in model or model.startswith("openrouter::") for model in models)


def _model_label(model_name: str) -> str:
    if model_name == "rotation::answer":
        return "Provider rotation · configured order"
    configured = PROVIDER_ROUTER.models.get(model_name)
    if configured:
        return f"{configured.provider} · {configured.model_id}"
    return model_name


def _render_source_citations(hits: list[dict[str, Any]]) -> None:
    citations: dict[str, str] = {}
    for hit in hits:
        source = str(hit.get("source", ""))
        url = str(hit.get("source_url", ""))
        if source.startswith("Jolpica:") and not url.startswith(JOLPICA_API_BASE + "/"):
            url = jolpica_source_url(source)
        elif source.startswith("Wikipedia:") and not url.startswith(WIKIPEDIA_ARTICLE_BASE):
            url = wikipedia_source_url(source.removeprefix("Wikipedia:"))
        if url.startswith((JOLPICA_API_BASE + "/", WIKIPEDIA_ARTICLE_BASE)):
            label = (
                f"{JOLPICA_ATTRIBUTION}: {source}"
                if url.startswith(JOLPICA_API_BASE + "/")
                else source
            )
            citations[label or JOLPICA_ATTRIBUTION] = url
    if citations:
        st.markdown("**Sources**")
        for source, url in citations.items():
            st.markdown(f"- [{source}]({url})")


try:
    api_key = get_api_key()
    api_key_error = None
except ValueError as error:
    api_key = ""
    api_key_error = str(error)

model_info, catalog_error = model_catalog()
model_options = list(
    dict.fromkeys(
        (
            *DEFAULT_MODELS,
            DEFAULT_JUDGE_MODEL,
            DEFAULT_REWRITE_MODEL,
            *ADDITIONAL_MODELS,
            *sorted(model_info),
            *PROVIDER_ROUTER.models.keys(),
            "rotation::answer",
        )
    )
)

with st.sidebar:
    st.markdown("## Apex RAG")
    st.caption("F1 knowledge retrieval & model benchmark")
    st.space("small")

    dataset_title = st.selectbox("Knowledge source", ("Wikipedia", "Jolpica"), index=0)
    dataset = dataset_title.lower()
    store_names = st.multiselect(
        "Vector stores — compare",
        ("NumPy (exact)", "FAISS", "ChromaDB", "BM25", "Hybrid (NumPy + BM25)"),
        default=["NumPy (exact)"],
        help="FAISS and ChromaDB need the optional vector-store dependencies.",
    )
    selected_models = st.multiselect(
        "Answer models",
        model_options,
        default=["rotation::answer"] if "rotation::answer" in model_options else [],
        help="A model is queried once per selected vector store.",
        format_func=_model_label,
    )
    judge_enabled = st.toggle(
        "Use judge model",
        value=True,
        help="Turn this off to skip judge calls in answer comparisons and QA accuracy runs.",
    )
    judge_model = st.selectbox(
        "Judge model",
        model_options,
        index=model_options.index(DEFAULT_JUDGE_MODEL) if DEFAULT_JUDGE_MODEL in model_options else 0,
        disabled=not judge_enabled,
        format_func=_model_label,
    )
    rewrite_enabled = st.toggle("Rewrite query before embedding", value=False)
    rewrite_model = st.selectbox(
        "Query-rewrite model",
        model_options,
        index=model_options.index(DEFAULT_REWRITE_MODEL) if DEFAULT_REWRITE_MODEL in model_options else 0,
        disabled=not rewrite_enabled,
        format_func=_model_label,
    )
    st.space("small")
    chunk_size = st.select_slider(
        "Chunk size (embedding tokens)",
        options=(128, 192, 224),
        value=192,
        help="Tokenized with the embedding model tokenizer; each chunk stays within its 256-token input limit.",
    )
    overlap = st.select_slider(
        "Chunk overlap (tokens)",
        options=(0, 16, 32, 48),
        value=32,
    )
    if overlap >= chunk_size:
        st.error("Overlap must be smaller than the selected chunk size.")
    top_k = st.slider("Retrieved context count", min_value=1, max_value=10, value=5)

    st.space("small")
    st.markdown("**Model context windows**")
    context_models = (
        *selected_models,
        *((judge_model,) if judge_enabled else ()),
        *((rewrite_model,) if rewrite_enabled else ()),
    )
    for model in dict.fromkeys(context_models):
        context = _context_window(model)
        context_label = f"{int(context):,} tokens" if isinstance(context, (int, float)) else "not listed"
        st.caption(f"`{model}` · {context_label}")
    st.caption(f"Embedding: `{EMBEDDING_MODEL_ID}`")
    st.caption(f"Local weights: `{EMBEDDING_WEIGHTS_PATH}`")
    if api_key_error and _requires_openrouter_key((*selected_models, *((judge_model,) if judge_enabled else ()), *((rewrite_model,) if rewrite_enabled else ()))):
        st.error(api_key_error)
    elif not api_key and _requires_openrouter_key((*selected_models, *((judge_model,) if judge_enabled else ()), *((rewrite_model,) if rewrite_enabled else ()))):
        st.warning("Set `OPENROUTER_API_KEY` in the environment or project `.env` file.")
    elif not api_key:
        st.caption("Direct provider keys are read from their matching environment variables.")
    if catalog_error:
        st.caption(f"Live model catalog unavailable; showing configured defaults. {catalog_error}")

manifest_path = KNOWLEDGE_BASE / "manifest.json"
if not manifest_path.is_file():
    st.error(
        "The independent knowledge base is missing. Restore the original "
        "`data/raw/wikipedia`, `data/raw/jolpica`, and `data/final/qa.jsonl` inputs, "
        "then run `python -m f1_rag.ingest`."
    )
    st.stop()

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_total = manifest["documents"][dataset]
    qa_total = manifest["documents"]["evaluation"]
except (OSError, KeyError, ValueError) as error:
    st.error(f"The knowledge-base manifest is invalid: {error}")
    st.stop()

summary_columns = st.columns(4)
for column, label, value in zip(
    summary_columns,
    ("Selected source", "Source documents", "Evaluation Q&A", "Chunk configuration"),
    (dataset_title, f"{corpus_total:,}", f"{qa_total:,}", f"{chunk_size} / {overlap}"),
):
    column.metric(label, value, border=True)

if dataset == "jolpica":
    st.info(
        f"Data source: [{JOLPICA_ATTRIBUTION}]({JOLPICA_API_BASE}/). "
        "Retrieved records link to their corresponding API endpoints."
    )

query_tab, eval_tab, experiments_tab, about_tab = st.tabs(
    (":material/search: Compare answers", ":material/query_stats: Evaluation", ":material/tune: Experiments", ":material/folder: Project guide")
)

with query_tab:
    st.subheader("Ask the F1 knowledge base")
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder="Who won the 1976 Formula One World Championship?",
            height=90,
        )
        submitted = st.form_submit_button("Retrieve & compare", type="primary", width="stretch")

    if submitted:
        if not question.strip():
            st.error("Enter a question before running the comparison.")
        elif not store_names:
            st.error("Select at least one vector store.")
        elif not selected_models:
            st.error("Select at least one answer model.")
        elif overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
        elif _requires_openrouter_key((*selected_models, *((judge_model,) if judge_enabled else ()), *((rewrite_model,) if rewrite_enabled else ()))) and not api_key:
            st.error("Configure the OpenRouter key for the selected model or auxiliary model.")
        else:
            payload: dict[str, Any] = {
                "question": question.strip(),
                "query": question.strip(),
                "hits": {},
                "answers": {},
                "errors": {},
                "chunks": {},
            }
            try:
                if rewrite_enabled:
                    with st.spinner("Rewriting the search query…"):
                        payload["query"] = rewrite_query(
                            question.strip(),
                            rewrite_model,
                            corpus_preview(dataset),
                            api_key,
                        )
                with st.spinner("Loading the embedding model…"):
                    embedder = embedding_model(*embedding_signature())
                query_vector = encode_query(embedder, payload["query"])
            except (RuntimeError, ValueError, OSError) as error:
                LOGGER.exception("Failed to load the embedding model or encode the query.")
                payload["errors"]["retrieval"] = str(error)
                query_vector = None

            if query_vector is not None:
                for store_name in store_names:
                    try:
                        with st.spinner(f"Searching {store_name}… (first run builds its index)"):
                            hits, count = _get_hits(
                                dataset,
                                store_name,
                                chunk_size,
                                overlap,
                                query_vector,
                                top_k,
                                payload["query"],
                            )
                        payload["hits"][store_name] = hits
                        payload["chunks"][store_name] = count
                    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                        payload["errors"][store_name] = str(error)

                for store_name, hits in payload["hits"].items():
                    for model in selected_models:
                        key = f"{store_name}::{model}"
                        try:
                            with st.spinner(f"{model} · {store_name}…"):
                                context = _context_window(model)
                                response = answer_question(
                                    payload["question"],
                                    hits,
                                    model,
                                    api_key,
                                    context_length=context if isinstance(context, int) else None,
                                )
                            payload["answers"][key] = {"text": response, "judge": None}
                        except (RuntimeError, ValueError) as error:
                            payload["errors"][key] = str(error)
                            continue

                        if judge_enabled:
                            try:
                                with st.spinner(f"Judging {model} · {store_name}…"):
                                    verdict = judge_answer(
                                        payload["question"],
                                        "\n\n".join(
                                            f"Source: {hit.get('source', 'unknown')}\n"
                                            f"Source URL: {hit.get('source_url', '')}\n"
                                            f"{hit.get('text', '')}"
                                            for hit in hits
                                        ),
                                        response,
                                        judge_model,
                                        api_key,
                                    )
                                payload["answers"][key]["judge"] = verdict
                            except (RuntimeError, ValueError) as error:
                                payload["errors"][f"{key} judge"] = str(error)
            st.session_state["last_comparison"] = payload

    payload = st.session_state.get("last_comparison")
    if payload:
        st.markdown(f"**Question:** {payload['question']}")
        if payload["query"] != payload["question"]:
            st.caption(f"Rewritten retrieval query: {payload['query']}")
        if payload["errors"].get("retrieval"):
            st.error(payload["errors"]["retrieval"])
        for key, message in payload["errors"].items():
            if key != "retrieval":
                if key.endswith(" judge") and not judge_enabled:
                    continue
                st.error(f"{key}: {message}")

        for store_name, hits in payload["hits"].items():
            st.caption(f"{store_name} · {payload['chunks'][store_name]:,} chunks · top {len(hits)}")
        for model in selected_models:
            st.markdown(f"### `{model}`")
            model_columns = st.columns(max(1, len(store_names)))
            for column, store_name in zip(model_columns, store_names):
                key = f"{store_name}::{model}"
                with column:
                    st.markdown(f"**{store_name}**")
                    result = payload["answers"].get(key)
                    if result:
                        st.write(result["text"])
                        store_hits = payload["hits"].get(store_name, [])
                        _render_source_citations(store_hits)
                        verdict = result["judge"] if judge_enabled else None
                        if verdict:
                            st.badge(
                                "Correct" if verdict["correct"] else "Needs review",
                                icon=":material/check_circle:" if verdict["correct"] else ":material/warning:",
                                color="green" if verdict["correct"] else "orange",
                            )
                            st.caption(f"Judge note: {verdict['reason']}")
                    elif key not in payload["errors"]:
                        st.info("No answer was generated for this store/model combination.")
                    store_hits = payload["hits"].get(store_name, [])
                    if store_hits:
                        with st.expander("Retrieved evidence"):
                            for rank, hit in enumerate(store_hits, start=1):
                                st.markdown(
                                    f"**[{rank}] {hit.get('source', 'Unknown source')}** · "
                                    f"score `{hit.get('score', 0):.3f}`"
                                )
                                source_url = hit.get("source_url")
                                if source_url and source_url.startswith(JOLPICA_API_BASE + "/"):
                                    st.markdown(f"[View {JOLPICA_ATTRIBUTION} record]({source_url})")
                                elif hit.get("source", "").startswith("Wikipedia:"):
                                    wikipedia_url = (
                                        source_url
                                        if source_url and source_url.startswith(WIKIPEDIA_ARTICLE_BASE)
                                        else wikipedia_source_url(
                                            str(hit["source"]).removeprefix("Wikipedia:")
                                        )
                                    )
                                    st.markdown(f"[View Wikipedia article]({wikipedia_url})")
                                st.write(hit.get("text", ""))
                                if rank < len(store_hits):
                                    st.divider()

with eval_tab:
    st.subheader("Retrieval & answer evaluation")
    st.caption(
        "Reference-source hit@10 and source-level NDCG@10 use the single source label on each QA row; "
        "they are not document/chunk recall. QA accuracy is a stratified sample estimate, not a corpus-wide score."
    )
    eval_pool = _dataset_questions(dataset)
    total_eval_rows = sum(
        str(row.get("source", "")).startswith("Wikipedia:") == (dataset == "wikipedia")
        for row in evaluation_rows()
    )
    st.caption(
        f"QA questions with a matching reference source: **{len(eval_pool):,} / {total_eval_rows:,}** "
        f"for {dataset_title}"
    )
    metric_count = st.slider("Retrieval evaluation sample", min_value=10, max_value=100, value=20, step=10)
    eval_button = st.button("Run source-match benchmark (top 10)", type="primary", disabled=not store_names)
    if eval_button:
        if not eval_pool:
            st.error(f"No QA rows map to the {dataset_title} corpus.")
        else:
            sample = stratified_sample(eval_pool, min(metric_count, len(eval_pool)), seed=42)
            rows = []
            progress = st.progress(0, text="Preparing retrieval evaluation…")
            for index, store_name in enumerate(store_names):
                try:
                    result = _run_retrieval_evaluation(dataset, store_name, chunk_size, overlap, sample)
                    rows.append({"Vector store": store_name, **result})
                except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                    st.error(f"{store_name}: {error}")
                progress.progress((index + 1) / len(store_names), text=f"Evaluated {store_name}")
            st.session_state["retrieval_metrics"] = rows
            progress.empty()

    metrics = st.session_state.get("retrieval_metrics", [])
    if metrics:
        metric_columns = st.columns(len(metrics))
        for column, result in zip(metric_columns, metrics):
            with column:
                st.markdown(f"**{result['Vector store']}**")
                st.metric("Reference-source hit@10", f"{result['recall@10']:.1%}")
                st.metric("Distinct-source NDCG@10", f"{result['ndcg@10']:.3f}")
                st.caption(f"{result['questions']} questions")

    st.space("medium")
    st.markdown("#### QA accuracy (LLM judge)")
    if not judge_enabled:
        st.info("Enable the judge model in the sidebar to run QA accuracy evaluation.")
    accuracy_count = st.slider("QA accuracy sample size", min_value=10, max_value=100, value=30, step=10)
    st.caption(
        "Questions are sampled across entity types. Treat judge results as an estimate for this sample "
        "and calibrate them with human review."
    )
    accuracy_model = st.selectbox("Model to evaluate", selected_models or model_options, key="accuracy_model", format_func=_model_label)
    accuracy_store = st.selectbox("Vector store to evaluate", store_names or ("NumPy (exact)",), key="accuracy_store")
    accuracy_button = st.button(
        "Evaluate answers with judge",
        disabled=not judge_enabled or (_requires_openrouter_key((accuracy_model, judge_model)) and not api_key) or not store_names or not selected_models,
    )
    if accuracy_button:
        if not eval_pool:
            st.error(f"No QA rows map to the {dataset_title} corpus.")
        else:
            sample = stratified_sample(eval_pool, min(accuracy_count, len(eval_pool)), seed=7)
            verdicts = []
            progress = st.progress(0, text="Running judged QA sample…")
            try:
                embedder = embedding_model(*embedding_signature())
                for index, row in enumerate(sample):
                    query_vector = encode_query(embedder, str(row["question"]))
                    hits, _ = _get_hits(
                        dataset,
                        accuracy_store,
                        chunk_size,
                        overlap,
                        query_vector,
                        top_k,
                        str(row["question"]),
                    )
                    generated = answer_question(
                        str(row["question"]),
                        hits,
                        accuracy_model,
                        api_key,
                        context_length=_context_window(accuracy_model),
                    )
                    verdict = judge_answer(
                        str(row["question"]),
                        str(row.get("answer", "")),
                        generated,
                        judge_model,
                        api_key,
                    )
                    verdicts.append({"question": row["question"], **verdict})
                    progress.progress((index + 1) / len(sample), text=f"Judged {index + 1}/{len(sample)}")
                st.session_state["qa_verdicts"] = verdicts
            except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                st.error(f"QA accuracy run failed: {error}")
            progress.empty()
    verdicts = st.session_state.get("qa_verdicts", [])
    if judge_enabled and verdicts:
        st.metric("QA accuracy", f"{qa_accuracy(verdicts):.1%}", f"{len(verdicts)} judged questions")
        with st.expander("QA judge details"):
            for verdict in verdicts:
                st.markdown(
                    f"{'✅' if verdict['correct'] else '❌'} **{verdict['question']}**  \n"
                    f"{verdict['reason']}"
                )

with experiments_tab:
    st.subheader("Controlled retrieval experiments")
    st.caption(
        "The tuning grid uses a deterministic tuning split. The separate holdout action runs the selected configuration once. "
        "The current QA labels identify a source, so retrieval scores are source-level until complete qrels are added."
    )
    experiment_rows = _dataset_questions(dataset)
    tuning_rows = _evaluation_split(experiment_rows, "tune")
    holdout_rows = _evaluation_split(experiment_rows, "test")
    vector_store_options = ("NumPy (exact)", "FAISS", "ChromaDB")
    experiment_stores = st.multiselect(
        "Vector stores to compare",
        vector_store_options,
        default=["NumPy (exact)"],
        key="experiment_stores",
    )
    experiment_sizes = st.multiselect(
        "Chunk sizes (embedding tokens)", (128, 192, 224), default=[128, 192, 224], key="experiment_chunk_sizes"
    )
    experiment_overlaps = st.multiselect(
        "Overlap (tokens)", (0, 16, 32, 48), default=[0, 16, 32], key="experiment_overlaps"
    )
    experiment_top_ks = st.multiselect(
        "Retrieved chunk counts", (3, 5, 8, 10), default=[3, 5, 8], key="experiment_top_ks"
    )
    experiment_repeats = st.slider("Index and query repetitions", 3, 10, 5, key="experiment_repeats")
    experiment_sample_size = st.slider(
        "Tuning Q&A sample", 10, 100, 30, step=10, key="experiment_sample_size"
    )
    if st.button("Run tuning grid", type="primary", disabled=not experiment_stores or not tuning_rows):
        tuning_sample = stratified_sample(
            tuning_rows, min(experiment_sample_size, len(tuning_rows)), seed=20261117
        )
        corpus_path = resolve_corpus_path(dataset)
        corpus_stat = corpus_path.stat()

        def prepare_experiment(chunk_size: int, overlap: int) -> tuple[list[dict[str, Any]], Any, Any]:
            chunks, vectors = embedded_corpus(
                dataset,
                *embedding_signature(),
                chunk_size,
                overlap,
                corpus_stat.st_size,
                corpus_stat.st_mtime_ns,
                corpus_total,
            )
            query_embedder = embedding_model(*embedding_signature())
            return chunks, vectors, lambda text: encode_query(query_embedder, text)

        with st.spinner("Building and measuring the selected configurations…"):
            try:
                grid_results = run_hyperparameter_grid(
                    dataset=dataset,
                    store_names=experiment_stores,
                    embedding_model=EMBEDDING_MODEL_ID,
                    chunk_sizes=experiment_sizes,
                    overlaps=experiment_overlaps,
                    top_ks=experiment_top_ks,
                    repeats=experiment_repeats,
                    prepare=prepare_experiment,
                    queries=tuning_sample,
                    output_dir=KNOWLEDGE_BASE / "runs",
                )
                st.session_state["tuning_results"] = grid_results
            except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                st.error(f"The tuning experiment failed: {error}")

    grid_results = st.session_state.get("tuning_results", [])
    if grid_results:
        st.dataframe(
            pd.DataFrame([
                {
                    "Run": item["run_id"],
                    "Store": item["config"]["store_name"],
                    "Chunk": item["config"]["chunk_size"],
                    "Overlap": item["config"]["overlap"],
                    "Top K": item["config"]["top_k"],
                    "Build seconds": item["results"]["index_build_seconds_median"],
                    "Search p50 ms": item["results"]["search_latency_ms_p50"],
                    "Search p95 ms": item["results"]["search_latency_ms_p95"],
                    "RAM delta bytes": item["results"]["rss_delta_bytes_median"],
                    "Build CPU seconds": item["results"]["index_build_cpu_seconds_median"],
                    "Search CPU seconds": item["results"]["search_cpu_seconds_median"],
                    "Recall@K": item["results"]["recall_at_k"],
                    "NDCG@K": item["results"]["ndcg_at_k"],
                    "Error": item["error"],
                }
                for item in grid_results
            ]),
            alt="Comparison of vector store benchmark runs across configuration, latency, resource use, and retrieval quality",
            hide_index=True,
            width="stretch",
        )
        st.caption("Each result is appended to knowledge_base/runs/runs.jsonl with its config and environment metadata.")

    st.markdown("#### One-time holdout run")
    test_store = st.selectbox("Vector store", vector_store_options, key="holdout_store")
    test_chunk_size = st.selectbox("Chunk size", (128, 192, 224), index=1, key="holdout_chunk_size")
    test_overlap = st.selectbox("Overlap", (0, 16, 32, 48), index=2, key="holdout_overlap")
    test_top_k = st.selectbox("Top K", (3, 5, 8, 10), index=1, key="holdout_top_k")
    holdout_settings = {
        "dataset": dataset,
        "store_name": test_store,
        "embedding_model": EMBEDDING_MODEL_ID,
        "chunk_size": test_chunk_size,
        "overlap": test_overlap,
        "top_k": test_top_k,
        "repeats": 1,
        "evaluation_split": "holdout",
    }
    run_log = KNOWLEDGE_BASE / "runs" / "runs.jsonl"
    prior_holdout = False
    if run_log.is_file():
        try:
            for line in run_log.read_text(encoding="utf-8").splitlines():
                saved_run = json.loads(line)
                if (
                    saved_run.get("config", {}).get("dataset") == dataset
                    and saved_run.get("config", {}).get("evaluation_split") == "holdout"
                    and saved_run.get("error") is None
                ):
                    prior_holdout = True
                    break
        except (OSError, json.JSONDecodeError):
            st.warning("Could not read the experiment history; holdout protection is unavailable.")
    if prior_holdout:
        st.info("This dataset already has a successful holdout result in the experiment log.")
    if st.button("Run selected config on holdout once", disabled=not holdout_rows or prior_holdout):
        test_sample = stratified_sample(holdout_rows, len(holdout_rows), seed=20261118)
        corpus_path = resolve_corpus_path(dataset)
        corpus_stat = corpus_path.stat()
        with st.spinner("Preparing selected holdout configuration…"):
            try:
                chunks, vectors = embedded_corpus(
                    dataset,
                    *embedding_signature(),
                    test_chunk_size,
                    test_overlap,
                    corpus_stat.st_size,
                    corpus_stat.st_mtime_ns,
                    corpus_total,
                )
                query_embedder = embedding_model(*embedding_signature())
                holdout_result = benchmark_store(
                    BenchmarkConfig(
                        dataset,
                        test_store,
                        EMBEDDING_MODEL_ID,
                        test_chunk_size,
                        test_overlap,
                        test_top_k,
                        repeats=1,
                        evaluation_split="holdout",
                    ),
                    chunks,
                    vectors,
                    test_sample,
                    lambda text: encode_query(query_embedder, text),
                    KNOWLEDGE_BASE / "runs",
                )
                st.session_state["holdout_result"] = holdout_result
            except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                st.error(f"The holdout run failed: {error}")
    holdout_result = st.session_state.get("holdout_result")
    if holdout_result:
        if holdout_result["error"]:
            st.error(holdout_result["error"])
        else:
            result = holdout_result["results"]
            st.success(f"Holdout run {holdout_result['run_id']} · {result['queries']} questions")
            first_metrics = st.columns(4)
            first_metrics[0].metric("Recall@K", f"{result['recall_at_k']:.3f}")
            first_metrics[1].metric("NDCG@K", f"{result['ndcg_at_k']:.3f}")
            first_metrics[2].metric("Search p50", f"{result['search_latency_ms_p50']:.2f} ms")
            first_metrics[3].metric("Search p95", f"{result['search_latency_ms_p95']:.2f} ms")
            second_metrics = st.columns(4)
            second_metrics[0].metric("Index build", f"{result['index_build_seconds_median']:.2f} s")
            second_metrics[1].metric("RAM delta", _format_bytes(result["rss_delta_bytes_median"]))
            second_metrics[2].metric("Build CPU", _format_seconds(result["index_build_cpu_seconds_median"]))
            second_metrics[3].metric("Search CPU", _format_seconds(result["search_cpu_seconds_median"]))

with about_tab:
    st.subheader("Project map")
    st.markdown(
        """
        - `knowledge_base/corpora/` — Independent, ready-to-use Wikipedia and Jolpica corpora
        - `knowledge_base/evaluation/` — Reference QA set for evaluation
        - `knowledge_base/indices/` — Rebuildable local indexes (excluded from Git)
        - `models/embedding/` — Downloaded Hugging Face embedding model
        - `models/llm/` — OpenRouter-hosted models; model weights are not stored locally
        - `config/models.toml` — Answer, judge, and rewrite model IDs; embedding model paths
        - `.env` — Provider API keys; excluded from Git
        - `src/f1_rag/` — Ingestion, chunking, embeddings, vector stores, retrieval, generation, and evaluation

        The third vector-store option is **NumPy exact cosine**: Windows-friendly and service-free.
        FAISS and ChromaDB are also available for comparison.
        """
    )
    st.info(
        "Chunks are sized in tokens with the embedding model tokenizer and capped below its 256-token "
        "input limit. First-time index builds embed documents in batches and show progress. Model context "
        "windows come from the OpenRouter catalog; unavailable values are shown as “not listed.”"
    )
    st.markdown(
        f"Jolpica data attribution: [{JOLPICA_ATTRIBUTION}]({JOLPICA_API_BASE}/). "
        "Retrieved Jolpica evidence includes direct API record links."
    )
