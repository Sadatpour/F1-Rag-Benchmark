from __future__ import annotations

import hashlib
import base64
from http.client import HTTPException
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
TRACK_ARTWORK = base64.b64encode((ROOT / "assets" / "monaco-circuit.svg").read_bytes()).decode("ascii")
LOGGER = logging.getLogger(__name__)
sys.path.insert(0, str(ROOT / "src"))

from f1_rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_PROFILES,
    embedding_profile,
    load_embedding_model,
)
from f1_rag.chunking import (
    CHUNKING_METHODS,
    FIXED_CHUNKING,
    RECURSIVE_CHUNKING,
    SEMANTIC_CHUNKING,
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
    ANSWER_NOT_FOUND,
    PROVIDER_ROUTER,
    answer_question,
    get_api_key,
    judge_answer,
    list_models,
    rewrite_query_safely,
)
from f1_rag.reranking import (
    DEFAULT_RERANKER_MODEL,
    RERANKER_CANDIDATE_COUNT,
    RERANKER_PROFILES,
    load_reranker,
    rerank_candidates,
    reranker_profile,
)
from f1_rag.retrieval import embed_documents, encode_query, load_evaluation_rows, prepare_index
from f1_rag.reporting import (
    dataframe_to_csv,
    dataframe_to_pdf,
    load_evaluation_runs,
    save_evaluation_run,
)
from f1_rag.sources import (
    JOLPICA_API_BASE,
    JOLPICA_ATTRIBUTION,
    WIKIPEDIA_ARTICLE_BASE,
    jolpica_source_url,
    wikipedia_source_url,
)
from f1_rag.sparse import BM25Index, build_source_search_index, hybrid_search


st.set_page_config(
    page_title="Apex RAG | F1 Knowledge Lab",
    page_icon=":material/speed:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply the updated Ask defaults once per browser session. This also updates
# sessions that already had the previous widget values, without overriding
# later user selections on every Streamlit rerun.
if st.session_state.get("_ask_defaults_version") != 2:
    st.session_state["ask_vector_stores"] = ["NumPy (exact)", "FAISS", "ChromaDB"]
    st.session_state["ask_retrieval_method"] = "Hybrid"
    st.session_state["ask_top_k"] = 3
    st.session_state["_ask_defaults_version"] = 2

st.html(
    """
    <style>
    :root { --apex-red:#e10600; --apex-ink:#172033; --apex-muted:#687386; --apex-line:#e5e9ef; --apex-panel:#fff; }
    .stApp { background:#f4f6f9; color:var(--apex-ink); }
    [data-testid="stHeader"] { background:rgba(244,246,249,.9); }
    [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--apex-line); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.35rem; }
    [data-testid="stMainBlockContainer"] { max-width:1560px; padding-top:1.35rem; padding-bottom:3rem; }
    .hero { position:relative; overflow:hidden; display:flex; justify-content:space-between; align-items:center; gap:2rem; min-height:190px; border:1px solid #e1e5eb; border-radius:22px; padding:2rem 2.4rem; margin:.15rem 0 1.1rem; background:linear-gradient(112deg,#fff 0%,#fff 68%,#fcebea 100%); box-shadow:0 12px 30px rgba(23,32,51,.055); }
    .hero:before { content:""; position:absolute; inset:0 auto 0 0; width:8px; background:var(--apex-red); }
    .hero:after { content:""; position:absolute; right:7%; bottom:-70px; width:39%; height:150px; transform:skewX(-32deg); border-top:2px solid rgba(225,6,0,.15); background:repeating-linear-gradient(90deg,transparent 0 28px,rgba(23,32,51,.035) 28px 30px); }
    .hero-main { position:relative; z-index:1; max-width:760px; }
    .eyebrow { color:#a00b08; text-transform:uppercase; letter-spacing:.16em; font-weight:800; font-size:.72rem; }
    .hero-title { color:var(--apex-ink); font-size:clamp(2.25rem,4vw,3.2rem); line-height:1; font-weight:850; margin:.55rem 0 .7rem; letter-spacing:-.06em; }
    .hero-title span { color:var(--apex-red); }
    .hero-copy { color:#5b6575; max-width:690px; line-height:1.6; margin:0; }
    .hero-track { position:relative; z-index:1; width:min(42%,400px); height:auto; border:0; background:transparent; box-shadow:none; }
    [data-testid="stSidebar"] h2 { color:var(--apex-ink); font-weight:800; letter-spacing:-.03em; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color:#344054; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#4d5869; font-weight:650; }
    div[data-testid="stMetric"] { min-height:104px; background:var(--apex-panel); border:1px solid var(--apex-line); border-radius:16px; padding:1rem 1.1rem; box-shadow:0 4px 14px rgba(23,32,51,.035); }
    div[data-testid="stMetricLabel"] { color:var(--apex-muted); font-size:.78rem; font-weight:650; }
    div[data-testid="stMetricValue"] { color:var(--apex-ink); font-size:1.32rem; font-weight:800; }
    .stTabs [data-baseweb="tab-list"] { display:flex; gap:.35rem; padding:.4rem; border:1px solid #e6e9ee; border-radius:14px; background:#e9edf2; }
    .stTabs [data-baseweb="tab"] { min-height:2.7rem; border-radius:10px; padding:.55rem 1rem; color:#576174; font-weight:700; }
    .stTabs [aria-selected="true"] { background:#fff; color:#a10e0a; box-shadow:0 2px 8px rgba(23,32,51,.1); }
    .stTabs [data-baseweb="tab-highlight"] { background:var(--apex-red); }
    .stTabs [data-baseweb="tab-border"] { display:none; }
    .stButton > button, .stFormSubmitButton > button { min-height:2.75rem; border-radius:10px; font-weight:750; transition:transform .15s ease,box-shadow .15s ease; }
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background:var(--apex-red); border-color:var(--apex-red); color:#fff; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { transform:translateY(-1px); box-shadow:0 7px 18px rgba(225,6,0,.16); }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:var(--apex-line); border-radius:15px; box-shadow:0 4px 14px rgba(23,32,51,.035); }
    div[data-testid="stDataFrame"] { border:1px solid var(--apex-line); border-radius:14px; overflow:hidden; }
    [data-testid="stExpander"] { background:#fff; border:1px solid var(--apex-line); border-radius:12px; }
    [data-testid="stAlert"] { border-radius:12px; }
    h1,h2,h3 { color:var(--apex-ink); letter-spacing:-.025em; }
    p,li { line-height:1.58; }
    @media (max-width:700px) { .hero { min-height:0; padding:1.4rem 1.3rem; flex-direction:column; align-items:flex-start; } .hero-track { width:100%; } .hero:after { width:75%; } [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; } .stTabs [data-baseweb="tab-list"] { overflow-x:auto; } }
    </style>
    <div class="hero">
      <div class="hero-main">
        <div class="eyebrow">Formula 1 · Data engineering · AI</div>
        <div class="hero-title">APEX <span>RAG</span> LAB</div>
        <p class="hero-copy">Explore F1 knowledge, compare retrieval strategies, and measure answer quality in one focused workspace.</p>
      </div>
      <img class="hero-track" src="data:image/svg+xml;base64,__TRACK_ARTWORK__" alt="Illustrated Circuit de Monaco" />
    </div>
    """.replace("__TRACK_ARTWORK__", TRACK_ARTWORK),
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


def embedding_signature(model_name: str = DEFAULT_EMBEDDING_MODEL) -> tuple[str, str, str, int, int]:
    profile = embedding_profile(model_name)
    weights_path = profile["weights_path"]
    try:
        stat = weights_path.stat()
        size, modified = stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        size, modified = 0, 0
    return (
        profile["repo_id"],
        str(profile["local_path"]),
        str(weights_path),
        size,
        modified,
    )


@st.cache_resource(show_spinner=False, max_entries=3)
def reranker_model(
    model_name: str,
    model_path: str,
    max_length: int,
    weights_size: int,
    weights_mtime: int,
) -> Any:
    del model_path, max_length, weights_size, weights_mtime
    return load_reranker(model_name)


def load_selected_reranker(model_name: str) -> Any:
    profile = reranker_profile(model_name)
    weights_path = profile["local_path"] / "model.safetensors"
    stat = weights_path.stat()
    return reranker_model(
        model_name,
        str(profile["local_path"]),
        profile["max_length"],
        stat.st_size,
        stat.st_mtime_ns,
    )


@st.cache_data(ttl=3600, show_spinner=False, max_entries=1)
def model_catalog(api_key_configured: bool) -> tuple[dict[str, dict[str, Any]], str | None]:
    if not api_key_configured:
        return {}, None
    try:
        return list_models(), None
    except (RuntimeError, TimeoutError, HTTPException, OSError) as error:
        message = str(error)
        if "HTTP 403" in message and "security policy" in message.lower():
            message = (
                "OpenRouter blocked the model-catalog request (HTTP 403). Configured models remain available. "
                "Check the API key's IP allowlist in OpenRouter Security settings."
            )
        return {}, message


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
    chunking_method: str,
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
            chunking_method=chunking_method,
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
    chunking_method: str,
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
            chunking_method,
            corpus_size,
            corpus_mtime,
            corpus_document_count,
        ),
        embedding_model_id=model_id,
        embedding_model_path=Path(model_path),
        embedding_weights_path=Path(weights_path),
        chunking_method=chunking_method,
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


def _ready_index(
    dataset: str, store_name: str, chunk_size: int, overlap: int,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunking_method: str = FIXED_CHUNKING,
) -> tuple[Any, int, bool]:
    path = resolve_corpus_path(dataset)
    stat = path.stat()
    return indexed_corpus(
        dataset,
        store_name,
        *embedding_signature(model_name),
        chunk_size,
        overlap,
        chunking_method,
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
    chunking_method: str,
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
        chunking_method,
        corpus_size,
        corpus_mtime,
        corpus_document_count,
    )
    return BM25Index(chunks)


@st.cache_resource(show_spinner=False, max_entries=6)
def direct_source_index(
    dataset: str,
    chunk_size: int,
    overlap: int,
    corpus_size: int,
    corpus_mtime: int,
) -> BM25Index:
    documents = list(iter_documents(dataset))
    index = build_source_search_index(documents, chunk_size, overlap)
    if not index.documents:
        raise ValueError(f"The {dataset} corpus did not produce any searchable chunks.")
    return index


def _get_hits(
    dataset: str,
    store_name: str,
    chunk_size: int,
    overlap: int,
    query_vector: Any,
    count: int,
    query_text: str = "",
    retrieval_method: str = "Dense",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunking_method: str = FIXED_CHUNKING,
) -> tuple[list[dict[str, Any]], int]:
    if retrieval_method == "Direct source search":
        corpus_path = resolve_corpus_path(dataset)
        corpus_stat = corpus_path.stat()
        index = direct_source_index(
            dataset, chunk_size, overlap, corpus_stat.st_size, corpus_stat.st_mtime_ns
        )
        return index.search(query_text, count), len(index.documents)
    if retrieval_method in {"BM25", "Hybrid"}:
        corpus_path = resolve_corpus_path(dataset)
        corpus_stat = corpus_path.stat()
        sparse = sparse_index(
            dataset,
            *embedding_signature(model_name),
            chunk_size,
            overlap,
            chunking_method,
            corpus_stat.st_size,
            corpus_stat.st_mtime_ns,
            corpus_total,
        )
        requested_candidates = max(count * 3, count) if retrieval_method == "Hybrid" else count
        candidate_count = min(requested_candidates, len(sparse.documents))
        sparse_hits = sparse.search(query_text, candidate_count)
        if retrieval_method == "BM25":
            return sparse_hits, len(sparse.documents)
        dense_store, _, _ = _ready_index(
            dataset, store_name, chunk_size, overlap, model_name, chunking_method
        )
        dense_hits = dense_store.search(query_vector, candidate_count)
        return hybrid_search(dense_hits, sparse_hits, count), len(sparse.documents)
    store, chunk_count, _ = _ready_index(
        dataset, store_name, chunk_size, overlap, model_name, chunking_method
    )
    return store.search(query_vector, count), chunk_count


def _run_retrieval_evaluation(
    dataset: str,
    store_name: str,
    chunk_size: int,
    overlap: int,
    rows: list[dict[str, Any]],
    top_ks: list[int],
    retrieval_method: str = "Dense",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    embedder = embedding_model(*embedding_signature(model_name))
    cutoffs = sorted(set(top_ks))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("Select at least one Top K value greater than zero.")
    recalls = {cutoff: [] for cutoff in cutoffs}
    ndcgs = {cutoff: [] for cutoff in cutoffs}
    record_count = 0
    for row in rows:
        query = encode_query(embedder, str(row["question"]))
        hits, record_count = _get_hits(
            dataset,
            store_name,
            chunk_size,
            overlap,
            query,
            max(cutoffs),
            str(row["question"]),
            retrieval_method,
            model_name,
        )
        relevant = {str(row["source"])} if row.get("source") else set()
        for cutoff in cutoffs:
            recalls[cutoff].append(recall_at_n(hits, relevant, cutoff))
            ndcgs[cutoff].append(ndcg_at_n(hits, relevant, cutoff))
    return {
        "questions": len(rows),
        "Records": record_count,
        "metrics_at_k": {
            cutoff: {
                "recall": sum(recalls[cutoff]) / len(recalls[cutoff]) if recalls[cutoff] else 0.0,
                "ndcg": sum(ndcgs[cutoff]) / len(ndcgs[cutoff]) if ndcgs[cutoff] else 0.0,
            }
            for cutoff in cutoffs
        },
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

model_info, catalog_error = model_catalog(bool(api_key))
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

eval_store_names: Sequence[str] = ()
qa_store_names: Sequence[str] = ()
benchmark_stores: Sequence[str] = ()
tuning_stores: Sequence[str] = ()
holdout_stores: Sequence[str] = ()

with st.sidebar:
    st.markdown("## :material/speed: Apex RAG")
    st.caption("F1 knowledge lab")
    dataset_title = st.selectbox("Knowledge source", ("Wikipedia", "Jolpica"), index=0)
    dataset = dataset_title.lower()
    embedding_options = list(EMBEDDING_PROFILES)
    selected_embedding_model = st.selectbox(
        "Embedding model",
        embedding_options,
        index=embedding_options.index(DEFAULT_EMBEDDING_MODEL),
        key="embedding_model_choice",
        help="This model converts both corpus chunks and questions to vectors. Add profiles in config/models.toml.",
    )
    workspace = st.radio(
        "Workspace",
        ("Ask", "Evaluation & benchmark", "Project guide"),
        key="workspace_page",
    )

    if workspace == "Ask":
        st.markdown("**Retrieval setup**")
        answer_mode = st.radio(
            "Answer mode",
            ("RAG", "Direct source search"),
            horizontal=True,
            key="ask_answer_mode",
        )
        direct_source_search = answer_mode == "Direct source search"
        if direct_source_search:
            chunking_method = FIXED_CHUNKING
            retrieval_method = "Direct source search"
            store_names = ["Direct source search"]
            st.caption(
                "Searches the selected local corpus with BM25 word matching, without loading embedding models. "
                "Matching source passages are then provided to the LLM."
            )
        else:
            vector_stores = st.multiselect(
                "Vector store / index backends",
                ("NumPy (exact)", "FAISS", "ChromaDB"),
                key="ask_vector_stores",
                help="Each selected backend gets its own answer column.",
            )
            retrieval_method = st.selectbox(
                "Retrieval method", ("Dense", "Hybrid", "BM25"), key="ask_retrieval_method",
                help="Dense uses vectors; BM25 uses keywords; Hybrid combines both rankings.",
            )
            store_names = ["BM25"] if retrieval_method == "BM25" else list(vector_stores)
            if retrieval_method == "BM25":
                st.caption("BM25 runs alone; vector-store selections do not apply.")
            chunking_method = st.selectbox(
                "Chunking method", CHUNKING_METHODS, index=0, key="ask_chunking_method"
            )
            chunking_hints = {
                FIXED_CHUNKING: (
                    "Splits text into fixed token-sized chunks with the selected overlap. "
                    "Fast and predictable; selected by default."
                ),
                SEMANTIC_CHUNKING: (
                    "Groups neighboring sentences by embedding similarity and starts a new chunk "
                    "when their meaning changes. Uses sentence embeddings, so it is slower than fixed chunking."
                ),
                RECURSIVE_CHUNKING: (
                    "Splits by paragraph, then sentence, then word as needed to respect the "
                    "embedding model's token limit. Chunk size and overlap still apply."
                ),
            }
            st.caption(chunking_hints[chunking_method])
        selected_models = st.multiselect(
            "Answer models", model_options,
            default=["rotation::answer"] if "rotation::answer" in model_options else [],
            key="ask_answer_models", format_func=_model_label,
        )
        top_k = st.selectbox("Retrieved chunks (Top K)", (1, 3, 5), key="ask_top_k")
        chunk_unit = "words" if direct_source_search else "tokens"
        chunk_size = st.select_slider(
            f"Chunk size ({chunk_unit})", (128, 192, 224), value=192, key="ask_chunk_size"
        )
        overlap = st.select_slider(
            f"Chunk overlap ({chunk_unit})", (0, 16, 32, 48), value=32, key="ask_overlap"
        )
        st.markdown("**Advanced settings**")
        reranking_enabled = st.toggle(
            "Enable reranking",
            value=False,
            key="ask_reranking_enabled",
            help="The configured MS MARCO CrossEncoder is English-focused. Enable query rewrite for Persian questions.",
            disabled=direct_source_search,
        )
        reranker_options = list(RERANKER_PROFILES)
        reranker_name = st.selectbox(
            "Reranker model",
            reranker_options,
            index=reranker_options.index(DEFAULT_RERANKER_MODEL) if DEFAULT_RERANKER_MODEL in reranker_options else 0,
            key="ask_reranker_model",
            disabled=not reranking_enabled or direct_source_search,
        ) if reranker_options else ""
        reranker_candidate_count = st.select_slider(
            "Candidate chunks before reranking",
            (10, 20, 30, 50),
            value=RERANKER_CANDIDATE_COUNT if RERANKER_CANDIDATE_COUNT in (10, 20, 30, 50) else 30,
            key="ask_reranker_candidates",
            disabled=not reranking_enabled or direct_source_search,
            help="Retrieve this many candidates first. The reranker then selects the final Top K chunks.",
        )
        judge_enabled = st.toggle("Judge against retrieved evidence", value=False, key="ask_judge_enabled")
        judge_model = st.selectbox(
            "Judge model", model_options,
            index=model_options.index(DEFAULT_JUDGE_MODEL) if DEFAULT_JUDGE_MODEL in model_options else 0,
            key="ask_judge_model", disabled=not judge_enabled, format_func=_model_label,
        )
        rewrite_enabled = st.toggle(
            "Rewrite query before search", value=False, key="ask_rewrite_enabled",
            disabled=direct_source_search,
        )
        rewrite_model = st.selectbox(
            "Query rewrite model", model_options,
            index=model_options.index(DEFAULT_REWRITE_MODEL) if DEFAULT_REWRITE_MODEL in model_options else 0,
            key="ask_rewrite_model", disabled=not rewrite_enabled, format_func=_model_label,
        )
    elif workspace == "Evaluation & benchmark":
        evaluation_view = st.radio(
            "Evaluation mode",
            ("Retrieval quality", "Answer accuracy", "Vector-store benchmark", "Hyperparameter tuning", "Final holdout"),
            key="evaluation_mode",
        )
        vector_store_options = ("NumPy (exact)", "FAISS", "ChromaDB")
        method_options = ("Dense", "Hybrid", "BM25")

        if evaluation_view == "Retrieval quality":
            eval_method = st.selectbox("Retrieval method", method_options, key="eval_method")
            eval_stores = st.multiselect(
                "Vector stores", vector_store_options, default=["NumPy (exact)"],
                key="eval_stores", disabled=eval_method == "BM25",
            )
            eval_store_names = ["BM25"] if eval_method == "BM25" else eval_stores
            eval_top_ks = st.pills(
                "Top K cutoffs", (1, 3, 5), default=[1, 3, 5], selection_mode="multi", key="eval_top_ks"
            ) or []
            eval_chunk_size = st.selectbox("Chunk size (tokens)", (128, 192, 224), index=1, key="eval_chunk")
            eval_overlap = st.selectbox("Overlap (tokens)", (0, 16, 32, 48), index=2, key="eval_overlap")
            eval_sample_size = st.slider("Tuning questions", 10, 100, 30, step=10, key="eval_sample_size")
            st.caption("Metrics use the matched source labels in qa.jsonl.")
            run_retrieval_eval = st.button(
                "Run retrieval evaluation", type="primary", width="stretch",
                disabled=not eval_store_names or not eval_top_ks or eval_overlap >= eval_chunk_size,
                key="run_retrieval_eval",
            )
        elif evaluation_view == "Answer accuracy":
            qa_method = st.selectbox("Retrieval method", method_options, key="qa_method")
            qa_stores = st.multiselect(
                "Vector stores", vector_store_options, default=["NumPy (exact)"],
                key="qa_stores", disabled=qa_method == "BM25",
            )
            qa_store_names = ["BM25"] if qa_method == "BM25" else qa_stores
            qa_top_k = st.selectbox("Retrieved chunks (Top K)", (1, 3, 5), index=2, key="qa_top_k")
            qa_chunk_size = st.selectbox("Chunk size (tokens)", (128, 192, 224), index=1, key="qa_chunk")
            qa_overlap = st.selectbox("Overlap (tokens)", (0, 16, 32, 48), index=2, key="qa_overlap")
            qa_model = st.selectbox("Answer model", model_options, key="qa_answer_model", format_func=_model_label)
            qa_judge_model = st.selectbox(
                "Judge model", model_options,
                index=model_options.index(DEFAULT_JUDGE_MODEL) if DEFAULT_JUDGE_MODEL in model_options else 0,
                key="qa_judge_model", format_func=_model_label,
            )
            qa_sample_size = st.slider("Tuning questions", 10, 100, 30, step=10, key="qa_sample_size")
            st.caption("One answer call and one judge call run per question and backend.")
            run_qa_eval = st.button(
                "Run answer evaluation", type="primary", width="stretch",
                disabled=not qa_store_names or qa_overlap >= qa_chunk_size, key="run_qa_eval",
            )
        elif evaluation_view == "Vector-store benchmark":
            benchmark_stores = st.multiselect(
                "Vector stores to compare", vector_store_options, default=list(vector_store_options),
                key="benchmark_stores",
            )
            benchmark_top_ks = st.pills(
                "Top K cutoffs", (1, 3, 5), default=[1, 3, 5], selection_mode="multi",
                key="benchmark_top_ks",
            ) or []
            benchmark_chunk_size = st.selectbox("Chunk size (tokens)", (128, 192, 224), index=1, key="benchmark_chunk")
            benchmark_overlap = st.selectbox("Overlap (tokens)", (0, 16, 32, 48), index=2, key="benchmark_overlap")
            benchmark_repeats = st.slider("Repeated runs", 3, 10, 5, key="benchmark_repeats")
            benchmark_sample_size = st.slider("Tuning questions", 10, 100, 30, step=10, key="benchmark_sample_size")
            st.caption("Measures index build, search latency, RAM, CPU, Recall@K, and NDCG@K.")
            run_backend_benchmark = st.button(
                "Run vector-store benchmark", type="primary", width="stretch",
                disabled=not benchmark_stores or not benchmark_top_ks or benchmark_overlap >= benchmark_chunk_size,
                key="run_backend_benchmark",
            )
        elif evaluation_view == "Hyperparameter tuning":
            tuning_stores = st.multiselect(
                "Vector stores", vector_store_options, default=["NumPy (exact)"], key="tuning_stores"
            )
            tuning_chunk_sizes = st.multiselect(
                "Chunk sizes (tokens)", (128, 192, 224), default=[128, 192, 224], key="tuning_chunk_sizes"
            )
            tuning_overlaps = st.multiselect(
                "Overlaps (tokens)", (0, 16, 32, 48), default=[0, 16, 32], key="tuning_overlaps"
            )
            tuning_top_ks = st.multiselect(
                "Top K values", (1, 3, 5), default=[1, 3, 5], key="tuning_top_ks"
            )
            tuning_repeats = st.slider("Runs per setting", 3, 10, 5, key="tuning_repeats")
            tuning_sample_size = st.slider("Tuning questions", 10, 100, 30, step=10, key="tuning_sample_size")
            valid_tuning_overlap = any(
                overlap_value < size for size in tuning_chunk_sizes for overlap_value in tuning_overlaps
            )
            tuning_run_count = (
                len(tuning_stores) * len(tuning_chunk_sizes) * len(tuning_overlaps)
                * len(tuning_top_ks) * tuning_repeats
            )
            st.caption(f"Estimated index builds: {tuning_run_count:,}; invalid pairs are skipped.")
            run_tuning = st.button(
                "Run hyperparameter tuning", type="primary", width="stretch",
                disabled=not all((tuning_stores, tuning_chunk_sizes, tuning_overlaps, tuning_top_ks, valid_tuning_overlap)),
                key="run_tuning",
            )
        else:
            holdout_stores = st.multiselect(
                "Vector stores for final comparison", vector_store_options,
                default=list(vector_store_options), key="holdout_stores",
            )
            holdout_top_ks = st.pills(
                "Top K cutoffs", (1, 3, 5), default=[1, 3, 5], selection_mode="multi",
                key="holdout_top_ks",
            ) or []
            holdout_chunk_size = st.selectbox("Chunk size (tokens)", (128, 192, 224), index=1, key="holdout_chunk")
            holdout_overlap = st.selectbox("Overlap (tokens)", (0, 16, 32, 48), index=2, key="holdout_overlap")
            st.caption("Uses reserved holdout questions after tuning is complete.")
            run_log = KNOWLEDGE_BASE / "runs" / "runs.jsonl"
            prior_holdout = False
            if run_log.is_file():
                try:
                    prior_holdout = any(
                        (saved_run := json.loads(line)).get("config", {}).get("dataset") == dataset
                        and saved_run.get("config", {}).get("evaluation_split") == "holdout"
                        and saved_run.get("error") is None
                        for line in run_log.read_text(encoding="utf-8").splitlines()
                    )
                except (OSError, json.JSONDecodeError):
                    st.warning("Could not read the run history; holdout protection is unavailable.")
            if prior_holdout:
                st.info("A successful holdout run already exists for this source. Review it below.")
            run_holdout = st.button(
                "Run final holdout once", type="primary", width="stretch",
                disabled=(not holdout_stores or not holdout_top_ks or prior_holdout or holdout_overlap >= holdout_chunk_size),
                key="run_holdout",
            )
    else:
        run_retrieval_eval = run_qa_eval = run_backend_benchmark = run_tuning = run_holdout = False

if api_key_error and workspace == "Ask" and _requires_openrouter_key(
    (*selected_models, *((judge_model,) if judge_enabled else ()), *((rewrite_model,) if rewrite_enabled else ()))
):
    st.sidebar.error(api_key_error)
elif not api_key and workspace == "Ask" and _requires_openrouter_key(
    (*selected_models, *((judge_model,) if judge_enabled else ()), *((rewrite_model,) if rewrite_enabled else ()))
):
    st.sidebar.warning("Set `OPENROUTER_API_KEY` in the environment or the project `.env` file.")
elif not api_key:
    st.sidebar.caption("Direct provider keys are read from their environment variables.")
if catalog_error:
    st.sidebar.caption(f"Live model catalog unavailable; showing configured defaults. {catalog_error}")

manifest_path = KNOWLEDGE_BASE / "manifest.json"
if not manifest_path.is_file():
    st.error("The knowledge base is missing. Restore its source and evaluation inputs, then run `python -m f1_rag.ingest`.")
    st.stop()
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_total = manifest["documents"][dataset]
    qa_total = manifest["documents"]["evaluation"]
except (OSError, KeyError, ValueError) as error:
    st.error(f"The knowledge-base manifest is invalid: {error}")
    st.stop()


def _prepare_experiment(
    chunk_size: int,
    overlap: int,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunking_method: str = FIXED_CHUNKING,
) -> tuple[list[dict[str, Any]], Any, Any]:
    corpus_path = resolve_corpus_path(dataset)
    corpus_stat = corpus_path.stat()
    chunks, vectors = embedded_corpus(
        dataset, *embedding_signature(model_name), chunk_size, overlap, chunking_method,
        corpus_stat.st_size, corpus_stat.st_mtime_ns, corpus_total,
    )
    query_embedder = embedding_model(*embedding_signature(model_name))
    return chunks, vectors, lambda text: encode_query(query_embedder, text)


def _benchmark_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        values = item["results"].get("metrics_at_k", {})
        row = {
            "Run ID": item["run_id"], "Dataset": item["config"]["dataset"],
            "Vector store": item["config"]["store_name"], "Chunk size": item["config"]["chunk_size"],
            "Overlap": item["config"]["overlap"], "Search limit": item["config"]["top_k"],
            "Records": item["results"]["documents"],
            "Questions": item["results"]["queries"], "Repeats": item["results"]["successful_repeats"],
            "Index build (s)": item["results"]["index_build_seconds_median"],
            "Search p50 (ms)": item["results"]["search_latency_ms_p50"],
            "Search p95 (ms)": item["results"]["search_latency_ms_p95"],
            "RAM delta": _format_bytes(item["results"]["rss_delta_bytes_median"]),
            "Index CPU (s)": _format_seconds(item["results"]["index_build_cpu_seconds_median"]),
            "Search CPU (s/repeat)": _format_seconds(item["results"]["search_cpu_seconds_median"]),
            "Relevance labels": item["results"]["relevance_level"], "Error": item["error"],
        }
        for cutoff, metrics in values.items():
            row[f"Recall@{cutoff}"] = metrics["recall"]
            row[f"NDCG@{cutoff}"] = metrics["ndcg"]
        rows.append(row)
    return pd.DataFrame(rows)


def _retrieval_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        row = {key: value for key, value in item.items() if key != "metrics_at_k"}
        for cutoff, metrics in item["metrics_at_k"].items():
            row[f"Recall@{cutoff}"] = metrics["recall"]
            row[f"NDCG@{cutoff}"] = metrics["ndcg"]
        rows.append(row)
    return pd.DataFrame(rows)


def _record_evaluation(
    test_type: str,
    configuration: dict[str, Any],
    frame: pd.DataFrame,
    errors: list[str],
) -> None:
    try:
        rows = [
            {str(key): value for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
        save_evaluation_run(
            KNOWLEDGE_BASE / "runs",
            test_type=test_type,
            dataset=dataset,
            configuration=configuration,
            rows=rows,
            errors=errors,
        )
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Could not save this evaluation run: {error}")


def _show_export_buttons(frame: pd.DataFrame, title: str, key_prefix: str) -> None:
    csv_column, pdf_column = st.columns(2)
    csv_column.download_button(
        "Download CSV",
        data=dataframe_to_csv(frame),
        file_name=f"{key_prefix}.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
        on_click="ignore",
    )
    pdf_column.download_button(
        "Download PDF",
        data=dataframe_to_pdf(frame, title),
        file_name=f"{key_prefix}.pdf",
        mime="application/pdf",
        key=f"{key_prefix}_pdf",
        on_click="ignore",
    )


def _show_metric_table(results: list[dict[str, Any]], alt: str) -> None:
    if results:
        frame = _benchmark_table(results)
        st.dataframe(frame, hide_index=True, width="stretch", alt=alt)
        _show_export_buttons(frame, alt, "benchmark-results")
        st.caption(
            "Lower latency, RAM, and CPU use reduce resource cost. Higher Recall@K and NDCG@K improve retrieval. "
            "Resource values cover index construction and search in this Python process; they exclude embedding generation."
        )


def _record_evaluation_run(
    evaluation: str,
    dataset: str,
    results: list[dict[str, Any]],
    errors: list[str] | None = None,
) -> None:
    errors = errors or []
    failed_results = sum(result.get("error") is not None for result in results)
    if not results:
        status = "Failed"
    elif errors or failed_results:
        status = "Partial"
    else:
        status = "Completed"

    summaries = []
    for result in results[:3]:
        backend = result.get("Search backend") or result.get("config", {}).get("store_name", "Result")
        if "QA accuracy" in result:
            summary = f"QA accuracy {result['QA accuracy']:.1%} ({result['Questions judged']} questions)"
        else:
            metrics = result.get("metrics_at_k")
            if metrics is None:
                metrics = result.get("results", {}).get("metrics_at_k", {})
            metric_text = [
                f"R@{cutoff} {values['recall']:.3f}, NDCG@{cutoff} {values['ndcg']:.3f}"
                for cutoff, values in metrics.items()
                if values.get("recall") is not None and values.get("ndcg") is not None
            ]
            summary = "; ".join(metric_text) or "No metric results"
        summaries.append(f"{backend}: {summary}")
    if len(results) > 3:
        summaries.append(f"+{len(results) - 3} more")
    if errors:
        summaries.append(f"{len(errors)} error(s)")

    history = list(st.session_state.get("evaluation_run_history", []))
    history.append({
        "id": uuid4().hex,
        "Time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "Evaluation": evaluation,
        "Dataset": dataset,
        "Status": status,
        "Results": " | ".join(summaries) or "No results",
        "Details": results,
        "Errors": errors,
    })
    st.session_state["evaluation_run_history"] = history[-3:]


def _open_evaluation_run() -> None:
    click = st.session_state["evaluation_run_open"]
    recent_runs = list(reversed(st.session_state.get("evaluation_run_history", [])))
    if 0 <= click.row < len(recent_runs):
        st.session_state["selected_evaluation_run_id"] = recent_runs[click.row]["id"]


def _show_recent_evaluation_runs() -> None:
    st.markdown("#### Last 3 evaluation runs")
    history = [dict(run) for run in st.session_state.get("evaluation_run_history", [])]
    if history:
        for run in history:
            run.setdefault("id", uuid4().hex)
            run.setdefault("Errors", [])
            run["_has_details"] = "Details" in run
            run.setdefault("Details", [])
        st.session_state["evaluation_run_history"] = history[-3:]
        recent_runs = list(reversed(history))
        table = pd.DataFrame([
            {
                key: run[key]
                for key in ("Time", "Evaluation", "Dataset", "Status", "Results")
            } | {"Open": "View"}
            for run in recent_runs
        ])
        st.dataframe(
            table,
            column_config={
                "Open": st.column_config.ButtonColumn(
                    "Details", type="primary", on_click=_open_evaluation_run,
                    key="evaluation_run_open",
                ),
            },
            hide_index=True,
            width="stretch",
            alt="Summary of the three most recent evaluation runs",
        )
        selected_id = st.session_state.get("selected_evaluation_run_id")
        selected_run = next((run for run in recent_runs if run["id"] == selected_id), None)
        if selected_run:
            with st.expander(
                f"{selected_run['Evaluation']} — {selected_run['Time']}", expanded=True
            ):
                st.caption(f"{selected_run['Dataset']} · {selected_run['Status']}")
                for error in selected_run["Errors"]:
                    st.error(error)
                results = selected_run["Details"]
                if not selected_run["_has_details"]:
                    st.info("Full results are available for runs performed after this update.")
                elif results:
                    if selected_run["Evaluation"] == "Retrieval quality":
                        rows = []
                        for item in results:
                            row = {key: value for key, value in item.items() if key != "metrics_at_k"}
                            for cutoff, metrics in item["metrics_at_k"].items():
                                row[f"Recall@{cutoff}"] = metrics["recall"]
                                row[f"NDCG@{cutoff}"] = metrics["ndcg"]
                            rows.append(row)
                        detail_table = pd.DataFrame(rows)
                    elif selected_run["Evaluation"] == "Answer accuracy":
                        detail_table = pd.DataFrame(results)
                    else:
                        detail_table = _benchmark_table(results)
                    st.dataframe(
                        detail_table,
                        hide_index=True,
                        width="stretch",
                        alt=f"Full results for {selected_run['Evaluation']}",
                    )
                else:
                    st.info("This run has no result rows.")
    else:
        st.info("The latest evaluation runs will appear here.")


def _show_saved_evaluation_history(selected_dataset: str) -> None:
    try:
        records = load_evaluation_runs(KNOWLEDGE_BASE / "runs", dataset=selected_dataset)
    except (OSError, ValueError) as error:
        st.error(f"Could not load saved evaluation history: {error}")
        return
    if not records:
        return
    with st.expander("Saved test history", expanded=False):
        history = pd.DataFrame([
            {
                "Run ID": record["run_id"],
                "Saved at (UTC)": record["created_at"],
                "Test": record["test_type"],
                "Dataset": record["dataset"],
                "Result rows": len(record["rows"]),
                "Errors": len(record.get("errors", [])),
            }
            for record in records
        ])
        st.dataframe(history, hide_index=True, width="stretch", alt="Saved evaluation test runs")
        _show_export_buttons(history, "Saved evaluation test history", "evaluation-history")
        selected_index = st.selectbox(
            "Open a saved test",
            range(len(records)),
            format_func=lambda index: (
                f"{records[index]['created_at']} · {records[index]['test_type']} · "
                f"{len(records[index]['rows'])} rows"
            ),
            key=f"saved_evaluation_run_{selected_dataset}",
        )
        selected = records[selected_index]
        st.caption("Saved test configuration")
        st.json(selected.get("configuration", {}))
        saved_frame = pd.DataFrame(selected["rows"])
        if not saved_frame.empty:
            st.dataframe(
                saved_frame, hide_index=True, width="stretch", alt="Selected saved evaluation results"
            )
            _show_export_buttons(
                saved_frame,
                f"{selected['test_type']} evaluation · {selected['created_at']}",
                f"saved-{selected['run_id']}",
            )
        else:
            st.info("This saved test has no result rows.")
        for error in selected.get("errors", []):
            st.error(str(error))


def _evaluation_rows_for_source() -> list[dict[str, Any]]:
    return _dataset_questions(dataset)

if workspace == "Ask":
    work_column, metrics_rail = st.columns([4.2, 1], gap="large", vertical_alignment="top")
    with work_column:
        if dataset == "jolpica":
            st.info(f"Knowledge source: {JOLPICA_ATTRIBUTION}. Retrieved records retain their source links.")
        st.subheader(":material/search: Ask the F1 knowledge base")
        st.caption("Ask once, then compare the evidence and answers from each selected retrieval setup.")
        with st.form("query_form", clear_on_submit=False, border=True):
            question = st.text_area(
                "Question", placeholder="Who won the 1976 Formula One World Championship?",
                height=90, label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Retrieve and compare", type="primary", width="stretch")

        if submitted:
            if not question.strip():
                st.error("Enter a question before running the comparison.")
            elif not store_names:
                st.error("Select at least one vector store.")
            elif not selected_models:
                st.error("Select at least one answer model.")
            elif overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than chunk size.")
            elif _requires_openrouter_key(
                (
                    *selected_models,
                    *((judge_model,) if judge_enabled else ()),
                    *((rewrite_model,) if rewrite_enabled and not direct_source_search else ()),
                )
            ) and not api_key:
                st.error("Configure the API key for the selected answer, judge, or rewrite model.")
            else:
                use_rewrite = rewrite_enabled and not direct_source_search
                use_reranking = reranking_enabled and not direct_source_search
                payload: dict[str, Any] = {
                    "question": question.strip(), "query": question.strip(), "stores": list(store_names),
                    "models": list(selected_models), "retrieval_method": retrieval_method,
                    "answer_mode": answer_mode,
                    "chunking_method": chunking_method,
                    "hits": {}, "answers": {}, "errors": {}, "chunks": {},
                    "config": {
                        "dataset": dataset_title,
                        "embedding_model": None if direct_source_search else selected_embedding_model,
                        "chunking_method": chunking_method,
                        "chunk_size": chunk_size, "overlap": overlap, "top_k": top_k,
                        "reranking_enabled": use_reranking,
                        "reranker_model": reranker_name if use_reranking else None,
                        "candidate_count": reranker_candidate_count if use_reranking else top_k,
                    },
                }
                query_vector = None
                try:
                    if use_rewrite:
                        with st.spinner("Rewriting the search query…"):
                            payload["query"], rewrite_error = rewrite_query_safely(
                                question.strip(), rewrite_model, corpus_preview(dataset), api_key
                            )
                            if rewrite_error:
                                payload["errors"]["rewrite"] = rewrite_error
                    if not direct_source_search:
                        with st.spinner("Loading the embedding model…"):
                            embedder = embedding_model(*embedding_signature(selected_embedding_model))
                        query_vector = encode_query(embedder, payload["query"])
                except (RuntimeError, ValueError, OSError) as error:
                    LOGGER.exception("Failed to load the embedding model or encode the query.")
                    payload["errors"]["retrieval"] = str(error)
                    query_vector = None

                if query_vector is not None or direct_source_search:
                    selected_reranker = None
                    if use_reranking:
                        try:
                            with st.spinner(f"Loading reranker {reranker_name}…"):
                                selected_reranker = load_selected_reranker(reranker_name)
                        except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                            payload["errors"]["reranking"] = str(error)
                            payload["config"]["reranking_enabled"] = False
                    for store_name in store_names:
                        try:
                            with st.spinner(f"Searching {store_name}… (the first run builds its index)"):
                                candidate_limit = reranker_candidate_count if selected_reranker else top_k
                                candidates, count = _get_hits(
                                    dataset, store_name, chunk_size, overlap, query_vector, candidate_limit,
                                    payload["query"], retrieval_method, selected_embedding_model,
                                    chunking_method,
                                )
                            if selected_reranker:
                                hits = rerank_candidates(
                                    selected_reranker, payload["query"], candidates, top_k
                                )
                            else:
                                hits = candidates[:top_k]
                            payload["hits"][store_name] = hits
                            payload["chunks"][store_name] = count
                        except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                            payload["errors"][store_name] = str(error)

                    for store_name, hits in payload["hits"].items():
                        for model in selected_models:
                            key = f"{store_name}::{model}"
                            try:
                                with st.spinner(f"Generating with {_model_label(model)} · {store_name}…"):
                                    context = _context_window(model)
                                    response = answer_question(
                                        payload["question"], hits, model, api_key,
                                        context_length=context if isinstance(context, int) else None,
                                    )
                                payload["answers"][key] = {"text": response, "judge": None}
                            except (RuntimeError, ValueError) as error:
                                payload["errors"][key] = str(error)
                                continue

                            if judge_enabled:
                                try:
                                    with st.spinner(f"Judging {_model_label(model)} · {store_name}…"):
                                        evidence = "\n\n".join(
                                            f"Source: {hit.get('source', 'unknown')}\n"
                                            f"Source URL: {hit.get('source_url', '')}\n{hit.get('text', '')}"
                                            for hit in hits
                                        )
                                        payload["answers"][key]["judge"] = judge_answer(
                                            payload["question"], evidence, response, judge_model, api_key
                                        )
                                except (RuntimeError, ValueError) as error:
                                    payload["errors"][f"{key} judge"] = str(error)
                st.session_state["last_comparison"] = payload

        payload = st.session_state.get("last_comparison")
        if payload:
            st.markdown(f"**Question:** {payload['question']}")
            config = payload.get("config", {})
            embedding_label = config.get("embedding_model") or "no embedding model"
            st.caption(
                f"{config.get('dataset', dataset_title)} · {payload.get('answer_mode', 'RAG')} · "
                f"{embedding_label} · {payload.get('retrieval_method', 'Dense')} · "
                f"{payload.get('chunking_method', FIXED_CHUNKING)} · "
                f"chunk {config.get('chunk_size', '—')} / overlap {config.get('overlap', '—')} / top {config.get('top_k', '—')} · "
                f"reranking {'on' if config.get('reranking_enabled') else 'off'}"
            )
            if payload.get("query") != payload.get("question"):
                st.caption(f"Rewritten retrieval query: {payload['query']}")
            if payload["errors"].get("rewrite"):
                st.warning(payload["errors"]["rewrite"])
            if payload["errors"].get("reranking"):
                st.warning(f"Reranking failed; showing retrieval order. {payload['errors']['reranking']}")
            if payload["errors"].get("retrieval"):
                st.error(payload["errors"]["retrieval"])
            displayed_stores = payload.get("stores", list(payload["hits"]))
            displayed_models = payload.get("models", selected_models)
            if not displayed_stores:
                st.info("No retrieval setup was included in this comparison.")
            else:
                columns = st.columns(len(displayed_stores))
                for column, store_name in zip(columns, displayed_stores):
                    with column:
                        st.markdown(f"### {store_name}")
                        store_hits = payload["hits"].get(store_name, [])
                        has_answer_with_evidence = False
                        if store_name in payload["chunks"]:
                            st.caption(f"{payload['chunks'][store_name]:,} chunks · {len(store_hits)} retrieved")
                        elif store_name in payload["errors"]:
                            st.error(f"Search failed: {payload['errors'][store_name]}")
                        elif payload["errors"].get("retrieval"):
                            st.info("Search did not run because query preparation failed.")
                        for model in displayed_models:
                            key = f"{store_name}::{model}"
                            st.markdown(f"**{_model_label(model)}**")
                            result = payload["answers"].get(key)
                            if result:
                                if result["text"] == ANSWER_NOT_FOUND:
                                    st.info("I couldn't find the answer in the selected source.")
                                else:
                                    st.write(result["text"])
                                    _render_source_citations(store_hits)
                                    has_answer_with_evidence = True
                                verdict = result.get("judge")
                                if verdict:
                                    st.badge(
                                        "Supported" if verdict["correct"] else "Needs review",
                                        icon=":material/check_circle:" if verdict["correct"] else ":material/warning:",
                                        color="green" if verdict["correct"] else "orange",
                                    )
                                    st.caption(f"Judge note: {verdict['reason']}")
                            elif key in payload["errors"]:
                                st.error(f"Answer failed: {payload['errors'][key]}")
                        if store_hits and has_answer_with_evidence:
                            with st.expander(f"Retrieved evidence · {store_name}"):
                                for rank, hit in enumerate(store_hits, start=1):
                                    st.markdown(
                                        f"**[{rank}] {hit.get('source', 'Unknown source')}** · score `{hit.get('score', 0):.3f}`"
                                    )
                                if "rerank_score" in hit:
                                    st.caption(
                                        f"Cross-encoder score: {hit['rerank_score']:.3f} · "
                                        f"retrieval rank: {hit['retrieval_rank']}"
                                    )
                                    source_url = hit.get("source_url")
                                    if source_url and source_url.startswith(JOLPICA_API_BASE + "/"):
                                        st.markdown(f"[View {JOLPICA_ATTRIBUTION} record]({source_url})")
                                    elif str(hit.get("source", "")).startswith("Wikipedia:"):
                                        wikipedia_url = source_url or wikipedia_source_url(
                                            str(hit["source"]).removeprefix("Wikipedia:")
                                        )
                                        st.markdown(f"[View Wikipedia article]({wikipedia_url})")
                                    st.write(hit.get("text", ""))
                                    if rank < len(store_hits):
                                        st.divider()
            for key, message in payload["errors"].items():
                if key == "retrieval" or key.endswith(" judge"):
                    st.caption(f"{key}: {message}")
    with metrics_rail:
        st.caption("SESSION SNAPSHOT")
        st.metric("Selected source", dataset_title, border=True)
        st.metric("Source documents", f"{corpus_total:,}", border=True)
        st.metric("Evaluation Q&A", f"{qa_total:,}", border=True)
        st.metric("Chunk / overlap", f"{chunk_size} / {overlap}", border=True)
        st.caption("[by Mojtaba](https://github.com/Sadatpour)")

elif workspace == "Evaluation & benchmark":
    evaluation_pool = _evaluation_rows_for_source()
    tuning_rows = _evaluation_split(evaluation_pool, "tune")
    holdout_rows = _evaluation_split(evaluation_pool, "test")
    st.subheader(":material/query_stats: Evaluation and benchmark")
    st.caption(
        f"{dataset_title} · {len(evaluation_pool):,} of {qa_total:,} QA questions match the corpus; "
        f"{len(tuning_rows):,} are available for tuning and evaluation. Completed runs are saved locally in "
        "`knowledge_base/runs/evaluations.jsonl`; result tables can be downloaded as CSV or PDF."
    )

    if evaluation_view == "Retrieval quality":
        st.markdown("#### Retrieval quality by Top K")
        st.caption("Recall@K and NDCG@K use source labels from qa.jsonl. These metrics measure retrieval, not answer generation.")
        if run_retrieval_eval:
            st.session_state["retrieval_eval_results"] = []
            st.session_state["retrieval_eval_errors"] = []
            if not tuning_rows:
                st.error("No labeled questions match the selected source.")
                _record_evaluation_run("Retrieval quality", dataset, [], ["No labeled questions"])
                _record_evaluation(
                    "retrieval_quality",
                    {"method": eval_method, "stores": eval_store_names},
                    pd.DataFrame(),
                    ["No labeled questions"],
                )
            else:
                sample = stratified_sample(tuning_rows, min(eval_sample_size, len(tuning_rows)), seed=42)
                results = []
                errors = []
                progress = st.progress(0, text="Running retrieval evaluation…")
                try:
                    for index, store_name in enumerate(eval_store_names):
                        try:
                            metrics = _run_retrieval_evaluation(
                                dataset, store_name, eval_chunk_size, eval_overlap,
                                sample, eval_top_ks, eval_method, selected_embedding_model,
                            )
                            results.append({
                                "Search backend": store_name, "Retrieval method": eval_method,
                                "Chunk size": eval_chunk_size, "Overlap": eval_overlap, **metrics,
                            })
                        except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                            errors.append(f"{store_name}: {error}")
                        progress.progress((index + 1) / len(eval_store_names), text=f"Evaluated {store_name}")
                finally:
                    progress.empty()
                st.session_state["retrieval_eval_results"] = results
                frame = _retrieval_table(results)
                st.session_state["retrieval_eval_errors"] = errors
                _record_evaluation_run("Retrieval quality", dataset, results, errors)
                _record_evaluation(
                    "retrieval_quality",
                    {
                        "method": eval_method, "stores": eval_store_names,
                        "chunk_size": eval_chunk_size, "overlap": eval_overlap,
                        "top_ks": eval_top_ks, "sample_size": len(sample), "seed": 42,
                    },
                    frame,
                    errors,
                )
        results = st.session_state.get("retrieval_eval_results", [])
        if results:
            frame = _retrieval_table(results)
            st.dataframe(
                frame, hide_index=True, width="stretch",
                alt="Retrieval Recall and NDCG by vector store and Top K",
            )
            _show_export_buttons(frame, "Retrieval quality results", "retrieval-results")
        for error in st.session_state.get("retrieval_eval_errors", []):
            st.error(error)

    elif evaluation_view == "Answer accuracy":
        st.markdown("#### Generated answer accuracy")
        st.caption("The judge compares each generated answer with the matching reference answer in qa.jsonl.")
        if run_qa_eval:
            st.session_state["answer_eval_results"] = []
            st.session_state["answer_eval_errors"] = []
            if not tuning_rows:
                st.error("No labeled questions match the selected source.")
                _record_evaluation_run("Answer accuracy", dataset, [], ["No labeled questions"])
                _record_evaluation(
                    "answer_accuracy",
                    {"method": qa_method, "stores": qa_store_names},
                    pd.DataFrame(),
                    ["No labeled questions"],
                )
            elif _requires_openrouter_key((qa_model, qa_judge_model)) and not api_key:
                st.error("Configure the API key for the selected answer and judge models.")
                _record_evaluation_run("Answer accuracy", dataset, [], ["Missing API key"])
                _record_evaluation(
                    "answer_accuracy",
                    {"method": qa_method, "stores": qa_store_names},
                    pd.DataFrame(),
                    ["Missing API key"],
                )
            else:
                sample = stratified_sample(tuning_rows, min(qa_sample_size, len(tuning_rows)), seed=7)
                errors = []
                results = []
                progress = st.progress(0, text="Preparing answer evaluation…")
                total = len(sample) * len(qa_store_names)
                completed = 0
                try:
                    embedder = embedding_model(*embedding_signature(selected_embedding_model))
                    encoded = [encode_query(embedder, str(row["question"])) for row in sample]
                    for store_name in qa_store_names:
                        verdicts = []
                        store_error = None
                        record_count = 0
                        try:
                            for row, query_vector in zip(sample, encoded, strict=True):
                                hits, record_count = _get_hits(
                                    dataset, store_name, qa_chunk_size, qa_overlap, query_vector,
                                    qa_top_k, str(row["question"]), qa_method, selected_embedding_model,
                                )
                                answer = answer_question(
                                    str(row["question"]), hits, qa_model, api_key,
                                    context_length=_context_window(qa_model),
                                )
                                verdict = judge_answer(
                                    str(row["question"]), str(row.get("answer", "")),
                                    answer, qa_judge_model, api_key,
                                )
                                verdicts.append(verdict)
                                completed += 1
                                progress.progress(completed / total, text=f"{store_name}: {len(verdicts)}/{len(sample)}")
                        except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                            store_error = f"{store_name}: {error}"
                            errors.append(store_error)
                        if verdicts:
                            results.append({
                                "Search backend": store_name, "Retrieval method": qa_method,
                                "Top K": qa_top_k, "Chunk size": qa_chunk_size, "Overlap": qa_overlap,
                                "Records": record_count,
                                "QA accuracy": qa_accuracy(verdicts), "Questions judged": len(verdicts),
                                "Answer model": qa_model, "Judge model": qa_judge_model,
                                "Error": store_error or "",
                            })
                    st.session_state["answer_eval_results"] = results
                    st.session_state["answer_eval_errors"] = errors
                except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                    st.error(f"Answer evaluation failed: {error}")
                    errors.append(f"Answer evaluation failed: {error}")
                finally:
                    progress.empty()
                st.session_state["answer_eval_results"] = results
                st.session_state["answer_eval_errors"] = errors
                _record_evaluation_run("Answer accuracy", dataset, results, errors)
                _record_evaluation(
                    "answer_accuracy",
                    {
                        "method": qa_method, "stores": qa_store_names, "top_k": qa_top_k,
                        "chunk_size": qa_chunk_size, "overlap": qa_overlap,
                        "answer_model": qa_model, "judge_model": qa_judge_model,
                        "sample_size": len(sample), "seed": 7,
                    },
                    pd.DataFrame(results),
                    errors,
                )
        results = st.session_state.get("answer_eval_results", [])
        if results:
            frame = pd.DataFrame(results)
            st.dataframe(
                frame, hide_index=True, width="stretch",
                alt="LLM judged answer accuracy across retrieval backends",
            )
            _show_export_buttons(frame, "Generated answer accuracy results", "answer-accuracy-results")
        for error in st.session_state.get("answer_eval_errors", []):
            st.error(error)

    elif evaluation_view == "Vector-store benchmark":
        st.markdown("#### Vector-store resource and retrieval comparison")
        st.caption(
            "Every backend receives the same chunks, document vectors, and questions. Search latency excludes query embedding. "
            "RAM and CPU are measured in the app process while building and searching the index."
        )
        if run_backend_benchmark:
            if not tuning_rows:
                st.error("No labeled questions match the selected source.")
                _record_evaluation_run("Vector-store benchmark", dataset, [], ["No labeled questions"])
                _record_evaluation(
                    "vector_store_benchmark",
                    {"stores": benchmark_stores, "top_ks": benchmark_top_ks},
                    pd.DataFrame(),
                    ["No labeled questions"],
                )
            else:
                sample = stratified_sample(
                    tuning_rows, min(benchmark_sample_size, len(tuning_rows)), seed=20261117
                )
                st.session_state["backend_benchmark_results"] = []
                results = []
                errors = []
                status = st.status("Running vector-store benchmark…", expanded=True)
                try:
                    results = run_hyperparameter_grid(
                        dataset=dataset, store_names=benchmark_stores,
                        embedding_model=embedding_profile(selected_embedding_model)["repo_id"],
                        chunk_sizes=(benchmark_chunk_size,), overlaps=(benchmark_overlap,),
                        top_ks=(max(benchmark_top_ks),), metric_ks=benchmark_top_ks,
                        repeats=benchmark_repeats,
                        prepare=lambda size, overlap: _prepare_experiment(size, overlap, selected_embedding_model),
                        queries=sample, output_dir=KNOWLEDGE_BASE / "runs",
                    )
                    st.session_state["backend_benchmark_results"] = results
                    errors.extend(
                        f"{result['config']['store_name']}: {result['error']}"
                        for result in results if result["error"]
                    )
                    status.update(
                        label="Vector-store benchmark completed with errors." if errors
                        else "Vector-store benchmark complete.",
                        state="error" if errors else "complete",
                        expanded=bool(errors),
                    )
                except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                    st.error(f"Vector-store benchmark failed: {error}")
                    errors.append(str(error))
                    status.update(label="Vector-store benchmark failed.", state="error", expanded=True)
                _record_evaluation_run("Vector-store benchmark", dataset, results, errors)
                _record_evaluation(
                    "vector_store_benchmark",
                    {
                        "stores": benchmark_stores, "top_ks": benchmark_top_ks,
                        "chunk_size": benchmark_chunk_size, "overlap": benchmark_overlap,
                        "repeats": benchmark_repeats, "sample_size": len(sample), "seed": 20261117,
                    },
                    _benchmark_table(results),
                    errors,
                )
        results = st.session_state.get("backend_benchmark_results", [])
        _show_metric_table(
            results, "Vector-store benchmark with resource use and retrieval quality at multiple Top K values"
        )
        if results:
            st.caption("Run IDs and hardware/software details are saved in knowledge_base/runs/runs.jsonl.")
    elif evaluation_view == "Hyperparameter tuning":
        st.markdown("#### Tune chunking and retrieval settings")
        st.caption("Use tuning questions to choose settings. Each record includes its backend and full configuration.")
        if run_tuning:
            if not tuning_rows:
                st.error("No labeled questions match the selected source.")
                _record_evaluation_run("Hyperparameter tuning", dataset, [], ["No labeled questions"])
                _record_evaluation(
                    "hyperparameter_tuning",
                    {"stores": tuning_stores, "chunk_sizes": tuning_chunk_sizes},
                    pd.DataFrame(),
                    ["No labeled questions"],
                )
            else:
                sample = stratified_sample(tuning_rows, min(tuning_sample_size, len(tuning_rows)), seed=20261117)
                st.session_state["tuning_results"] = []
                results = []
                errors = []
                status = st.status("Running hyperparameter tuning…", expanded=True)
                try:
                    results = run_hyperparameter_grid(
                        dataset=dataset, store_names=tuning_stores,
                        embedding_model=embedding_profile(selected_embedding_model)["repo_id"],
                        chunk_sizes=tuning_chunk_sizes, overlaps=tuning_overlaps,
                        top_ks=tuning_top_ks, repeats=tuning_repeats,
                        prepare=lambda size, overlap: _prepare_experiment(size, overlap, selected_embedding_model),
                        queries=sample,
                        output_dir=KNOWLEDGE_BASE / "runs",
                    )
                    st.session_state["tuning_results"] = results
                    errors.extend(
                        f"{result['config']['store_name']}: {result['error']}"
                        for result in results if result["error"]
                    )
                    status.update(
                        label="Hyperparameter tuning completed with errors." if errors
                        else "Hyperparameter tuning complete.",
                        state="error" if errors else "complete",
                        expanded=bool(errors),
                    )
                except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                    st.error(f"Hyperparameter tuning failed: {error}")
                    errors.append(str(error))
                    status.update(label="Hyperparameter tuning failed.", state="error", expanded=True)
                _record_evaluation_run("Hyperparameter tuning", dataset, results, errors)
                _record_evaluation(
                    "hyperparameter_tuning",
                    {
                        "stores": tuning_stores, "chunk_sizes": tuning_chunk_sizes,
                        "overlaps": tuning_overlaps, "top_ks": tuning_top_ks,
                        "repeats": tuning_repeats, "sample_size": len(sample), "seed": 20261117,
                    },
                    _benchmark_table(results),
                    errors,
                )
        results = st.session_state.get("tuning_results", [])
        _show_metric_table(results, "Hyperparameter tuning results by chunk size, overlap, Top K, and vector store")
        if results:
            st.caption("Compare Recall@K and NDCG@K with p50/p95 latency, RAM delta, and CPU before selecting settings.")

    else:
        st.markdown("#### Final holdout comparison")
        st.caption("The reserved split is for the final comparison, after settings are selected on tuning data.")
        if run_holdout:
            st.session_state["holdout_results"] = []
            results = []
            errors = []
            progress = st.progress(0, text="Running final comparison…")
            try:
                if not holdout_rows:
                    raise ValueError("No labeled holdout questions match the selected source.")
                chunks, vectors, query_encoder = _prepare_experiment(
                    holdout_chunk_size, holdout_overlap, selected_embedding_model
                )
                sample = stratified_sample(holdout_rows, len(holdout_rows), seed=20261118)
                for index, store_name in enumerate(holdout_stores):
                    try:
                        result = benchmark_store(
                            BenchmarkConfig(
                                dataset=dataset, store_name=store_name,
                                embedding_model=embedding_profile(selected_embedding_model)["repo_id"],
                                chunk_size=holdout_chunk_size, overlap=holdout_overlap,
                                top_k=max(holdout_top_ks), repeats=1,
                                evaluation_split="holdout", metric_ks=tuple(holdout_top_ks),
                            ),
                            chunks, vectors, sample, query_encoder, KNOWLEDGE_BASE / "runs",
                        )
                        results.append(result)
                        if result["error"]:
                            errors.append(f"{store_name}: {result['error']}")
                    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                        errors.append(f"{store_name}: {error}")
                        st.error(f"{store_name}: {error}")
                    progress.progress((index + 1) / len(holdout_stores), text=f"Measured {store_name}")
                st.session_state["holdout_results"] = results
            except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                errors.append(str(error))
                st.error(f"Final holdout comparison failed: {error}")
            finally:
                progress.empty()
            st.session_state["holdout_results"] = results
            _record_evaluation_run("Final holdout", dataset, results, errors)
            _record_evaluation(
                "final_holdout",
                {
                    "stores": holdout_stores, "top_ks": holdout_top_ks,
                    "chunk_size": holdout_chunk_size, "overlap": holdout_overlap,
                    "sample_size": len(holdout_rows), "seed": 20261118,
                },
                _benchmark_table(results),
                errors,
            )
        results = st.session_state.get("holdout_results", [])
        _show_metric_table(results, "Final holdout benchmark across vector stores")
        if results and all(result["error"] is None for result in results):
            st.success(f"Compared {len(results)} vector stores on {len(holdout_rows):,} held-out questions.")
        elif results:
            st.warning("At least one backend failed. Resolve that backend before relying on the comparison.")
    _show_saved_evaluation_history(dataset)

    _show_recent_evaluation_runs()

else:
    st.subheader(":material/route: Project guide")
    st.markdown(
        """
        - `knowledge_base/corpora/` — Ready-to-use Wikipedia and Jolpica corpora
        - `knowledge_base/evaluation/qa.jsonl` — Reference questions and answers
        - `knowledge_base/indices/` — Rebuildable local indexes
        - `models/embedding/` — Local embedding model files
        - `config/models.toml` — Answer, judge, rewrite, and embedding model settings
        - `.env` — Provider API keys; excluded from Git
        - `src/f1_rag/` — Ingestion, chunking, embeddings, vector stores, retrieval, generation, and evaluation

        **NumPy exact cosine** is the local dense-search baseline. **FAISS** is a similarity-search library. **ChromaDB** is a vector database.
        These are vector-store/index backends. **Dense**, **BM25**, and **Hybrid** are retrieval methods.
        """
    )
    st.info(
        "The index reuses document vectors. Each question is embedded once for search. Retrieval evaluation checks source labels in qa.jsonl; "
        "answer accuracy compares generated answers with its reference answers."
    )
    st.markdown(
        f"Jolpica data attribution: [{JOLPICA_ATTRIBUTION}]({JOLPICA_API_BASE}/). "
        "Retrieved Jolpica evidence includes direct API record links."
    )
