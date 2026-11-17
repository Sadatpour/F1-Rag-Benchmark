from __future__ import annotations

import hashlib
import base64
from http.client import HTTPException
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
TRACK_ARTWORK = base64.b64encode((ROOT / "assets" / "monaco-circuit.svg").read_bytes()).decode("ascii")
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
    except (RuntimeError, TimeoutError, HTTPException, OSError) as error:
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
    retrieval_method: str = "Dense",
) -> tuple[list[dict[str, Any]], int]:
    if retrieval_method in {"BM25", "Hybrid"}:
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
        if retrieval_method == "BM25":
            return sparse_hits, len(sparse.documents)
        dense_store, _, _ = _ready_index(dataset, store_name, chunk_size, overlap)
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
    retrieval_method: str = "Dense",
) -> dict[str, float | int]:
    embedder = embedding_model(*embedding_signature())
    recalls = []
    ndcgs = []
    for row in rows:
        query = encode_query(embedder, str(row["question"]))
        hits, _ = _get_hits(
            dataset,
            store_name,
            chunk_size,
            overlap,
            query,
            10,
            str(row["question"]),
            retrieval_method,
        )
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
    st.markdown("## :material/speed: Apex RAG")
    st.caption("F1 knowledge lab")

    st.markdown("**Knowledge source**")
    dataset_title = st.selectbox("Knowledge source", ("Wikipedia", "Jolpica"), index=0)
    dataset = dataset_title.lower()
    st.markdown("**Retrieval setup**")
    vector_stores = st.multiselect(
        "Vector store / index backends",
        ("NumPy (exact)", "FAISS", "ChromaDB"),
        default=["NumPy (exact)"],
        help=(
            "Choose where dense vectors are indexed: NumPy is the local exact-search baseline, "
            "FAISS is a similarity-search library, and ChromaDB is a vector database."
        ),
    )
    retrieval_method = st.selectbox(
        "Retrieval method",
        ("Dense", "Hybrid", "BM25"),
        help=(
            "Dense searches each selected vector backend. Hybrid combines each selected backend's dense results "
            "with BM25 using RRF. BM25 searches words and runs as one standalone result."
        ),
    )
    store_names = ["BM25"] if retrieval_method == "BM25" else list(vector_stores)
    if retrieval_method == "BM25":
        st.caption("BM25 is independent of the selected vector databases and produces one result column.")
    st.markdown("**Answer models**")
    selected_models = st.multiselect(
        "Answer models",
        model_options,
        default=["rotation::answer"] if "rotation::answer" in model_options else [],
        help="A model is queried once per selected vector store.",
        format_func=_model_label,
    )
    top_k = st.slider("Retrieved context count", min_value=1, max_value=10, value=5)

    with st.expander("Advanced settings", expanded=False):
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
        chunk_size = st.select_slider(
            "Chunk size (embedding tokens)",
            options=(128, 192, 224),
            value=192,
            help="Tokenized with the embedding model tokenizer; each chunk stays within its 256-token input limit.",
        )
        overlap = st.select_slider("Chunk overlap (tokens)", options=(0, 16, 32, 48), value=32)
        if overlap >= chunk_size:
            st.error("Overlap must be smaller than the selected chunk size.")
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

if dataset == "jolpica":
    st.info(
        f"Data source: [{JOLPICA_ATTRIBUTION}]({JOLPICA_API_BASE}/). "
        "Retrieved records link to their corresponding API endpoints."
    )

work_column, metrics_rail = st.columns([4.2, 1], gap="large", vertical_alignment="top")
with work_column:
    st.subheader(":material/travel_explore: Ask the F1 knowledge base")
    st.caption("Ask once, then compare how each retrieval setup supports the answer with evidence.")
    with st.form("query_form", clear_on_submit=False, border=True):
        question = st.text_area(
            "Question",
            placeholder="Who won the 1976 Formula One World Championship?",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Retrieve & compare", type="primary", width="stretch")
    query_tab, eval_tab, experiments_tab, about_tab = st.tabs(
        (
            ":material/search: Compare answers",
            ":material/query_stats: Evaluation",
            ":material/query_stats: Benchmarks",
            ":material/folder: Project guide",
        )
    )
with metrics_rail:
    st.markdown("**SESSION SNAPSHOT**")
    st.metric("Selected source", dataset_title, border=True)
    st.metric("Source documents", f"{corpus_total:,}", border=True)
    st.metric("Evaluation Q&A", f"{qa_total:,}", border=True)
    st.metric("Chunk / overlap", f"{chunk_size} / {overlap}", border=True)

with query_tab:
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
                "stores": list(store_names),
                "models": list(selected_models),
                "retrieval_method": retrieval_method,
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
                                retrieval_method,
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
        displayed_stores = payload.get("stores", list(payload["hits"]))
        displayed_models = payload.get("models", selected_models)
        displayed_method = payload.get("retrieval_method", "Dense")
        st.markdown(f"**Question:** {payload['question']}")
        st.caption(f"Retrieval method: {displayed_method}")
        if payload["query"] != payload["question"]:
            st.caption(f"Rewritten retrieval query: {payload['query']}")
        if payload["errors"].get("retrieval"):
            st.error(payload["errors"]["retrieval"])
        if not displayed_stores:
            st.info("No retrieval setup was included in this comparison.")
        else:
            store_columns = st.columns(len(displayed_stores))
            for column, store_name in zip(store_columns, displayed_stores):
                with column:
                    st.markdown(f"### {store_name}")
                    store_hits = payload["hits"].get(store_name, [])
                    if store_name in payload["chunks"]:
                        st.caption(
                            f"{payload['chunks'][store_name]:,} chunks · top {len(store_hits)} results"
                        )
                    elif store_name in payload["errors"]:
                        st.error(f"Search failed: {payload['errors'][store_name]}")
                    elif payload["errors"].get("retrieval"):
                        st.info("Search did not run because query preparation failed.")
                    else:
                        st.info("No search results were returned for this database.")

                    for model in displayed_models:
                        key = f"{store_name}::{model}"
                        st.markdown(f"**{_model_label(model)}**")
                        result = payload["answers"].get(key)
                        if result:
                            st.write(result["text"])
                            _render_source_citations(store_hits)
                            verdict = result["judge"]
                            if verdict:
                                st.badge(
                                    "Correct" if verdict["correct"] else "Needs review",
                                    icon=":material/check_circle:" if verdict["correct"] else ":material/warning:",
                                    color="green" if verdict["correct"] else "orange",
                                )
                                st.caption(f"Judge note: {verdict['reason']}")
                        elif key in payload["errors"]:
                            st.error(f"Answer failed: {payload['errors'][key]}")
                        elif store_hits:
                            st.info("No answer was generated for this model.")

                    if store_hits:
                        with st.expander(f"Retrieved evidence · {store_name}"):
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

        for key, message in payload["errors"].items():
            if key == "retrieval" or key.endswith(" judge"):
                if key.endswith(" judge") and not judge_enabled:
                    continue
                st.caption(f"{key}: {message}")

with eval_tab:
    st.subheader(":material/query_stats: Retrieval & answer evaluation")
    st.caption(
        "Recall@10 and NDCG@10 measure retrieval quality against source labels. QA accuracy measures generated answers "
        "with an LLM judge. These are different metrics."
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
                    result = _run_retrieval_evaluation(
                        dataset,
                        store_name,
                        chunk_size,
                        overlap,
                        sample,
                        retrieval_method,
                    )
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
    accuracy_stores = st.multiselect(
        "Backends to compare for QA accuracy",
        store_names,
        default=list(store_names),
        key="accuracy_stores",
    )
    st.caption(
        "The same questions, answer model, and judge are used for each selected backend. "
        "This makes one answer-model call and one judge call per question per backend."
    )
    accuracy_button = st.button(
        "Compare QA accuracy",
        disabled=(
            not judge_enabled
            or (_requires_openrouter_key((accuracy_model, judge_model)) and not api_key)
            or not accuracy_stores
        ),
    )
    if accuracy_button:
        if not eval_pool:
            st.error(f"No QA rows map to the {dataset_title} corpus.")
        else:
            sample = stratified_sample(eval_pool, min(accuracy_count, len(eval_pool)), seed=7)
            accuracy_results = []
            accuracy_errors = []
            progress = st.progress(0, text="Comparing QA answers…")
            total_comparisons = len(sample) * len(accuracy_stores)
            completed_comparisons = 0
            try:
                embedder = embedding_model(*embedding_signature())
                for accuracy_store in accuracy_stores:
                    verdicts = []
                    try:
                        for row in sample:
                            query_vector = encode_query(embedder, str(row["question"]))
                            hits, _ = _get_hits(
                                dataset,
                                accuracy_store,
                                chunk_size,
                                overlap,
                                query_vector,
                                top_k,
                                str(row["question"]),
                                retrieval_method,
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
                            completed_comparisons += 1
                            progress.progress(
                                completed_comparisons / total_comparisons,
                                text=f"{accuracy_store}: {len(verdicts)}/{len(sample)} questions",
                            )
                        accuracy_results.append(
                            {
                                "Backend": accuracy_store,
                                "QA accuracy": qa_accuracy(verdicts),
                                "Questions judged": len(verdicts),
                                "Answer model": accuracy_model,
                                "Judge model": judge_model,
                            }
                        )
                    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                        accuracy_errors.append(f"{accuracy_store}: {error}")
                st.session_state["qa_accuracy_results"] = accuracy_results
                st.session_state["qa_accuracy_errors"] = accuracy_errors
            except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                st.error(f"QA accuracy run failed: {error}")
            progress.empty()
    accuracy_results = st.session_state.get("qa_accuracy_results", [])
    if judge_enabled and accuracy_results:
        st.dataframe(
            pd.DataFrame(accuracy_results),
            alt="LLM-judged answer accuracy comparison for selected retrieval backends",
            hide_index=True,
            width="stretch",
        )
    for error in st.session_state.get("qa_accuracy_errors", []):
        st.error(error)

with experiments_tab:
    st.subheader(":material/query_stats: Vector backend benchmarks")
    st.caption(
        "A vector backend is the index/storage implementation. A retrieval method is the search strategy: "
        "Dense, BM25, or Hybrid. This page compares backends with one shared configuration, then tunes RAG parameters separately."
    )
    experiment_rows = _dataset_questions(dataset)
    tuning_rows = _evaluation_split(experiment_rows, "tune")
    holdout_rows = _evaluation_split(experiment_rows, "test")
    vector_store_options = ("NumPy (exact)", "FAISS", "ChromaDB")
    compare_tab, tune_tab, holdout_tab = st.tabs(
        ("Compare vector stores", "Tune hyperparameters", "Final holdout")
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

    def _benchmark_table(results: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Run ID": item["run_id"],
                    "Dataset": item["config"]["dataset"],
                    "Backend": item["config"]["store_name"],
                    "Chunk": item["config"]["chunk_size"],
                    "Overlap": item["config"]["overlap"],
                    "Top K": item["config"]["top_k"],
                    "Questions": item["results"]["queries"],
                    "Relevance labels": item["results"]["relevance_level"],
                    "Repeats": item["results"]["successful_repeats"],
                    "Index build (s)": item["results"]["index_build_seconds_median"],
                    "Search p50 (ms)": item["results"]["search_latency_ms_p50"],
                    "Search p95 (ms)": item["results"]["search_latency_ms_p95"],
                    "RAM change": _format_bytes(item["results"]["rss_delta_bytes_median"]),
                    "Build CPU (s)": _format_seconds(item["results"]["index_build_cpu_seconds_median"]),
                    "Search CPU (s)": _format_seconds(item["results"]["search_cpu_seconds_median"]),
                    "Recall@K": item["results"]["recall_at_k"],
                    "NDCG@K": item["results"]["ndcg_at_k"],
                    "Error": item["error"],
                }
                for item in results
            ]
        )

    with compare_tab:
        st.markdown("#### Same-settings backend comparison")
        st.caption(
            "This compares the storage/index backends with identical documents, embeddings, questions, and settings. "
            "Search p50/p95 measure vector search only; Recall@K and NDCG@K measure retrieval quality. "
            "They do not measure generated-answer accuracy."
        )
        compare_stores = st.multiselect(
            "Backends to compare",
            vector_store_options,
            default=list(vector_store_options),
            key="backend_compare_stores",
        )
        compare_columns = st.columns(4)
        compare_chunk = compare_columns[0].selectbox("Chunk size", (128, 192, 224), index=1, key="backend_compare_chunk")
        compare_overlap = compare_columns[1].selectbox("Overlap", (0, 16, 32, 48), index=2, key="backend_compare_overlap")
        compare_top_k = compare_columns[2].selectbox("Top K", (3, 5, 8, 10), index=1, key="backend_compare_top_k")
        compare_repeats = compare_columns[3].slider("Repeats", 3, 10, 5, key="backend_compare_repeats")
        compare_sample_size = st.slider(
            "Tuning questions", 10, 100, 30, step=10, key="backend_compare_sample"
        )
        if compare_overlap >= compare_chunk:
            st.error("Overlap must be smaller than chunk size.")
        if st.button(
            "Compare selected backends",
            type="primary",
            disabled=(not compare_stores or not tuning_rows or compare_overlap >= compare_chunk),
            key="run_backend_comparison",
        ):
            sample = stratified_sample(tuning_rows, min(compare_sample_size, len(tuning_rows)), seed=20261117)
            with st.spinner("Running the same benchmark for each selected backend…"):
                try:
                    results = run_hyperparameter_grid(
                        dataset=dataset,
                        store_names=compare_stores,
                        embedding_model=EMBEDDING_MODEL_ID,
                        chunk_sizes=(compare_chunk,),
                        overlaps=(compare_overlap,),
                        top_ks=(compare_top_k,),
                        repeats=compare_repeats,
                        prepare=prepare_experiment,
                        queries=sample,
                        output_dir=KNOWLEDGE_BASE / "runs",
                    )
                    st.session_state["backend_compare_results"] = results
                except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                    st.error(f"Backend comparison failed: {error}")
        comparison_results = st.session_state.get("backend_compare_results", [])
        if comparison_results:
            st.dataframe(
                _benchmark_table(comparison_results),
                hide_index=True,
                width="stretch",
                alt="Same-settings vector backend comparison by latency, retrieval quality, RAM, and CPU",
            )
            st.caption(
                "Lower search latency, RAM, and CPU use reduce resource cost. Higher Recall@K and NDCG@K improve retrieval. "
                "No single backend is best on every metric. Results are saved with run IDs and environment details."
            )
        st.info("Generated-answer QA accuracy is evaluated separately in the Evaluation tab with the selected judge model.")

    with tune_tab:
        st.markdown("#### Tune retrieval settings")
        st.caption(
            "Use the tuning split to compare chunk size, overlap, and Top K. The grid reuses the same corpus and question sample "
            "for each backend and setting. Do not choose settings from the holdout results."
        )
        experiment_stores = st.multiselect(
            "Backends to tune",
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
        valid_overlap = any(overlap < size for size in experiment_sizes for overlap in experiment_overlaps)
        estimate = len(experiment_sizes) * len(experiment_overlaps) * len(experiment_top_ks) * len(experiment_stores) * experiment_repeats
        st.caption(f"Approximate index builds: {estimate:,}; invalid overlap/size pairs are skipped.")
        if st.button(
            "Run hyperparameter grid",
            type="primary",
            disabled=not all((experiment_stores, experiment_sizes, experiment_overlaps, experiment_top_ks, tuning_rows, valid_overlap)),
        ):
            tuning_sample = stratified_sample(
                tuning_rows, min(experiment_sample_size, len(tuning_rows)), seed=20261117
            )
            with st.spinner("Measuring the selected settings…"):
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
                _benchmark_table(grid_results),
                alt="Hyperparameter tuning results by backend, latency, retrieval quality, RAM, and CPU",
                hide_index=True,
                width="stretch",
            )
            st.caption("Full results, including run ID and software/hardware metadata, are appended to knowledge_base/runs/runs.jsonl.")

    with holdout_tab:
        st.markdown("#### One-time final comparison")
        st.caption(
            "Select all backends you want to compare before the first run. They use the same held-out questions and one shared configuration. "
            "After a successful holdout run for this dataset, do not tune against it."
        )
        test_stores = st.multiselect(
            "Backends for final comparison",
            vector_store_options,
            default=list(vector_store_options),
            key="holdout_stores",
        )
        holdout_columns = st.columns(3)
        test_chunk_size = holdout_columns[0].selectbox("Chunk size", (128, 192, 224), index=1, key="holdout_chunk_size")
        test_overlap = holdout_columns[1].selectbox("Overlap", (0, 16, 32, 48), index=2, key="holdout_overlap")
        test_top_k = holdout_columns[2].selectbox("Top K", (3, 5, 8, 10), index=1, key="holdout_top_k")
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
                st.warning("Could not read the experiment history; holdout protection is unavailable.")
        if prior_holdout:
            st.info("This dataset already has a successful holdout result in the experiment log.")
        if test_overlap >= test_chunk_size:
            st.error("Overlap must be smaller than chunk size.")
        if st.button(
            "Run selected backends on holdout once",
            type="primary",
            disabled=(not test_stores or not holdout_rows or prior_holdout or test_overlap >= test_chunk_size),
        ):
            test_sample = stratified_sample(holdout_rows, len(holdout_rows), seed=20261118)
            try:
                chunks, vectors, query_encoder = prepare_experiment(test_chunk_size, test_overlap)
                holdout_results = []
                progress = st.progress(0, text="Comparing final holdout results…")
                for index, test_store in enumerate(test_stores):
                    result = benchmark_store(
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
                        query_encoder,
                        KNOWLEDGE_BASE / "runs",
                    )
                    holdout_results.append(result)
                    progress.progress((index + 1) / len(test_stores), text=f"Measured {test_store}")
                st.session_state["holdout_results"] = holdout_results
                progress.empty()
            except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
                st.error(f"The holdout comparison failed: {error}")
        holdout_results = st.session_state.get("holdout_results", [])
        if holdout_results:
            st.dataframe(
                _benchmark_table(holdout_results),
                alt="Final holdout comparison across the selected vector backends",
                hide_index=True,
                width="stretch",
            )
            if any(result["error"] for result in holdout_results):
                st.warning("At least one backend failed. The holdout can be used only once, so resolve optional dependencies before running it.")
            else:
                st.success(f"Compared {len(holdout_results)} backends on the same {len(holdout_rows)} held-out questions.")

with about_tab:
    st.subheader(":material/route: Project map")
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

        **NumPy exact cosine** is a local vector-index baseline. **FAISS** is a similarity-search library. **ChromaDB** is a vector database.
        They are grouped as alternative vector-store/index backends for controlled comparison; none of these names is a retrieval method.
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
