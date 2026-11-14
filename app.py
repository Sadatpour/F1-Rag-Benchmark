from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
sys.path.insert(0, str(ROOT / "src"))

from f1_rag.embeddings import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_PATH, EMBEDDING_WEIGHTS_PATH, load_embedding_model
from f1_rag.evaluation import ndcg_at_n, qa_accuracy, recall_at_n
from f1_rag.ingest import KNOWLEDGE_BASE, load_documents, resolve_corpus_path
from f1_rag.openrouter import (
    ADDITIONAL_MODELS,
    CONTEXT_WINDOW_OVERRIDES,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODELS,
    DEFAULT_REWRITE_MODEL,
    answer_question,
    get_api_key,
    judge_answer,
    list_models,
    rewrite_query,
)
from f1_rag.retrieval import encode_query, load_evaluation_rows, prepare_index
from f1_rag.sources import JOLPICA_API_BASE, JOLPICA_ATTRIBUTION, jolpica_source_url


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
    docs = load_documents(dataset)
    return "\n".join(
        f"{doc.get('source', '')} ({doc.get('source_url', '')}): "
        f"{doc.get('title', doc.get('text', '')[:300])}"
        for doc in docs[:12]
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
) -> tuple[list[dict[str, Any]], Any]:
    from f1_rag.chunking import chunk_documents

    documents = load_documents(dataset)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    vectors = embedding_model(model_id, model_path, weights_path, weights_size, weights_mtime).encode(
        [chunk["text"] for chunk in chunks],
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
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
    )


def _get_hits(
    dataset: str,
    store_name: str,
    chunk_size: int,
    overlap: int,
    query_vector: Any,
    count: int,
) -> tuple[list[dict[str, Any]], int]:
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
        hits, _ = _get_hits(dataset, store_name, chunk_size, overlap, query, 10)
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


def _render_source_citations(hits: list[dict[str, Any]]) -> None:
    citations: dict[str, str] = {}
    for hit in hits:
        source = str(hit.get("source", ""))
        url = str(hit.get("source_url", ""))
        if source.startswith("Jolpica:") and not url.startswith(JOLPICA_API_BASE + "/"):
            url = jolpica_source_url(source)
        if url.startswith(JOLPICA_API_BASE + "/"):
            citations[source or JOLPICA_ATTRIBUTION] = url
    if citations:
        st.markdown("**Sources**")
        for source, url in citations.items():
            st.markdown(f"- [{JOLPICA_ATTRIBUTION}: {source}]({url})")


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
        ("NumPy (exact)", "FAISS", "ChromaDB"),
        default=["NumPy (exact)"],
        help="FAISS and ChromaDB need the optional vector-store dependencies.",
    )
    selected_models = st.multiselect(
        "Answer models",
        model_options,
        default=[model for model in DEFAULT_MODELS[:2] if model in model_options],
        help="A model is queried once per selected vector store.",
    )
    judge_model = st.selectbox(
        "Judge model",
        model_options,
        index=model_options.index(DEFAULT_JUDGE_MODEL) if DEFAULT_JUDGE_MODEL in model_options else 0,
    )
    rewrite_enabled = st.toggle("Rewrite query before embedding", value=False)
    rewrite_model = st.selectbox(
        "Query-rewrite model",
        model_options,
        index=model_options.index(DEFAULT_REWRITE_MODEL) if DEFAULT_REWRITE_MODEL in model_options else 0,
        disabled=not rewrite_enabled,
    )
    st.space("small")
    chunk_size = st.select_slider("Chunk size (approx. words)", options=(500, 750, 1000), value=500)
    overlap = st.select_slider(
        "Chunk overlap (words)",
        options=(50, 80, 100, 150),
        value=80,
    )
    if overlap >= chunk_size:
        st.error("Overlap must be smaller than the selected chunk size.")
    top_k = st.slider("Retrieved context count", min_value=1, max_value=10, value=5)

    st.space("small")
    st.markdown("**Model context windows**")
    for model in dict.fromkeys((*selected_models, judge_model, *((rewrite_model,) if rewrite_enabled else ()))):
        context = _context_window(model)
        context_label = f"{int(context):,} tokens" if isinstance(context, (int, float)) else "not listed"
        st.caption(f"`{model}` · {context_label}")
    st.caption(f"Embedding: `{EMBEDDING_MODEL_ID}`")
    st.caption(f"Local weights: `{EMBEDDING_WEIGHTS_PATH}`")
    if api_key_error:
        st.error(api_key_error)
    elif not api_key:
        st.warning("Add an OpenRouter key in `.streamlit/secrets.toml` or set `OPENROUTER_API_KEY`.")
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

query_tab, eval_tab, about_tab = st.tabs(
    (":material/search: Compare answers", ":material/query_stats: Evaluation", ":material/folder: Project guide")
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
        elif not api_key:
            st.error("Configure an OpenRouter API key before asking models to answer.")
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
                        _render_source_citations(hits)
                        verdict = result["judge"]
                        if verdict:
                            st.badge(
                                "Correct" if verdict["correct"] else "Needs review",
                                icon=":material/check_circle:" if verdict["correct"] else ":material/warning:",
                                color="green" if verdict["correct"] else "orange",
                            )
                            st.caption(f"Judge note: {verdict['reason']}")
                    elif key not in payload["errors"]:
                        st.info("No answer was generated for this store/model combination.")
                    hits = payload["hits"].get(store_name, [])
                    if hits:
                        with st.expander("Retrieved evidence"):
                            for rank, hit in enumerate(hits, start=1):
                                st.markdown(
                                    f"**[{rank}] {hit.get('source', 'Unknown source')}** · "
                                    f"score `{hit.get('score', 0):.3f}`"
                                )
                                source_url = hit.get("source_url")
                                if source_url and source_url.startswith(JOLPICA_API_BASE + "/"):
                                    st.markdown(f"[View {JOLPICA_ATTRIBUTION} record]({source_url})")
                                st.write(hit.get("text", ""))
                                if rank < len(hits):
                                    st.divider()

with eval_tab:
    st.subheader("Retrieval & answer evaluation")
    st.caption(
        "Recall@10 and NDCG@10 use QA reference-source matching. Answer accuracy is "
        "estimated by an OpenRouter judge on a configurable sample."
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
    eval_button = st.button("Run Recall@10 / NDCG@10", type="primary", disabled=not store_names)
    if eval_button:
        if not eval_pool:
            st.error(f"No QA rows map to the {dataset_title} corpus.")
        else:
            sample = random.Random(42).sample(eval_pool, min(metric_count, len(eval_pool)))
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
                st.metric("Recall@10", f"{result['recall@10']:.1%}")
                st.metric("NDCG@10", f"{result['ndcg@10']:.3f}")
                st.caption(f"{result['questions']} questions")

    st.space("medium")
    st.markdown("#### QA accuracy (LLM judge)")
    accuracy_count = st.slider("QA accuracy sample", min_value=1, max_value=10, value=3)
    accuracy_model = st.selectbox("Model to evaluate", selected_models or model_options, key="accuracy_model")
    accuracy_store = st.selectbox("Vector store to evaluate", store_names or ("NumPy (exact)",), key="accuracy_store")
    accuracy_button = st.button(
        "Evaluate answers with judge",
        disabled=not api_key or not store_names or not selected_models,
    )
    if accuracy_button:
        if not eval_pool:
            st.error(f"No QA rows map to the {dataset_title} corpus.")
        else:
            sample = random.Random(7).sample(eval_pool, min(accuracy_count, len(eval_pool)))
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
    if verdicts:
        st.metric("QA accuracy", f"{qa_accuracy(verdicts):.1%}", f"{len(verdicts)} judged questions")
        with st.expander("QA judge details"):
            for verdict in verdicts:
                st.markdown(
                    f"{'✅' if verdict['correct'] else '❌'} **{verdict['question']}**  \n"
                    f"{verdict['reason']}"
                )

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
        - `.streamlit/secrets.toml` — Private OpenRouter key; never commit this file
        - `src/f1_rag/` — Ingestion, chunking, embeddings, vector stores, retrieval, generation, and evaluation

        The third vector-store option is **NumPy exact cosine**: Windows-friendly and service-free.
        FAISS and ChromaDB are also available for comparison.
        """
    )
    st.info(
        "Chunk sizes are approximate word counts. Model context windows come from the OpenRouter "
        "catalog; unavailable values are shown as “not listed.”"
    )
    st.markdown(
        f"Jolpica data attribution: [{JOLPICA_ATTRIBUTION}]({JOLPICA_API_BASE}/). "
        "Retrieved Jolpica evidence includes direct API record links."
    )
