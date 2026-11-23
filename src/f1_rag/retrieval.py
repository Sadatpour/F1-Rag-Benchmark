from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from f1_rag.chunking import CHUNKING_METHODS, FIXED_CHUNKING, chunk_documents
from f1_rag.embeddings import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_WEIGHTS_PATH,
    embedding_tokenizer,
)
from f1_rag.ingest import KNOWLEDGE_BASE, iter_documents, resolve_corpus_path
from f1_rag.stores import create_store


INDEX_ROOT = KNOWLEDGE_BASE / "indices"


def _fingerprint(
    corpus_path: Path,
    chunk_size: int,
    overlap: int,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    embedding_model_path: Path = EMBEDDING_MODEL_PATH,
    embedding_weights_path: Path = EMBEDDING_WEIGHTS_PATH,
    chunking_method: str = FIXED_CHUNKING,
) -> str:
    if chunking_method not in CHUNKING_METHODS:
        raise ValueError(f"Unknown chunking method {chunking_method!r}.")
    stat = corpus_path.stat()
    weights_stat = embedding_weights_path.stat() if embedding_weights_path.is_file() else None
    weights_signature = (
        f"{weights_stat.st_size}:{weights_stat.st_mtime_ns}" if weights_stat else "not-present"
    )
    payload = (
        f"chunking-v3:{chunking_method}:{corpus_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{chunk_size}:{overlap}:{embedding_model_id}:{embedding_model_path.resolve()}:"
        f"{embedding_weights_path.resolve()}:{weights_signature}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_path(dataset: str, store_name: str, fingerprint: str) -> Path:
    safe_store = store_name.lower().replace(" ", "-").replace("(", "").replace(")", "")
    return INDEX_ROOT / dataset / safe_store / fingerprint[:16]


def embed_documents(
    documents: Iterable[dict[str, Any]],
    embedder: Any,
    chunk_size: int,
    overlap: int,
    batch_size: int = 128,
    on_progress: Callable[[int], None] | None = None,
    chunking_method: str = FIXED_CHUNKING,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")
    tokenizer, max_tokens = embedding_tokenizer(embedder)
    chunks: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    vector_batches: list[np.ndarray] = []
    document_count = 0

    def encode_batch(batch: list[dict[str, Any]]) -> None:
        vector_batches.append(
            embedder.encode(
                [chunk["text"] for chunk in batch],
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        )

    for document_count, document in enumerate(documents, start=1):
        document_chunks = chunk_documents(
            [document],
            chunk_size=chunk_size,
            overlap=overlap,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            method=chunking_method,
            semantic_encoder=embedder.encode if chunking_method == "Semantic chunking" else None,
        )
        chunks.extend(document_chunks)
        pending.extend(document_chunks)
        while len(pending) >= batch_size:
            encode_batch(pending[:batch_size])
            del pending[:batch_size]
        if on_progress and document_count % 250 == 0:
            on_progress(document_count)
    if pending:
        encode_batch(pending)
    if on_progress and document_count % 250:
        on_progress(document_count)

    vectors = np.concatenate(vector_batches) if vector_batches else np.empty((0, 0), dtype=np.float32)
    return chunks, vectors


def prepare_index(
    dataset: str,
    store_name: str,
    chunk_size: int,
    overlap: int,
    embedder: Any,
    knowledge_base: Path = KNOWLEDGE_BASE,
    precomputed: tuple[list[dict[str, Any]], np.ndarray] | None = None,
    precompute: Callable[[], tuple[list[dict[str, Any]], np.ndarray]] | None = None,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    embedding_model_path: Path = EMBEDDING_MODEL_PATH,
    embedding_weights_path: Path = EMBEDDING_WEIGHTS_PATH,
    chunking_method: str = FIXED_CHUNKING,
) -> tuple[Any, int, bool]:
    corpus_path = resolve_corpus_path(dataset, knowledge_base)
    fingerprint = _fingerprint(
        corpus_path, chunk_size, overlap,
        embedding_model_id, embedding_model_path, embedding_weights_path, chunking_method,
    )
    store = create_store(store_name, _index_path(dataset, store_name, fingerprint))
    if store.load(fingerprint):
        if hasattr(store, "documents"):
            chunk_count = len(store.documents)
        else:
            chunk_count = int(store.collection.count())
        return store, chunk_count, True

    if precomputed is None and precompute is not None:
        chunks, vectors = precompute()
    elif precomputed is None:
        if embedder is None:
            raise ValueError("An embedding model or precomputed vectors are required to build the index.")
        chunks, vectors = embed_documents(
            iter_documents(dataset, knowledge_base),
            embedder,
            chunk_size,
            overlap,
            chunking_method=chunking_method,
        )
    else:
        chunks, vectors = precomputed
    if not chunks:
        raise ValueError(f"The {dataset} corpus did not produce any chunks.")
    store.build(np.asarray(vectors, dtype=np.float32), chunks, fingerprint)
    return store, len(chunks), False


def encode_query(embedder: Any, query: str) -> np.ndarray:
    vector = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return np.asarray(vector, dtype=np.float32)


def load_evaluation_rows(knowledge_base: Path = KNOWLEDGE_BASE) -> list[dict[str, Any]]:
    path = knowledge_base / "evaluation" / "qa.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation set: {path}")
    rows = []
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid QA JSON at line {line_number}: {error}") from error
            if isinstance(row, dict):
                rows.append(row)
    return rows
