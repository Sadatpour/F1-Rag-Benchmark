from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from f1_rag.chunking import chunk_documents
from f1_rag.embeddings import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_PATH, EMBEDDING_WEIGHTS_PATH
from f1_rag.ingest import KNOWLEDGE_BASE, load_documents, resolve_corpus_path
from f1_rag.stores import create_store


INDEX_ROOT = KNOWLEDGE_BASE / "indices"


def _fingerprint(corpus_path: Path, chunk_size: int, overlap: int) -> str:
    stat = corpus_path.stat()
    weights_stat = EMBEDDING_WEIGHTS_PATH.stat() if EMBEDDING_WEIGHTS_PATH.is_file() else None
    weights_signature = (
        f"{weights_stat.st_size}:{weights_stat.st_mtime_ns}" if weights_stat else "not-present"
    )
    payload = (
        f"{corpus_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{chunk_size}:{overlap}:{EMBEDDING_MODEL_ID}:{EMBEDDING_MODEL_PATH.resolve()}:"
        f"{EMBEDDING_WEIGHTS_PATH.resolve()}:{weights_signature}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_path(dataset: str, store_name: str, fingerprint: str) -> Path:
    safe_store = store_name.lower().replace(" ", "-").replace("(", "").replace(")", "")
    return INDEX_ROOT / dataset / safe_store / fingerprint[:16]


def prepare_index(
    dataset: str,
    store_name: str,
    chunk_size: int,
    overlap: int,
    embedder: Any,
    knowledge_base: Path = KNOWLEDGE_BASE,
    precomputed: tuple[list[dict[str, Any]], np.ndarray] | None = None,
    precompute: Callable[[], tuple[list[dict[str, Any]], np.ndarray]] | None = None,
) -> tuple[Any, int, bool]:
    corpus_path = resolve_corpus_path(dataset, knowledge_base)
    fingerprint = _fingerprint(corpus_path, chunk_size, overlap)
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
        documents = load_documents(dataset, knowledge_base)
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        vectors = embedder.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
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
