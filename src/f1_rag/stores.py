from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyStore:
    """Persistent exact cosine-similarity search backed by NumPy."""

    def __init__(self, path: Path):
        self.path = path
        self.vectors: np.ndarray | None = None
        self.documents: list[dict[str, Any]] = []
        self.fingerprint = ""

    def load(self, fingerprint: str) -> bool:
        manifest_path = self.path / "manifest.json"
        vectors_path = self.path / "vectors.npy"
        documents_path = self.path / "documents.jsonl"
        if not all(path.is_file() for path in (manifest_path, vectors_path, documents_path)):
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            return False
        self.vectors = np.load(vectors_path, allow_pickle=False)
        self.documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines()]
        self.fingerprint = fingerprint
        return len(self.documents) == len(self.vectors)

    def build(self, vectors: np.ndarray, documents: list[dict[str, Any]], fingerprint: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.documents = documents
        self.fingerprint = fingerprint
        np.save(self.path / "vectors.npy", self.vectors, allow_pickle=False)
        with (self.path / "documents.jsonl").open("w", encoding="utf-8") as file:
            for document in documents:
                file.write(json.dumps(document, ensure_ascii=False) + "\n")
        (self.path / "manifest.json").write_text(
            json.dumps({"fingerprint": fingerprint, "count": len(documents)}, indent=2),
            encoding="utf-8",
        )

    def search(self, query_vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        if self.vectors is None:
            raise RuntimeError("The NumPy vector index has not been loaded.")
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32)
        indexes = np.argsort(scores)[::-1][: min(limit, len(scores))]
        return [{**self.documents[int(index)], "score": float(scores[index])} for index in indexes]


class FaissStore:
    def __init__(self, path: Path):
        self.path = path
        self.index: Any = None
        self.documents: list[dict[str, Any]] = []

    def load(self, fingerprint: str) -> bool:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "FAISS is not installed. Install it with `python -m pip install -r requirements-vectorstores.txt`."
            ) from error
        manifest_path = self.path / "manifest.json"
        index_path = self.path / "index.faiss"
        documents_path = self.path / "documents.jsonl"
        if not all(path.is_file() for path in (manifest_path, index_path, documents_path)):
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            return False
        self.index = faiss.read_index(str(index_path))
        self.documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines()]
        return self.index.ntotal == len(self.documents)

    def build(self, vectors: np.ndarray, documents: list[dict[str, Any]], fingerprint: str) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "FAISS is not installed. Install it with `python -m pip install -r requirements-vectorstores.txt`."
            ) from error
        self.path.mkdir(parents=True, exist_ok=True)
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self.documents = documents
        faiss.write_index(self.index, str(self.path / "index.faiss"))
        _write_documents(self.path, documents)
        _write_manifest(self.path, fingerprint, len(documents))

    def search(self, query_vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("The FAISS vector index has not been loaded.")
        scores, indexes = self.index.search(
            np.ascontiguousarray([query_vector], dtype=np.float32),
            min(limit, len(self.documents)),
        )
        return [
            {**self.documents[int(index)], "score": float(score)}
            for score, index in zip(scores[0], indexes[0])
            if index >= 0
        ]


class ChromaStore:
    def __init__(self, path: Path):
        self.path = path
        self.client: Any = None
        self.collection: Any = None

    def _connect(self) -> None:
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "ChromaDB is not installed. Install it with `python -m pip install -r requirements-vectorstores.txt`."
            ) from error
        self.client = chromadb.PersistentClient(path=str(self.path))

    def close(self) -> None:
        if self.client is not None:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
            self.client = None
            self.collection = None

    def load(self, fingerprint: str) -> bool:
        self._connect()
        manifest_path = self.path / "f1-rag-manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            return False
        names = _collection_names(self.client)
        if "f1_rag" not in names:
            return False
        self.collection = self.client.get_collection("f1_rag")
        return self.collection.count() == manifest.get("count")

    def build(self, vectors: np.ndarray, documents: list[dict[str, Any]], fingerprint: str) -> None:
        self._connect()
        names = _collection_names(self.client)
        if "f1_rag" in names:
            self.client.delete_collection("f1_rag")
        self.collection = self.client.create_collection(
            "f1_rag",
            metadata={"hnsw:space": "cosine"},
        )
        for start in range(0, len(documents), 1000):
            batch = documents[start : start + 1000]
            vectors_batch = vectors[start : start + 1000]
            self.collection.add(
                ids=[str(document["id"]) for document in batch],
                embeddings=vectors_batch.tolist(),
                documents=[str(document["text"]) for document in batch],
                metadatas=[_chroma_metadata(document) for document in batch],
            )
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "f1-rag-manifest.json").write_text(
            json.dumps({"fingerprint": fingerprint, "count": len(documents)}, indent=2),
            encoding="utf-8",
        )

    def search(self, query_vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        if self.collection is None:
            raise RuntimeError("The ChromaDB vector index has not been loaded.")
        result = self.collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        for document_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append({**metadata, "id": document_id, "text": text, "score": 1.0 - float(distance)})
        return hits


def _write_documents(path: Path, documents: list[dict[str, Any]]) -> None:
    with (path / "documents.jsonl").open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def _write_manifest(path: Path, fingerprint: str, count: int) -> None:
    (path / "manifest.json").write_text(
        json.dumps({"fingerprint": fingerprint, "count": count}, indent=2),
        encoding="utf-8",
    )


def _chroma_metadata(document: dict[str, Any]) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {}
    for key in (
        "source",
        "source_file",
        "source_url",
        "title",
        "season",
        "round",
        "endpoint",
        "chunk_index",
    ):
        value = document.get(key)
        if isinstance(value, (str, int)):
            metadata[key] = value
    return metadata


def _collection_names(client: Any) -> set[str]:
    return {
        collection if isinstance(collection, str) else collection.name
        for collection in client.list_collections()
    }


def create_store(name: str, path: Path) -> NumpyStore | FaissStore | ChromaStore:
    stores = {
        "NumPy (exact)": NumpyStore,
        "FAISS": FaissStore,
        "ChromaDB": ChromaStore,
    }
    try:
        store_type = stores[name]
    except KeyError as error:
        raise ValueError(f"Unknown vector store: {name}") from error
    return store_type(path)
