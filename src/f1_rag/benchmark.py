from __future__ import annotations

import json
import os
import platform
import statistics
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from f1_rag.evaluation import ndcg_at_n, ndcg_ids_at_n, recall_at_n, recall_ids_at_n
from f1_rag.stores import create_store


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str
    store_name: str
    embedding_model: str
    chunk_size: int
    overlap: int
    top_k: int
    repeats: int = 5
    evaluation_split: str = "tuning"
    metric_ks: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.chunk_size < 1 or self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")
        if self.top_k < 1 or self.repeats < 1:
            raise ValueError("top_k and repeats must be greater than zero.")
        if any(k < 1 for k in self.metric_ks):
            raise ValueError("All metric cutoffs must be greater than zero.")
        if self.evaluation_split not in {"tuning", "holdout"}:
            raise ValueError("evaluation_split must be 'tuning' or 'holdout'.")


def percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _process_resources() -> tuple[int | None, float | None]:
    try:
        import psutil
    except ImportError:
        return None, None
    process = psutil.Process(os.getpid())
    cpu = process.cpu_times()
    return process.memory_info().rss, cpu.user + cpu.system


def _optional_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _total_ram_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().total)


def _resource_delta(before: tuple[int | None, float | None], after: tuple[int | None, float | None]) -> tuple[int | None, float | None]:
    memory = after[0] - before[0] if before[0] is not None and after[0] is not None else None
    cpu = after[1] - before[1] if before[1] is not None and after[1] is not None else None
    return memory, cpu


def benchmark_store(
    config: BenchmarkConfig,
    documents: list[dict[str, Any]],
    vectors: np.ndarray,
    queries: list[dict[str, Any]],
    encode_query: Callable[[str], np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    """Compare one vector store using fixed documents, vectors, and labeled questions."""
    config.validate()
    if len(documents) != len(vectors):
        raise ValueError("The number of document chunks must match the number of vectors.")
    if not documents or not queries:
        raise ValueError("Benchmarking requires at least one document and one labeled query.")
    # Encode queries once so every store receives identical vectors and the
    # search CPU metric measures the store rather than embedding inference.
    query_vectors = [encode_query(str(row["question"])) for row in queries]
    id_labeled = [bool(row.get("relevant_ids") or row.get("relevance_by_id")) for row in queries]
    if any(id_labeled) and not all(id_labeled):
        raise ValueError("Use a consistent relevance-label level across the benchmark question set.")
    relevance_level = "document_or_chunk_id" if all(id_labeled) else "source"

    run_id = uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    build_seconds: list[float] = []
    search_latencies: list[float] = []
    cutoffs = tuple(sorted(set(config.metric_ks or (config.top_k,))))
    search_limit = max(config.top_k, *cutoffs)
    recalls: dict[int, list[float]] = {k: [] for k in cutoffs}
    ndcgs: dict[int, list[float]] = {k: [] for k in cutoffs}
    memory_deltas: list[int] = []
    cpu_deltas: list[float] = []
    search_cpu_deltas: list[float] = []
    failure: str | None = None

    for repeat in range(config.repeats):
        with tempfile.TemporaryDirectory(prefix=f"{run_id}-{repeat}-", dir=output_dir) as temporary:
            store_path = Path(temporary) / config.store_name.lower().replace(" ", "-")
            store = create_store(config.store_name, store_path)
            try:
                before = _process_resources()
                started = time.perf_counter()
                store.build(vectors, documents, f"{run_id}:{repeat}")
                build_seconds.append(time.perf_counter() - started)
                after = _process_resources()
                memory_delta, cpu_delta = _resource_delta(before, after)
                if memory_delta is not None:
                    memory_deltas.append(memory_delta)
                if cpu_delta is not None:
                    cpu_deltas.append(cpu_delta)

                search_before = _process_resources()
                query_hits: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
                for row, query_vector in zip(queries, query_vectors, strict=True):
                    started = time.perf_counter()
                    hits = store.search(query_vector, search_limit)
                    search_latencies.append((time.perf_counter() - started) * 1000)
                    query_hits.append((row, hits))
                search_after = _process_resources()
                _, search_cpu_delta = _resource_delta(search_before, search_after)
                if search_cpu_delta is not None:
                    search_cpu_deltas.append(search_cpu_delta)

                for row, hits in query_hits:
                    if relevance_level == "document_or_chunk_id":
                        relevance = {
                            str(document_id): int(grade)
                            for document_id, grade in row.get("relevance_by_id", {}).items()
                            if int(grade) > 0
                        }
                        relevance.update({
                            str(document_id): max(1, relevance.get(str(document_id), 0))
                            for document_id in row.get("relevant_ids", [])
                        })
                        for cutoff in cutoffs:
                            recalls[cutoff].append(recall_ids_at_n(hits, set(relevance), cutoff))
                            ndcgs[cutoff].append(ndcg_ids_at_n(hits, relevance, cutoff))
                    else:
                        relevant = {str(value) for value in row.get("relevant_sources", [])}
                        if not relevant and row.get("source"):
                            relevant = {str(row["source"])}
                        for cutoff in cutoffs:
                            recalls[cutoff].append(recall_at_n(hits, relevant, cutoff))
                            ndcgs[cutoff].append(ndcg_at_n(hits, relevant, cutoff))
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                break
            finally:
                close = getattr(store, "close", None)
                if callable(close):
                    close()

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "vectors_shape": list(vectors.shape),
            "cpu_model": platform.processor() or None,
            "total_ram_bytes": _total_ram_bytes(),
            "store_package_version": _optional_package_version(
                {"FAISS": "faiss-cpu", "ChromaDB": "chromadb"}.get(config.store_name, "numpy")
            ),
        },
        "results": {
            "documents": len(documents),
            "queries": len(queries),
            "relevance_level": relevance_level,
            "successful_repeats": len(build_seconds),
            "index_build_seconds_median": statistics.median(build_seconds) if build_seconds else None,
            "search_latency_ms_p50": percentile(search_latencies, 0.50) if search_latencies else None,
            "search_latency_ms_p95": percentile(search_latencies, 0.95) if search_latencies else None,
            "search_latency_ms_mean": statistics.fmean(search_latencies) if search_latencies else None,
            "metrics_at_k": {
                str(k): {
                    "recall": statistics.fmean(recalls[k]) if recalls[k] else None,
                    "ndcg": statistics.fmean(ndcgs[k]) if ndcgs[k] else None,
                }
                for k in cutoffs
            },
            "recall_at_k": statistics.fmean(recalls[config.top_k])
            if config.top_k in recalls and recalls[config.top_k]
            else None,
            "ndcg_at_k": statistics.fmean(ndcgs[config.top_k])
            if config.top_k in ndcgs and ndcgs[config.top_k]
            else None,
            "rss_delta_bytes_median": statistics.median(memory_deltas) if memory_deltas else None,
            "index_build_cpu_seconds_median": statistics.median(cpu_deltas) if cpu_deltas else None,
            "search_cpu_seconds_median": statistics.median(search_cpu_deltas) if search_cpu_deltas else None,
            "resource_metrics_available": bool(memory_deltas or cpu_deltas),
        },
        "error": failure,
    }
    with (output_dir / "runs.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def run_hyperparameter_grid(
    *,
    dataset: str,
    store_names: Iterable[str],
    embedding_model: str,
    chunk_sizes: Iterable[int],
    overlaps: Iterable[int],
    top_ks: Iterable[int],
    metric_ks: Iterable[int] | None = None,
    repeats: int,
    prepare: Callable[[int, int], tuple[list[dict[str, Any]], np.ndarray, Callable[[str], np.ndarray]]],
    queries: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run a tuning grid; callers must supply only the tuning split, never the held-out test split."""
    results = []
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= chunk_size:
                continue
            documents, vectors, encode_query = prepare(chunk_size, overlap)
            for top_k in top_ks:
                for store_name in store_names:
                    config = BenchmarkConfig(
                        dataset=dataset,
                        store_name=store_name,
                        embedding_model=embedding_model,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        top_k=top_k,
                        repeats=repeats,
                        evaluation_split="tuning",
                        metric_ks=tuple(metric_ks or (top_k,)),
                    )
                    results.append(
                        benchmark_store(config, documents, vectors, queries, encode_query, output_dir)
                    )
    return results
