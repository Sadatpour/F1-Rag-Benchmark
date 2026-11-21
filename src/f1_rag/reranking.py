from __future__ import annotations

from pathlib import Path
from typing import Any

from f1_rag.model_config import load_model_config, resolve_model_path


_CONFIG = load_model_config().get("reranking", {})
_MODEL_ROWS = _CONFIG.get("models", {})
RERANKER_PROFILES: dict[str, dict[str, Any]] = {
    str(name): {
        "repo_id": str(row["repo_id"]),
        "local_path": resolve_model_path(str(row["local_path"])),
        "max_length": int(row.get("max_length", 512)),
    }
    for name, row in _MODEL_ROWS.items()
}
DEFAULT_RERANKER_MODEL = str(_CONFIG.get("default_model", ""))
RERANKER_CANDIDATE_COUNT = int(_CONFIG.get("candidate_count", 30))
if RERANKER_PROFILES and DEFAULT_RERANKER_MODEL not in RERANKER_PROFILES:
    raise ValueError(f"Default reranker {DEFAULT_RERANKER_MODEL!r} is not registered.")


def reranker_profile(name: str | None = None) -> dict[str, Any]:
    selected = name or DEFAULT_RERANKER_MODEL
    try:
        return RERANKER_PROFILES[selected]
    except KeyError as error:
        raise ValueError(f"Reranker {selected!r} is not configured.") from error


def load_reranker(name: str | None = None) -> Any:
    profile = reranker_profile(name)
    model_path: Path = profile["local_path"]
    required_files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing = [filename for filename in required_files if not (model_path / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Reranker files are missing from {model_path}: {', '.join(missing)}."
        )
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "The reranker needs sentence-transformers. Install the project embedding dependencies."
        ) from error
    return CrossEncoder(str(model_path), max_length=profile["max_length"])


def rerank_candidates(
    model: Any,
    query: str,
    candidates: list[dict[str, Any]],
    final_k: int,
) -> list[dict[str, Any]]:
    if final_k < 1:
        raise ValueError("final_k must be greater than zero.")
    if not candidates:
        return []

    pairs = [(query, str(candidate.get("text", ""))) for candidate in candidates]
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    if len(scores) != len(candidates):
        raise ValueError("The reranker returned a different number of scores than candidates.")

    scored = [
        {
            **candidate,
            "retrieval_rank": rank,
            "rerank_score": float(score),
        }
        for rank, (candidate, score) in enumerate(zip(candidates, scores, strict=True), start=1)
    ]
    scored.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
    return scored[:final_k]
