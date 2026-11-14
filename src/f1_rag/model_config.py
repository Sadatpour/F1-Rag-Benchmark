from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "models.toml"


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Could not read model configuration {path}: {error}") from error

    embedding = config.get("embedding")
    openrouter = config.get("openrouter")
    if not isinstance(embedding, dict) or not isinstance(openrouter, dict):
        raise ValueError(f"{path} must define [embedding] and [openrouter] sections.")
    if not isinstance(embedding.get("repo_id"), str) or not embedding["repo_id"].strip():
        raise ValueError(f"{path} must set a non-empty embedding.repo_id.")
    if not isinstance(embedding.get("local_path"), str) or not embedding["local_path"].strip():
        raise ValueError(f"{path} must set a non-empty embedding.local_path.")
    weights_path = embedding.get("weights_path", "")
    if not isinstance(weights_path, str):
        raise ValueError(f"{path} embedding.weights_path must be a path string.")
    answer_models = openrouter.get("answer_models")
    if not isinstance(answer_models, list) or not answer_models:
        raise ValueError(f"{path} must define at least one openrouter.answer_models entry.")
    if any(not isinstance(model, str) or not model.strip() for model in answer_models):
        raise ValueError(f"{path} openrouter.answer_models must contain non-empty model IDs.")
    for key in ("judge_model", "query_rewrite_model"):
        if not isinstance(openrouter.get(key), str) or not openrouter[key].strip():
            raise ValueError(f"{path} must define openrouter.{key}.")
    additional = openrouter.get("additional_models", [])
    if not isinstance(additional, list) or any(not isinstance(model, str) or not model.strip() for model in additional):
        raise ValueError(f"{path} openrouter.additional_models must be a list of model IDs.")

    windows = config.get("context_windows", {})
    if not isinstance(windows, dict) or any(
        not isinstance(model, str) or not isinstance(size, int) or size <= 0
        for model, size in windows.items()
    ):
        raise ValueError(f"{path} context_windows must map model IDs to positive token counts.")
    return config


def resolve_model_path(local_path: str) -> Path:
    path = Path(local_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
