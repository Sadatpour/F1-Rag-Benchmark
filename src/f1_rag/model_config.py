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
    embedding_models = embedding.get("models")
    if embedding_models is not None:
        if not isinstance(embedding_models, dict) or not embedding_models:
            raise ValueError(f"{path} embedding.models must be a non-empty table.")
        for name, profile in embedding_models.items():
            if not isinstance(profile, dict):
                raise ValueError(f"{path} embedding.models.{name} must be a table.")
            for field in ("repo_id", "local_path"):
                if not isinstance(profile.get(field), str) or not profile[field].strip():
                    raise ValueError(f"{path} embedding.models.{name}.{field} is required.")
            profile_weights = profile.get("weights_path", "")
            if not isinstance(profile_weights, str):
                raise ValueError(f"{path} embedding.models.{name}.weights_path must be a path string.")
        default_embedding = embedding.get("default_model")
        if default_embedding not in embedding_models:
            raise ValueError(f"{path} embedding.default_model must match an embedding.models entry.")
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

    reranking = config.get("reranking", {})
    if not isinstance(reranking, dict):
        raise ValueError(f"{path} reranking must be a TOML table.")
    reranker_models = reranking.get("models", {})
    if not isinstance(reranker_models, dict):
        raise ValueError(f"{path} reranking.models must be a TOML table.")
    if reranker_models:
        for name, profile in reranker_models.items():
            if not isinstance(profile, dict):
                raise ValueError(f"{path} reranking.models.{name} must be a table.")
            for field in ("repo_id", "local_path"):
                if not isinstance(profile.get(field), str) or not profile[field].strip():
                    raise ValueError(f"{path} reranking.models.{name}.{field} is required.")
            max_length = profile.get("max_length", 512)
            if not isinstance(max_length, int) or max_length < 1:
                raise ValueError(f"{path} reranking.models.{name}.max_length must be positive.")
        if reranking.get("default_model") not in reranker_models:
            raise ValueError(f"{path} reranking.default_model must match a reranking.models entry.")
    candidate_count = reranking.get("candidate_count", 30)
    if not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError(f"{path} reranking.candidate_count must be positive.")

    windows = config.get("context_windows", {})
    if not isinstance(windows, dict) or any(
        not isinstance(model, str) or not isinstance(size, int) or size <= 0
        for model, size in windows.items()
    ):
        raise ValueError(f"{path} context_windows must map model IDs to positive token counts.")

    providers = config.get("providers", {})
    models = config.get("models", {})
    routing = config.get("routing", {})
    if not isinstance(providers, dict) or not isinstance(models, dict) or not isinstance(routing, dict):
        raise ValueError(f"{path} providers, models, and routing must be TOML tables.")
    for provider, settings in providers.items():
        if not isinstance(settings, dict):
            raise ValueError(f"{path} providers.{provider} must be a TOML table.")
        if settings.get("protocol") not in {"openai_chat", "google_generate_content"}:
            raise ValueError(f"{path} providers.{provider}.protocol is unsupported.")
        if not str(settings.get("base_url", "")).startswith("https://"):
            raise ValueError(f"{path} providers.{provider}.base_url must use HTTPS.")
        if not isinstance(settings.get("api_key_env"), str) or not settings["api_key_env"].strip():
            raise ValueError(f"{path} providers.{provider}.api_key_env must name an environment variable.")
    for name, settings in models.items():
        if not isinstance(settings, dict) or settings.get("provider") not in providers:
            raise ValueError(f"{path} models.{name} must reference a configured provider.")
        if not isinstance(settings.get("model_id"), str) or not settings["model_id"].strip():
            raise ValueError(f"{path} models.{name}.model_id must be a non-empty string.")
        capabilities = settings.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities or any(not isinstance(value, str) for value in capabilities):
            raise ValueError(f"{path} models.{name}.capabilities must be a non-empty string array.")
    rotation = routing.get("answer_rotation", [])
    if not isinstance(rotation, list) or any(name not in models for name in rotation):
        raise ValueError(f"{path} routing.answer_rotation must contain configured model names.")
    return config


def resolve_model_path(local_path: str) -> Path:
    path = Path(local_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
