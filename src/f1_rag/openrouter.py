from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from f1_rag.model_config import load_model_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MODEL_CONFIG = load_model_config()
OPENROUTER_CONFIG = MODEL_CONFIG["openrouter"]
DEFAULT_MODELS = tuple(OPENROUTER_CONFIG["answer_models"])
DEFAULT_JUDGE_MODEL = OPENROUTER_CONFIG["judge_model"]
DEFAULT_REWRITE_MODEL = OPENROUTER_CONFIG["query_rewrite_model"]
ADDITIONAL_MODELS = tuple(OPENROUTER_CONFIG.get("additional_models", []))
CONTEXT_WINDOW_OVERRIDES = MODEL_CONFIG.get("context_windows", {})


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    secrets_file = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_file.is_file():
        return ""
    try:
        settings = tomllib.loads(secrets_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Could not read {secrets_file}: {error}") from error
    return str(settings.get("openrouter", {}).get("api_key", "")).strip()


def _request(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    timeout: int = 90,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to OpenRouter: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("OpenRouter returned an invalid JSON response.") from error


def list_models() -> dict[str, dict[str, Any]]:
    response = _request(f"{OPENROUTER_URL}/models", {"Accept": "application/json"}, timeout=20)
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        raise RuntimeError("OpenRouter returned an unexpected model catalog.")
    models: dict[str, dict[str, Any]] = {}
    for model in response.get("data", []):
        if not isinstance(model, dict) or not model.get("id"):
            continue
        pricing = model.get("pricing", {})
        if not isinstance(pricing, dict):
            pricing = {}
        is_free = str(model["id"]).endswith(":free") or (
            pricing.get("prompt") == "0" and pricing.get("completion") == "0"
        )
        if is_free:
            models[str(model["id"])] = {
                "context_length": model.get("context_length"),
                "name": model.get("name", model["id"]),
            }
    return models


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    max_tokens: int = 1800,
    temperature: float = 0.2,
    response_format: dict[str, str] | None = None,
    reasoning: dict[str, str] | None = None,
) -> str:
    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY or add it to .streamlit/secrets.toml."
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if reasoning is not None:
        payload["reasoning"] = reasoning
    response = _request(
        f"{OPENROUTER_URL}/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "F1 RAG Benchmark",
        },
        payload,
    )
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"OpenRouter response did not contain a completion: {response}") from error
    if not isinstance(content, str) or not content.strip():
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            usage = response.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            details = usage.get("completion_tokens_details", {})
            reasoning_tokens = details.get("reasoning_tokens", 0)
            raise RuntimeError(
                f"OpenRouter stopped {model} at its {max_tokens}-token output limit without "
                f"returning an answer ({reasoning_tokens} of {completion_tokens} completion "
                "tokens were used for reasoning)."
            )
        raise RuntimeError(
            f"OpenRouter returned an empty completion for {model} "
            f"(finish reason: {finish_reason or 'unknown'})."
        )
    return content.strip()


def rewrite_query(question: str, model: str, corpus_preview: str, api_key: str) -> str:
    rewritten = chat_completion(
        model,
        [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question as one concise, standalone search query for a Formula 1 "
                    "retrieval system. Preserve names, dates, and intent. Use the collection overview only "
                    "to clarify terminology. Return only an English query, with no explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Collection overview:\n{corpus_preview[:3000]}\n\nQuestion:\n{question}",
            },
        ],
        api_key,
        max_tokens=1800,
    )
    return rewritten.strip().strip('"')


def judge_answer(
    question: str,
    expected_answer: str,
    generated_answer: str,
    model: str,
    api_key: str,
) -> dict[str, str | bool]:
    output = chat_completion(
        model,
        [
            {
                "role": "system",
                "content": (
                    "Judge factual QA. Decide whether the candidate gives the essential factual answer "
                    "supported by the reference answer or retrieved evidence. Ignore harmless wording "
                    "differences. Write the reason in English. Return only JSON "
                    'with keys "correct" (boolean) and "reason" (short string).'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nReference answer or retrieved evidence: {expected_answer}\n"
                    f"Candidate answer: {generated_answer}"
                ),
            },
        ],
        api_key,
        max_tokens=500,
        temperature=0,
        response_format={"type": "json_object"} if model == DEFAULT_JUDGE_MODEL else None,
        reasoning={"effort": "none"} if model == DEFAULT_JUDGE_MODEL else None,
    )
    match = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if not match:
        raise ValueError(f"The judge did not return JSON: {output}")
    try:
        result = json.loads(match.group())
    except json.JSONDecodeError as error:
        raise ValueError(f"The judge returned invalid JSON: {output}") from error
    if not isinstance(result.get("correct"), bool) or not isinstance(result.get("reason"), str):
        raise ValueError(f"The judge JSON must contain a boolean `correct` and string `reason`: {output}")
    return {"correct": result["correct"], "reason": result["reason"]}


def answer_question(
    question: str,
    contexts: list[dict[str, Any]],
    model: str,
    api_key: str,
    context_length: int | None = None,
) -> str:
    budget = min(32000, int(context_length * 0.65)) if context_length else 12000
    context_parts: list[str] = []
    used_chars = 0
    for index, context in enumerate(contexts, start=1):
        part = (
            f"[{index}] Source: {context.get('source', 'unknown')}\n"
            f"Source URL: {context.get('source_url', '')}\n"
            f"{context.get('text', '')}\n"
        )
        char_limit = budget * 4
        if used_chars + len(part) > char_limit:
            remaining = char_limit - used_chars
            if remaining > 200:
                context_parts.append(part[:remaining])
            break
        context_parts.append(part)
        used_chars += len(part)
    if not context_parts:
        raise ValueError("No retrieved context was available for answering.")
    return chat_completion(
        model,
        [
            {
                "role": "system",
                "content": (
                    "You are a Formula 1 research assistant. Answer in English and only from the supplied context. "
                    "If it does not support a reliable answer, say so. Cite factual claims with the "
                    "provided [n] references. Preserve the supplied source names and URLs; never invent links."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{''.join(context_parts)}\nQuestion: {question}",
            },
        ],
        api_key,
    )
