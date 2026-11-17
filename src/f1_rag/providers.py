from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    protocol: str
    base_url: str
    api_key_env: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model_id: str
    capabilities: tuple[str, ...]
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None


@dataclass(frozen=True)
class CompletionResult:
    provider: str
    model_id: str
    text: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    finish_reason: str | None


class ProviderRequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def configured_models(config: dict[str, Any]) -> dict[str, ModelConfig]:
    provider_rows = config.get("providers", {})
    model_rows = config.get("models", {})
    if not isinstance(provider_rows, dict) or not isinstance(model_rows, dict):
        raise ValueError("providers and models must be TOML tables.")
    providers: dict[str, ProviderConfig] = {}
    for name, row in provider_rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"providers.{name} must be a TOML table.")
        protocol = row.get("protocol")
        base_url = row.get("base_url")
        key_env = row.get("api_key_env")
        if protocol not in {"openai_chat", "google_generate_content"}:
            raise ValueError(f"providers.{name}.protocol is unsupported.")
        if not isinstance(base_url, str) or not base_url.startswith("https://"):
            raise ValueError(f"providers.{name}.base_url must be an HTTPS URL.")
        if not isinstance(key_env, str) or not key_env:
            raise ValueError(f"providers.{name}.api_key_env is required.")
        providers[name] = ProviderConfig(name, protocol, base_url.rstrip("/"), key_env)

    models: dict[str, ModelConfig] = {}
    for name, row in model_rows.items():
        if not isinstance(row, dict) or row.get("provider") not in providers:
            raise ValueError(f"models.{name} must reference a configured provider.")
        model_id = row.get("model_id")
        capabilities = row.get("capabilities")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"models.{name}.model_id must be a non-empty string.")
        if not isinstance(capabilities, list) or not capabilities or any(not isinstance(x, str) for x in capabilities):
            raise ValueError(f"models.{name}.capabilities must be a non-empty string array.")
        costs = []
        for key in ("input_cost_per_million", "output_cost_per_million"):
            value = row.get(key)
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"models.{name}.{key} must be non-negative.")
            costs.append(float(value) if value is not None else None)
        models[name] = ModelConfig(
            name=name,
            provider=str(row["provider"]),
            model_id=model_id.strip(),
            capabilities=tuple(capabilities),
            input_cost_per_million=costs[0],
            output_cost_per_million=costs[1],
        )
    return models


class ProviderRouter:
    """Select a compatible configured model and make normalized, logged completions."""

    def __init__(self, config: dict[str, Any], log_path: Path):
        self.config = config
        self.models = configured_models(config)
        self.providers = {
            name: ProviderConfig(
                name=name,
                protocol=row["protocol"],
                base_url=row["base_url"].rstrip("/"),
                api_key_env=row["api_key_env"],
            )
            for name, row in config["providers"].items()
        }
        self.log_path = log_path
        self.key_resolvers: dict[str, Callable[[], str]] = {}

    def _api_key(self, provider: ProviderConfig) -> str:
        key = os.environ.get(provider.api_key_env, "").strip()
        if not key and provider.name in self.key_resolvers:
            key = str(self.key_resolvers[provider.name]()).strip()
        return key

    def complete(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        *,
        capability: str = "chat",
        max_output_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> CompletionResult:
        model = self.models[model_name]
        if capability not in model.capabilities:
            raise ValueError(f"Model {model_name} does not support capability {capability!r}.")
        provider = self.providers[model.provider]
        api_key = self._api_key(provider)
        if not api_key:
            self._log({"provider": provider.name, "model_id": model.model_id, "status": "error", "error_type": "MissingAPIKey"})
            raise ValueError(f"Set the {provider.api_key_env} environment variable for provider {provider.name}.")

        started = time.perf_counter()
        try:
            payload, headers, url = self._request_parts(
                provider, model, messages, api_key, max_output_tokens, temperature, capability
            )
            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            result = self._normalize(provider, model, body, (time.perf_counter() - started) * 1000)
            self._log({"provider": result.provider, "model_id": result.model_id, "latency_ms": result.latency_ms,
                       "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                       "cost_usd": result.cost_usd, "finish_reason": result.finish_reason, "status": "success"})
            return result
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(error_body)
            except json.JSONDecodeError:
                error_payload = {}
            provider_error = error_payload.get("error", {}) if isinstance(error_payload, dict) else {}
            if not isinstance(provider_error, dict):
                provider_error = {"message": str(provider_error)}
            error_code = str(provider_error.get("code") or provider_error.get("type") or "")
            error_text = f"{error_code} {provider_error.get('message', '')}".lower()
            quota_exhausted = any(
                marker in error_text
                for marker in ("quota", "daily limit", "insufficient_quota", "tokens per day", "requests per day")
            )
            retryable = quota_exhausted or error.code in {408, 429} or error.code >= 500
            self._log({"provider": provider.name, "model_id": model.model_id, "latency_ms": (time.perf_counter() - started) * 1000,
                       "status": "error", "error_type": "HTTPError", "http_status": error.code,
                       "provider_error_code": error_code or None, "quota_exhausted": quota_exhausted})
            reason = "daily quota exhausted" if quota_exhausted else f"HTTP {error.code}"
            raise ProviderRequestError(f"Provider {provider.name}: {reason}.", retryable=retryable) from error
        except URLError as error:
            self._log({"provider": provider.name, "model_id": model.model_id, "latency_ms": (time.perf_counter() - started) * 1000,
                       "status": "error", "error_type": "URLError"})
            raise ProviderRequestError(f"Provider {provider.name} could not be reached: {error.reason}.", retryable=True) from error
        except TimeoutError as error:
            self._log({"provider": provider.name, "model_id": model.model_id, "latency_ms": (time.perf_counter() - started) * 1000,
                       "status": "error", "error_type": "TimeoutError"})
            raise ProviderRequestError(f"Provider {provider.name} timed out.", retryable=True) from error
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            self._log({"provider": provider.name, "model_id": model.model_id, "latency_ms": (time.perf_counter() - started) * 1000,
                       "status": "error", "error_type": type(error).__name__})
            raise ProviderRequestError(f"Provider {provider.name} returned an invalid completion.", retryable=False) from error
        except ValueError as error:
            self._log({"provider": provider.name, "model_id": model.model_id, "latency_ms": (time.perf_counter() - started) * 1000,
                       "status": "error", "error_type": "ValueError"})
            raise ProviderRequestError(f"Provider {provider.name} returned an invalid completion.", retryable=False) from error

    def rotate(
        self,
        model_names: list[str],
        messages: list[dict[str, str]],
        *,
        capability: str = "chat",
        max_output_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> CompletionResult:
        """Try configured models in order; only transient failures trigger fallback."""
        if not model_names:
            raise ValueError("At least one model is required for provider rotation.")
        attempted: list[str] = []
        for model_name in model_names:
            model = self.models.get(model_name)
            if model is None or capability not in model.capabilities:
                continue
            provider = self.providers[model.provider]
            if not self._api_key(provider):
                self._log({"provider": provider.name, "model_id": model.model_id, "status": "skipped", "error_type": "MissingAPIKey"})
                continue
            attempted.append(model_name)
            try:
                return self.complete(
                    model_name,
                    messages,
                    capability=capability,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            except ProviderRequestError as error:
                self._log({
                    "provider": provider.name,
                    "model_id": model.model_id,
                    "status": "fallback",
                    "error_type": type(error.__cause__).__name__ if error.__cause__ else "ProviderRequestError",
                    "reason": str(error),
                })
                if not error.retryable:
                    raise
        raise RuntimeError(f"No configured model completed the request. Tried: {', '.join(attempted) or 'none with a usable key'}.")

    @staticmethod
    def _request_parts(
        provider: ProviderConfig,
        model: ModelConfig,
        messages: list[dict[str, str]],
        api_key: str,
        max_output_tokens: int,
        temperature: float,
        capability: str,
    ) -> tuple[dict[str, Any], dict[str, str], str]:
        if max_output_tokens < 1 or not 0 <= temperature <= 2:
            raise ValueError("max_output_tokens must be positive and temperature must be from 0 to 2.")
        if provider.protocol == "openai_chat":
            url = provider.base_url if provider.base_url.endswith("/chat/completions") else f"{provider.base_url}/chat/completions"
            payload: dict[str, Any] = {
                "model": model.model_id,
                "messages": messages,
                "max_tokens": max_output_tokens,
                "temperature": temperature,
            }
            if capability == "json":
                payload["response_format"] = {"type": "json_object"}
            return (
                payload,
                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                url,
            )
        system = "\n\n".join(item["content"] for item in messages if item.get("role") == "system")
        contents = [
            {"role": "model" if item.get("role") == "assistant" else "user",
             "parts": [{"text": item.get("content", "")}]}
            for item in messages if item.get("role") != "system"
        ]
        url = f"{provider.base_url}/models/{quote(model.model_id, safe='-_.')}:generateContent"
        generation_config: dict[str, Any] = {"maxOutputTokens": max_output_tokens, "temperature": temperature}
        if capability == "json":
            generation_config["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload, {"x-goog-api-key": api_key, "Content-Type": "application/json"}, url

    @staticmethod
    def _normalize(provider: ProviderConfig, model: ModelConfig, body: dict[str, Any], latency_ms: float) -> CompletionResult:
        if provider.protocol == "openai_chat":
            choice = body["choices"][0]
            text = choice["message"]["content"]
            usage = body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            reported_cost = usage.get("cost")
            finish_reason = choice.get("finish_reason")
        else:
            candidate = body["candidates"][0]
            text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
            usage = body.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount")
            completion_tokens = usage.get("candidatesTokenCount")
            reported_cost = None
            finish_reason = candidate.get("finishReason")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty completion")
        cost = None
        if isinstance(reported_cost, (int, float)) and reported_cost >= 0:
            cost = float(reported_cost)
        if prompt_tokens is not None and completion_tokens is not None:
            if cost is None and model.input_cost_per_million is not None and model.output_cost_per_million is not None:
                cost = (prompt_tokens * model.input_cost_per_million + completion_tokens * model.output_cost_per_million) / 1_000_000
        return CompletionResult(
            provider=provider.name,
            model_id=model.model_id,
            text=text.strip(),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            finish_reason=finish_reason,
        )

    def _log(self, event: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {"event_id": os.urandom(8).hex(), "timestamp": datetime.now(UTC).isoformat(), **event}
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
