from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
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
    def __init__(
        self,
        message: str,
        *,
        fallbackable: bool | None = None,
        retryable: bool | None = None,
        category: str = "provider_error",
        cooldown_seconds: float = 0,
        http_status: int | None = None,
        provider_code: str | None = None,
        provider_message: str | None = None,
    ):
        super().__init__(message)
        if fallbackable is None:
            fallbackable = retryable
        if fallbackable is None:
            raise TypeError("fallbackable must be provided.")
        self.fallbackable = fallbackable
        self.retryable = fallbackable  # Backward-compatible alias.
        self.category = category
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.http_status = http_status
        self.provider_code = provider_code
        self.provider_message = provider_message


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _classify_http_error(
    status_code: int,
    error_code: str,
    error_type: str,
    message: str,
    retry_after: str | None = None,
) -> tuple[str, bool, float]:
    normalized = f"{error_code} {error_type} {message}".lower().replace("-", "_")
    daily_quota_markers = (
        "daily", "per day", "requests per day", "tokens per day", "rpd", "tpd",
        "daily_limit", "daily_quota",
    )
    quota_markers = (
        "quota_exceeded", "insufficient_quota", "quota exceeded", "quota has been exceeded",
        "quota exhausted", "exceeded your quota", "quota is exhausted", "usage limit exceeded",
    )
    billing_markers = (
        "credit_balance_exhausted", "blocked_api_access", "billing not enabled",
        "payment required", "insufficient credits", "spend limit", "billing limit",
    )
    rate_limit_markers = (
        "rate_limit", "rate limit", "resource_exhausted", "too many requests",
    )

    if any(marker in normalized for marker in daily_quota_markers):
        return "quota_exhausted", True, 24 * 60 * 60
    if status_code == 402 or any(marker in normalized for marker in billing_markers):
        return "billing_unavailable", True, 60 * 60
    if any(marker in normalized for marker in quota_markers):
        return "quota_exhausted", True, 60 * 60
    if status_code in {401, 403} or any(
        marker in normalized
        for marker in ("permission_denied", "permissions_error", "authentication", "unauthorized")
    ):
        return "access_denied", True, 15 * 60
    if status_code == 404 or any(
        marker in normalized for marker in ("model_not_found", "model_decommissioned", "not_found")
    ):
        return "model_unavailable", True, 15 * 60
    if status_code == 429 or any(marker in normalized for marker in rate_limit_markers):
        return "rate_limited", True, _retry_after_seconds(retry_after) or 60
    if status_code == 408:
        return "request_timeout", True, 30
    if status_code >= 500:
        return "provider_unavailable", True, 60
    if any(marker in normalized for marker in ("failed_precondition", "billing_required")):
        return "provider_unavailable", True, 60 * 60
    return "request_rejected", False, 0


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
        self.cooldown_path = log_path.with_name("provider-cooldowns.json")
        self.key_resolvers: dict[str, Callable[[], str]] = {}
        self._cooldown_lock = Lock()
        self._cooldowns: dict[str, dict[str, Any]] = {}
        self._cooldown_notices: set[tuple[str, float]] = set()
        self._load_cooldowns()

    def _load_cooldowns(self) -> None:
        try:
            data = json.loads(self.cooldown_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        now = time.time()
        self._cooldowns = {
            str(model_name): entry
            for model_name, entry in data.items()
            if isinstance(entry, dict)
            and isinstance(entry.get("until"), (int, float))
            and entry["until"] > now
            and isinstance(entry.get("category"), str)
        }
        if len(self._cooldowns) != len(data):
            self._persist_cooldowns()

    def _persist_cooldowns(self) -> None:
        try:
            self.cooldown_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.cooldown_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(self._cooldowns, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(self.cooldown_path)
        except OSError:
            # A local state-file failure must not stop an API request.
            return

    def _active_cooldown(self, model_name: str) -> tuple[str, float, float] | None:
        with self._cooldown_lock:
            entry = self._cooldowns.get(model_name)
            if entry is None:
                return None
            remaining = float(entry["until"]) - time.time()
            if remaining <= 0:
                del self._cooldowns[model_name]
                self._persist_cooldowns()
                return None
            return str(entry["category"]), remaining, float(entry["until"])

    def _set_cooldown(self, model_name: str, category: str, seconds: float) -> None:
        if seconds <= 0:
            return
        until = time.time() + seconds
        with self._cooldown_lock:
            self._cooldowns[model_name] = {"category": category, "until": until}
            self._persist_cooldowns()

    def _clear_cooldown(self, model_name: str) -> None:
        with self._cooldown_lock:
            if self._cooldowns.pop(model_name, None) is not None:
                self._persist_cooldowns()

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
            error_code = str(provider_error.get("code") or "")
            error_type = str(provider_error.get("type") or provider_error.get("status") or "")
            error_message = str(provider_error.get("message") or "")
            category, fallbackable, cooldown_seconds = _classify_http_error(
                error.code,
                error_code,
                error_type,
                error_message,
                error.headers.get("Retry-After") if error.headers else None,
            )
            quota_exhausted = category == "quota_exhausted"
            self._log({"provider": provider.name, "model_id": model.model_id, "latency_ms": (time.perf_counter() - started) * 1000,
                       "status": "error", "error_type": "HTTPError", "http_status": error.code,
                       "provider_error_code": error_code or error_type or None,
                       "provider_error_message": error_message[:500] if error_message else None,
                       "failure_category": category, "quota_exhausted": quota_exhausted,
                       "fallbackable": fallbackable, "cooldown_seconds": cooldown_seconds})
            raise ProviderRequestError(
                f"Provider {provider.name}: HTTP {error.code} ({category})"
                + (f" — {error_message[:300]}" if error_message else ""),
                fallbackable=fallbackable,
                category=category,
                cooldown_seconds=cooldown_seconds,
                http_status=error.code,
                provider_code=error_code or error_type or None,
                provider_message=error_message[:500] if error_message else None,
            ) from error
        except URLError as error:
            self._log({"provider": provider.name, "model_id": model.model_id, "latency_ms": (time.perf_counter() - started) * 1000,
                       "status": "error", "error_type": "URLError", "failure_category": "network_error",
                       "fallbackable": True, "cooldown_seconds": 30})
            raise ProviderRequestError(
                f"Provider {provider.name} could not be reached.",
                fallbackable=True,
                category="network_error",
                cooldown_seconds=30,
            ) from error
        except TimeoutError as error:
            self._log({"provider": provider.name, "model_id": model.model_id, "latency_ms": (time.perf_counter() - started) * 1000,
                       "status": "error", "error_type": "TimeoutError", "failure_category": "timeout",
                       "fallbackable": True, "cooldown_seconds": 30})
            raise ProviderRequestError(
                f"Provider {provider.name} timed out.",
                fallbackable=True,
                category="timeout",
                cooldown_seconds=30,
            ) from error
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
        """Try configured models in order and skip models in an active cooldown."""
        if not model_names:
            raise ValueError("At least one model is required for provider rotation.")
        attempted: list[str] = []
        failures: list[str] = []
        for model_name in model_names:
            model = self.models.get(model_name)
            if model is None or capability not in model.capabilities:
                failures.append(f"{model_name}: not configured for {capability}")
                continue
            provider = self.providers[model.provider]
            if not self._api_key(provider):
                self._log({"provider": provider.name, "model_id": model.model_id, "status": "skipped", "error_type": "MissingAPIKey"})
                failures.append(f"{provider.name}: missing API key")
                continue
            cooldown = self._active_cooldown(model_name)
            if cooldown is not None:
                category, remaining, until = cooldown
                notice_key = (model_name, until)
                with self._cooldown_lock:
                    should_log = notice_key not in self._cooldown_notices
                    self._cooldown_notices.add(notice_key)
                if should_log:
                    self._log({
                        "provider": provider.name,
                        "model_id": model.model_id,
                        "status": "skipped",
                        "failure_category": category,
                        "cooldown_remaining_seconds": round(remaining),
                    })
                failures.append(f"{provider.name}: cooling down ({category})")
                continue
            attempted.append(model_name)
            try:
                result = self.complete(
                    model_name,
                    messages,
                    capability=capability,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
                self._clear_cooldown(model_name)
                return result
            except ProviderRequestError as error:
                if error.fallbackable:
                    self._set_cooldown(model_name, error.category, error.cooldown_seconds)
                self._log({
                    "provider": provider.name,
                    "model_id": model.model_id,
                    "status": "fallback" if error.fallbackable else "failed",
                    "error_type": type(error.__cause__).__name__ if error.__cause__ else "ProviderRequestError",
                    "failure_category": error.category,
                    "fallbackable": error.fallbackable,
                    "reason": str(error),
                })
                diagnostic = f"{provider.name}: {error.category}"
                if error.http_status:
                    diagnostic += f" (HTTP {error.http_status}"
                    if error.provider_code:
                        diagnostic += f", {error.provider_code}"
                    diagnostic += ")"
                if error.provider_message:
                    diagnostic += f" — {error.provider_message[:240]}"
                failures.append(diagnostic)
                if not error.fallbackable:
                    raise
        tried = ", ".join(attempted) or "none"
        details = "; ".join(failures) or "no compatible configured models"
        raise RuntimeError(f"No configured model completed the request. Tried: {tried}. Details: {details}.")

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
