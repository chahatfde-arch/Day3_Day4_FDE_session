"""One door to every LLM: routing, fallbacks, retries, rate limiting, caching and offline mode.

Why a gateway? Free tiers are small (hundreds of requests a day) and change often.
If every module called a provider directly, a quota error in one place would break
the whole demo. Instead every call goes through `complete()`:

    1. Look in the on-disk cache. Same prompt, same model: same answer, zero quota.
    2. If LLM_OFFLINE=1, return a deterministic stub answer (tests, CI, no keys).
    3. Otherwise call the primary model through LiteLLM, falling back in order.
    4. Validate the answer against a Pydantic model when one is given, retry on failure.

Every call is traced with Langfuse when keys are present (see creditlens.observability).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from creditlens.config import settings
from creditlens.observability import observe, record_generation

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when every provider failed and no cached or offline answer exists."""


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    parsed: Optional[BaseModel] = None
    attempts: int = 1


# ---------------------------------------------------------------------------
# Cache: a folder of JSON files keyed by a hash of (model, messages, schema).
# ---------------------------------------------------------------------------


class DiskCache:
    def __init__(self, folder: Path):
        self.folder = folder

    def _path(self, key: str) -> Path:
        return self.folder / f"{key}.json"

    @staticmethod
    def key(model: str, messages: list[dict], schema: Optional[dict]) -> str:
        blob = json.dumps({"m": model, "msgs": messages, "schema": schema}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def put(self, key: str, value: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Rate limiter: never send more than N requests in any 60-second window.
# ---------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = max(1, per_minute)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            while self._times and now - self._times[0] > 60:
                self._times.popleft()
            if len(self._times) >= self.per_minute:
                sleep_for = 60 - (now - self._times[0]) + 0.05
                time.sleep(max(0, sleep_for))
            self._times.append(time.time())


# ---------------------------------------------------------------------------
# Offline stub: a tiny rule-based "model" so the whole system runs without keys.
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    """The text of a message. Multimodal messages (text plus an image) are a list of parts."""
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")


def _offline_answer(messages: list[dict], response_model: Optional[Type[BaseModel]]) -> str:
    """Return something shaped like a real answer. Modules may register better stubs."""
    user_text = " ".join(_text_of(m.get("content", "")) for m in messages if m.get("role") == "user")
    if response_model is not None:
        handler = OFFLINE_HANDLERS.get(response_model.__name__)
        if handler:
            return json.dumps(handler(messages))
        # Generic: build an instance from field defaults and examples where possible.
        return json.dumps(_skeleton(response_model.model_json_schema()))
    return f"[offline stub] I read {len(user_text)} characters and have no model to answer with."


def _skeleton(schema: dict, defs: Optional[dict] = None) -> Any:
    defs = defs or schema.get("$defs", {})
    if "$ref" in schema:
        return _skeleton(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        return _skeleton(schema["anyOf"][0], defs)
    if "oneOf" in schema:
        return _skeleton(schema["oneOf"][0], defs)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        return {k: _skeleton(v, defs) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return []
    if t == "string":
        return schema.get("default", "offline stub")
    if t == "integer":
        return schema.get("default", 0)
    if t == "number":
        return schema.get("default", 0.0)
    if t == "boolean":
        return schema.get("default", False)
    return schema.get("default")


# Modules register smarter offline stubs here: {"ModelClassName": fn(messages) -> dict}
OFFLINE_HANDLERS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# The gateway
# ---------------------------------------------------------------------------


@dataclass
class Gateway:
    primary: str = settings.llm.primary
    fallbacks: tuple[str, ...] = settings.llm.fallbacks
    offline: bool = settings.llm.offline
    cache: Optional[DiskCache] = field(default_factory=lambda: DiskCache(settings.paths.cache / "llm")
                                       if settings.llm.cache_enabled else None)
    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(settings.llm.requests_per_minute))
    max_retries: int = settings.llm.max_retries
    calls: int = 0
    cache_hits: int = 0

    @property
    def models(self) -> list[str]:
        return [self.primary, *self.fallbacks]

    def _has_key(self, model: str) -> bool:
        provider = model.split("/", 1)[0]
        needed = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
                  "anthropic": "ANTHROPIC_API_KEY", "ollama": None}.get(provider, None)
        return needed is None or bool(os.environ.get(needed))

    def _call_provider(self, model: str, messages: list[dict], json_mode: bool,
                       temperature: float, max_tokens: int) -> LLMResponse:
        import litellm  # imported lazily so tests run without it installed

        litellm.suppress_debug_info = True
        self.limiter.wait()
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                  "max_tokens": max_tokens, "num_retries": 0}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = litellm.completion(**kwargs)
        usage = getattr(resp, "usage", None)
        return LLMResponse(text=resp.choices[0].message.content or "", model=model,
                           prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                           completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                           latency_s=round(time.perf_counter() - t0, 3))

    @observe(name="llm.complete")
    def complete(self, messages: list[dict], response_model: Optional[Type[T]] = None, *,
                 model: Optional[str] = None, temperature: Optional[float] = None,
                 max_tokens: int = 2048, use_cache: bool = True) -> LLMResponse:
        """Call the LLM. If `response_model` is given, the answer is parsed into it (with retries)."""
        temperature = settings.llm.temperature if temperature is None else temperature
        schema = response_model.model_json_schema() if response_model else None
        if response_model is not None:
            messages = _with_schema_instruction(messages, schema)
        models = [model] if model else self.models
        key = DiskCache.key(models[0], messages, schema)

        if self.cache and use_cache:
            hit = self.cache.get(key)
            if hit:
                self.cache_hits += 1
                r = LLMResponse(**{k: v for k, v in hit.items() if k != "parsed"}, cached=True)
                if response_model:
                    r.parsed = response_model.model_validate_json(_extract_json(r.text))
                record_generation(model=r.model, input=messages, output=r.text, cached=True)
                return r

        if self.offline:
            text = _offline_answer(messages, response_model)
            r = LLMResponse(text=text, model="offline-stub")
            if response_model:
                r.parsed = response_model.model_validate_json(text)
            return r

        last_error: Optional[Exception] = None
        attempt_messages = list(messages)
        for attempt in range(1, self.max_retries + 1):
            for m in models:
                if not self._has_key(m):
                    continue
                try:
                    self.calls += 1
                    r = self._call_provider(m, attempt_messages, json_mode=response_model is not None,
                                            temperature=temperature, max_tokens=max_tokens)
                    r.attempts = attempt
                    if response_model:
                        r.parsed = response_model.model_validate_json(_extract_json(r.text))
                    if self.cache and use_cache:
                        self.cache.put(key, {"text": r.text, "model": r.model,
                                             "prompt_tokens": r.prompt_tokens,
                                             "completion_tokens": r.completion_tokens,
                                             "latency_s": r.latency_s})
                    record_generation(model=r.model, input=messages, output=r.text,
                                      usage={"input": r.prompt_tokens, "output": r.completion_tokens})
                    return r
                except (ValidationError, ValueError) as e:
                    # The provider answered but the JSON did not fit the schema: tell it what was wrong.
                    last_error = e
                    attempt_messages = messages + [
                        {"role": "assistant", "content": getattr(e, "raw_text", "")},
                        {"role": "user", "content": "Your previous answer did not match the required JSON "
                                                    f"schema. Error: {str(e)[:800]}. Return only valid JSON."}]
                    break  # retry with the same model, carrying the error message
                except Exception as e:  # quota, network, auth: try the next model
                    last_error = e
                    continue
        raise LLMUnavailable(f"All models failed after {self.max_retries} attempts: {last_error}")

    def chat(self, prompt: str, system: str = "", **kw) -> str:
        """Convenience for one-shot text answers."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.complete(msgs, **kw).text

    def extract(self, prompt: str, response_model: Type[T], system: str = "", **kw) -> T:
        """Convenience for structured answers: returns the parsed Pydantic object."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = self.complete(msgs, response_model=response_model, **kw)
        assert r.parsed is not None
        return r.parsed  # type: ignore[return-value]


def _with_schema_instruction(messages: list[dict], schema: dict) -> list[dict]:
    instruction = ("Respond with a single JSON object and nothing else. It must match this JSON schema:\n"
                   + json.dumps(schema, separators=(",", ":")))
    out = [dict(m) for m in messages]
    if out and out[0]["role"] == "system":
        out[0]["content"] = out[0]["content"] + "\n\n" + instruction
    else:
        out.insert(0, {"role": "system", "content": instruction})
    return out


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in prose or code fences. Pull out the first object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        err = ValueError("no JSON object in the answer")
        err.raw_text = text  # type: ignore[attr-defined]
        raise err
    return text[start:end + 1]


_default: Optional[Gateway] = None


def get_gateway() -> Gateway:
    """Process-wide gateway. Modules call `get_gateway().complete(...)`."""
    global _default
    if _default is None:
        _default = Gateway()
    return _default


def embed(texts: list[str], model: Optional[str] = None) -> list[list[float]]:
    """Local embeddings with sentence-transformers: free, offline, cached in memory per model."""
    from sentence_transformers import SentenceTransformer

    model = model or settings.llm.embedding_model
    if model not in _embedders:
        _embedders[model] = SentenceTransformer(model)
    return _embedders[model].encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


_embedders: dict[str, Any] = {}
