"""LLM backend.

- `LLM` protocol: one method, `complete()`.
- `OpenAICompatibleLLM`: works for OpenAI and Ollama (`base_url="http://localhost:11434/v1"`).
- `LoggingLLM`: wraps any LLM and records every call to a sink (normally a RunHandle).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TypeVar, runtime_checkable

from pydantic import BaseModel

from ..schemas import LLMCallRecord

T = TypeVar("T", bound=BaseModel)
Message = dict[str, Any]

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@runtime_checkable
class LLM(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: type[T] | None = None,
        images: Sequence[Path] = (),
        temperature: float = 0.0,
    ) -> str | T:
        """If `schema` is given, returns a validated instance; else raw text.
        `images` are attached to the last user message."""
        ...


# ---------------------------------------------------------------------------

def _image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def attach_images(messages: Sequence[Message], images: Sequence[Path]) -> list[Message]:
    """Return a copy of messages with images appended to the last user message."""
    msgs = [dict(m) for m in messages]
    if not images:
        return msgs
    for m in reversed(msgs):
        if m["role"] == "user":
            content = m["content"]
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
            parts.extend(_image_part(Path(p)) for p in images)
            m["content"] = parts
            return msgs
    raise ValueError("No user message to attach images to.")


def parse_structured(text: str, schema: type[T]) -> T:
    """Validate JSON text into `schema`, tolerating markdown fences."""
    cleaned = _FENCE.sub("", text.strip())
    return schema.model_validate_json(cleaned)


class OpenAICompatibleLLM:
    """OpenAI chat-completions client. Set base_url for Ollama or any compatible server.

    Handles two quirks of newer OpenAI reasoning models:
    - they take `max_completion_tokens`, not `max_tokens`
    - some reject a non-default `temperature`; we retry once without it and remember.
    """

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int = 8000):
        from openai import OpenAI  # lazy: keeps tests free of the dependency
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._temperature_ok = True
        self.last_usage: dict[str, int | None] = {}

    def complete(self, messages, *, schema=None, images=(), temperature=0.0):
        msgs = attach_images(messages, images)
        kwargs: dict[str, Any] = dict(model=self.model, messages=msgs)
        # OpenAI proper wants max_completion_tokens; compatible servers (Ollama) want max_tokens.
        kwargs["max_completion_tokens" if self.base_url is None else "max_tokens"] = self.max_tokens
        if self._temperature_ok and temperature is not None:
            kwargs["temperature"] = temperature
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            }
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if "temperature" in kwargs and "temperature" in str(e).lower():
                self._temperature_ok = False
                kwargs.pop("temperature")
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        usage = getattr(resp, "usage", None)
        self.last_usage = {
            "tokens_in": getattr(usage, "prompt_tokens", None),
            "tokens_out": getattr(usage, "completion_tokens", None),
        }
        text = resp.choices[0].message.content or ""
        if schema is not None and not text.strip():
            raise ValueError(
                f"Empty response from {self.model} "
                f"(finish_reason={resp.choices[0].finish_reason}); "
                "raise max_tokens if the model is spending its budget on reasoning."
            )
        return parse_structured(text, schema) if schema is not None else text


# ---------------------------------------------------------------------------

Sink = Callable[[LLMCallRecord], None]


class LoggingLLM:
    """Wraps an LLM; every call is written to `sink` as an LLMCallRecord.

    The pipeline sets `agent` and `attempt_index` before handing the wrapper
    to an agent, so call sites stay clean.
    """

    def __init__(self, inner: LLM, sink: Sink, agent: str = "unknown"):
        self.inner = inner
        self.sink = sink
        self.agent = agent
        self.attempt_index: int | None = None

    @property
    def model(self) -> str:
        return self.inner.model

    def for_agent(self, agent: str) -> "LoggingLLM":
        """A sibling wrapper sharing inner/sink but tagged with a different agent name."""
        w = LoggingLLM(self.inner, self.sink, agent)
        w.attempt_index = self.attempt_index
        return w

    def complete(self, messages, *, schema=None, images=(), temperature=0.0):
        t0 = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            result = self.inner.complete(messages, schema=schema, images=images,
                                         temperature=temperature)
            return result
        except Exception as e:  # log, then re-raise
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            usage = getattr(self.inner, "last_usage", {}) or {}
            self.sink(LLMCallRecord(
                id=uuid.uuid4().hex[:12],
                agent=self.agent,
                model=self.model,
                messages=_loggable(messages, images),
                response=(result.model_dump() if isinstance(result, BaseModel)
                          else result if error is None else {"error": error}),
                latency_s=round(time.perf_counter() - t0, 3),
                tokens_in=usage.get("tokens_in"),
                tokens_out=usage.get("tokens_out"),
                attempt_index=self.attempt_index,
            ))


def _loggable(messages: Sequence[Message], images: Sequence[Path]) -> list[Message]:
    """Messages as sent, but with image paths instead of base64 blobs."""
    out = [dict(m) for m in messages]
    if images:
        out.append({"role": "_images", "content": [str(p) for p in images]})
    return out
