import logging
import os
import threading
import time
from typing import List, Dict, Optional, Any

from openai import OpenAI


logger = logging.getLogger("echo_sql.llm")


def _is_reasoning_model(model: str) -> bool:
    """Models that disallow `temperature`/`top_p` and use `max_completion_tokens`."""
    m = model.lower()
    return m.startswith("o1") or m.startswith("o3") or m.startswith("o4") or m.startswith("gpt-5")


class LLMClient:
    """Thin OpenAI-compatible client with retry, token accounting, and tool calling.

    Auto-selects parameter set per model:
      - reasoning models (o1/o3/o4/gpt-5*): no temperature, `max_completion_tokens`
      - others: `temperature` + `max_tokens`
    """

    DEFAULT_REASONING_MAX_TOKENS = 10000
    DEFAULT_STANDARD_MAX_TOKENS = 4000

    def __init__(
        self,
        model: str,
        default_temperature: float = 0.6,
        request_timeout: float = 120.0,
        default_max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.default_temperature = default_temperature
        self.is_reasoning = _is_reasoning_model(model)
        self.default_max_tokens = (
            default_max_tokens
            if default_max_tokens is not None
            else (
                self.DEFAULT_REASONING_MAX_TOKENS
                if self.is_reasoning
                else self.DEFAULT_STANDARD_MAX_TOKENS
            )
        )
        self._client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=request_timeout,
        )
        self._lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.call_count = 0

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 5,
        tag: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {'content', 'tool_calls', 'prompt_tokens', 'completion_tokens'}.

        `tool_calls` is a list of {'id', 'name', 'arguments'} (arguments is a JSON string).
        """
        effective_max = max_tokens if max_tokens is not None else self.default_max_tokens
        prompt_chars = sum(len(str(m.get("content", "")) or "") for m in messages)

        kwargs: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "n": 1,
        }
        if self.is_reasoning:
            if effective_max is not None:
                kwargs["extra_body"] = {"max_completion_tokens": effective_max}
        else:
            kwargs["temperature"] = (
                temperature if temperature is not None else self.default_temperature
            )
            if effective_max is not None:
                kwargs["max_tokens"] = effective_max
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        with self._lock:
            self.call_count += 1
            call_id = self.call_count

        tag_part = f" [{tag}]" if tag else ""
        logger.info(
            f"call#{call_id}{tag_part} model={self.model} prompt={prompt_chars}ch "
            f"max={effective_max} tools={len(tools) if tools else 0}"
        )

        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            start = time.time()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                content = msg.content or ""
                raw_calls = getattr(msg, "tool_calls", None) or []
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in raw_calls
                ]
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) if usage else 0
                ct = getattr(usage, "completion_tokens", 0) if usage else 0
                # OpenAI puts cached prefix tokens here when prompt caching hits.
                cached = 0
                details = getattr(usage, "prompt_tokens_details", None) if usage else None
                if details is not None:
                    cached = getattr(details, "cached_tokens", 0) or 0
                with self._lock:
                    self.total_prompt_tokens += pt
                    self.total_completion_tokens += ct
                    self.total_cached_tokens += cached
                cache_part = f" cached={cached}" if cached else ""
                logger.info(
                    f"call#{call_id} ok tokens={pt}+{ct}{cache_part} "
                    f"latency={time.time()-start:.1f}s resp={len(content)}ch "
                    f"tool_calls={len(tool_calls)}"
                )
                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "cached_tokens": cached,
                }
            except Exception as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                logger.warning(
                    f"call#{call_id} attempt {attempt+1}/{max_retries} failed after "
                    f"{time.time()-start:.1f}s: {type(e).__name__}: {str(e)[:200]}; sleep {wait}s"
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM call#{call_id} failed after {max_retries} retries: {last_err}")

    def prompt(
        self,
        system: Optional[str],
        user: str,
        **kwargs,
    ) -> Dict[str, Any]:
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return self.chat(msgs, **kwargs)

    def usage_snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "cached_tokens": self.total_cached_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "call_count": self.call_count,
            }
