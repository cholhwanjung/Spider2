"""Tool definitions for the ECHO-SQL agent (OpenAI tool calling format).

The schema descriptions are minimal on purpose — the system prompt explains
when to use each tool. We keep payloads stable so the prompt cache holds.
"""
import json
import logging
from typing import Any, Dict, List

from echo_sql.agent.error_classifier import classify
from echo_sql.agent.submit_validator import heuristic_warnings
from echo_sql.agent.trial_memory import TrialMemory
from echo_sql.exec.result_compress import sigma_compress
from echo_sql.exec.snowflake_runner import SnowflakeRunner


logger = logging.getLogger("echo_sql.tools")


# OpenAI tool spec (passed via `tools=` to chat.completions).
TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "sample_table",
            "description": "Return up to N sample rows from a Snowflake table for shape/value inspection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Fully-qualified table name."},
                    "n": {"type": "integer", "description": "Row count (default 5, max 50)."},
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_columns",
            "description": "Return DISTINCT count, min, max for the specified columns of a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["table", "columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a Snowflake SQL query. Returns rows on success, error+hint on failure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the FINAL SQL as the answer. Ends the session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                },
                "required": ["sql"],
            },
        },
    },
]


def _quote(name: str) -> str:
    return f'"{name}"'


class ToolRegistry:
    """Tiny dispatcher: maps tool name → method on this object."""

    def __init__(self, runner: SnowflakeRunner):
        self.runner = runner
        self.trials = TrialMemory()

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        method = getattr(self, f"tool_{name}", None)
        if method is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        return method(**arguments)

    # ----- tools -----

    def tool_sample_table(self, table: str, n: int = 5) -> Dict[str, Any]:
        n = max(1, min(int(n or 5), 50))
        sql = f"SELECT * FROM {table} LIMIT {n}"
        result = self.runner.execute(sql, with_limit=False)
        return {"ok": result.get("ok", False), "observation": sigma_compress(result)}

    def tool_inspect_columns(self, table: str, columns: List[str]) -> Dict[str, Any]:
        if not columns:
            return {"ok": False, "error": "columns is empty"}
        parts: List[str] = []
        for c in columns:
            q = _quote(c)
            parts.append(f"COUNT(DISTINCT {q}) AS distinct_{c}")
            parts.append(f"MIN({q}) AS min_{c}")
            parts.append(f"MAX({q}) AS max_{c}")
        sql = f"SELECT {', '.join(parts)} FROM {table}"
        result = self.runner.execute(sql, with_limit=False)
        return {"ok": result.get("ok", False), "observation": sigma_compress(result)}

    def tool_run_sql(self, sql: str) -> Dict[str, Any]:
        # Reject re-runs of an already-tried failing pattern without hitting
        # the DB. The cached hint nudges the model to vary its strategy.
        prior = self.trials.lookup(sql)
        if prior is not None and not prior.get("ok"):
            return {
                "ok": False,
                "error": prior.get("error"),
                "hint": prior.get("hint"),
                "instruction": (
                    "이 SQL은 이전에 동일한 패턴으로 실패했습니다 (Trial Memory). "
                    "전략을 바꾸거나 컬럼/테이블 참조를 다시 검토하세요."
                ),
            }

        result = self.runner.execute(sql, with_limit=False)
        if result.get("ok"):
            self.trials.record(sql, ok=True)
            return {
                "ok": True,
                "observation": sigma_compress(result),
                "rows_returned": len(result.get("rows")) if result.get("rows") is not None else 0,
            }
        err = result.get("error", "")
        hint = classify(sql, err)
        self.trials.record(sql, ok=False, error=err, hint=hint)
        out: Dict[str, Any] = {"ok": False, "error": err}
        if hint:
            out["hint"] = hint
        return out

    def tool_submit(self, sql: str) -> Dict[str, Any]:
        """Validate-then-accept. The agent loop only terminates if `accepted=True`.

        - Compilation/runtime error → `accepted=False` so the loop retries.
        - Suspicious result (heuristic_warnings) → `accepted=False` with the
          warnings attached; the model can re-submit a different SQL or call
          submit again with the same one to force-accept (rare).
        - Clean result → `accepted=True`.
        """
        result = self.runner.execute(sql, with_limit=False)
        if not result.get("ok"):
            err = result.get("error", "")
            hint = classify(sql, err)
            out: Dict[str, Any] = {"ok": False, "accepted": False, "error": err}
            if hint:
                out["hint"] = hint
            out["instruction"] = (
                "Submission rejected: SQL did not execute. Fix the issue and submit again."
            )
            return out

        warnings = heuristic_warnings(sql, result)
        if warnings:
            return {
                "ok": True,
                "accepted": False,
                "observation": sigma_compress(result),
                "warnings": warnings,
                "instruction": (
                    "Submission flagged as suspicious. Either submit a corrected SQL "
                    "or, if you are certain this is correct, call submit() with the "
                    "same SQL again — the second attempt is force-accepted."
                ),
            }
        return {
            "ok": True,
            "accepted": True,
            "observation": sigma_compress(result),
            "sql": sql,
        }


def parse_arguments(raw: str) -> Dict[str, Any]:
    """Tolerate missing/empty/JSON-parse-failure cases gracefully."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"tool args parse failed: {e}; raw={raw[:120]!r}")
        return {"_raw": raw, "_parse_error": str(e)}
