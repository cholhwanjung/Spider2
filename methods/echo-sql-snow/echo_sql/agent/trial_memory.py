"""Track SQL attempts so the agent doesn't loop on the same broken pattern.

A trial is the (normalized SQL → outcome) mapping. When `run_sql` is called
with a SQL that normalizes to a previous failed attempt, the runner returns
the cached error directly with a "you already tried this" hint, skipping the
DB round-trip and forcing the model to vary its strategy.
"""
from typing import Any, Dict, List, Optional

from echo_sql.agent.sql_normalizer import normalize


class TrialMemory:
    def __init__(self):
        # normalized_sql → {sql, ok, error, hint}
        self._trials: Dict[str, Dict[str, Any]] = {}

    def lookup(self, sql: str) -> Optional[Dict[str, Any]]:
        key = normalize(sql)
        return self._trials.get(key)

    def record(self, sql: str, *, ok: bool, error: Optional[str] = None, hint: Optional[str] = None) -> None:
        key = normalize(sql)
        if not key:
            return
        # Don't overwrite a successful trial with anything else.
        prev = self._trials.get(key)
        if prev and prev.get("ok"):
            return
        self._trials[key] = {"sql": sql, "ok": ok, "error": error, "hint": hint}

    def all(self) -> List[Dict[str, Any]]:
        return list(self._trials.values())
