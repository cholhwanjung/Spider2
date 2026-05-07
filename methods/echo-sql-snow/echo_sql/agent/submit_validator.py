"""Validate the SQL the agent wants to submit.

Two layers:
  1. Snowflake compilation/execution: if the SQL fails, the model gets the
     error back and can retry (no fall-through to a half-broken final SQL).
  2. Result heuristics: even if execution succeeds, flag results that almost
     certainly aren't the intended answer (single 0/1, all NULLs, COUNT(*)
     with no GROUP BY when the question expects per-entity rows, etc.).
"""
import re
from typing import Any, Dict, List, Optional


_BARE_COUNT_RE = re.compile(
    r"^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)\s+(?:AS\s+\w+\s+)?FROM\b",
    re.IGNORECASE,
)


def heuristic_warnings(sql: str, result: Dict[str, Any]) -> List[str]:
    """Return human-readable warnings about a successfully-executed SQL.

    Empty list means the result looks plausible. Each warning is a short
    sentence the model should consider before re-submitting.
    """
    warnings: List[str] = []
    df = result.get("rows")
    if df is None:
        return warnings

    n_rows = len(df)
    n_cols = len(df.columns) if hasattr(df, "columns") else 0

    if n_rows == 0:
        warnings.append("결과가 0행입니다. WHERE 조건이 너무 엄격하지 않은지 확인.")
        return warnings

    # Bare COUNT(*) with single scalar result — almost always a debug query.
    if _BARE_COUNT_RE.match(sql.strip()) and n_rows == 1 and n_cols == 1:
        warnings.append(
            "단일 COUNT(*) 결과로 보입니다 (디버그 쿼리?). "
            "원래 질문이 per-entity 정보를 요구한다면 GROUP BY 또는 SELECT 컬럼을 늘려야 합니다."
        )

    # Single row, single value of 0/1 — typically a sanity probe, not the answer.
    if n_rows == 1 and n_cols == 1:
        try:
            val = df.iloc[0, 0]
            if val in (0, 1, "0", "1", True, False):
                warnings.append(
                    f"결과가 단일 값 {val!r}입니다. 의도한 답인지 확인 — "
                    "보통 행/컬럼이 더 필요합니다."
                )
        except Exception:
            pass

    # Single row, all NULL — usually a broken aggregation.
    if n_rows == 1:
        try:
            if all(df.iloc[0].isna()):
                warnings.append("단일 행이 모두 NULL입니다. JOIN 조건 또는 필터가 너무 엄격할 수 있음.")
        except Exception:
            pass

    return warnings
