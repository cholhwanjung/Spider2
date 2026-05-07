"""Tier 1 (deterministic) error classification.

Only matches errors where Snowflake explicitly says a function/feature is
missing. Functions that exist in both Snowflake and BigQuery but with different
signatures (e.g. DATEDIFF) are intentionally NOT in this dict — they go through
as raw error text so the LLM can read the actual Snowflake error message.

Reference: see Method Design.md §4.6.
"""
import re
from typing import Optional


# Mapping of "BigQuery-flavored token the LLM reached for" → Snowflake hint.
_HINTS = {
    "TIMESTAMP_SECONDS": "Snowflake에는 TIMESTAMP_SECONDS가 없음. TO_TIMESTAMP(epoch_seconds) 사용.",
    "TIMESTAMP_MILLIS":  "Snowflake에는 TIMESTAMP_MILLIS가 없음. TO_TIMESTAMP(epoch_ms / 1000) 사용.",
    "DATE_DIFF":         "Snowflake에서는 DATEDIFF (underscore 없음). 시그니처는 DATEDIFF(unit, start, end).",
    "SAFE_CAST":         "Snowflake에는 SAFE_CAST가 없음. TRY_CAST(x AS type) 사용.",
    "UNNEST":            "Snowflake에는 UNNEST가 없음. LATERAL FLATTEN(input => array_col) 사용.",
    "STRUCT":            "Snowflake에는 STRUCT 리터럴이 없음. OBJECT_CONSTRUCT(...) 사용.",
    "ARRAY_LENGTH":      "Snowflake에서는 ARRAY_SIZE(array_col) 사용.",
    "PARSE_DATE":        "Snowflake에서는 TO_DATE(string, format) 사용.",
    "PARSE_TIMESTAMP":   "Snowflake에서는 TO_TIMESTAMP(string, format) 사용.",
    "FORMAT_DATE":       "Snowflake에서는 TO_CHAR(date, format) 사용.",
    "FORMAT_TIMESTAMP":  "Snowflake에서는 TO_CHAR(timestamp, format) 사용.",
}

# "Unknown function 'X'" 또는 "Unknown user-defined function X"
_UNKNOWN_FN_RE = re.compile(r"unknown\s+(?:user-defined\s+)?function[: ]+['\"]?([A-Z_][A-Z0-9_]*)['\"]?", re.IGNORECASE)
# Wildcard table reference like `FROM <name>*` (BigQuery dialect leak).
_WILDCARD_FROM_RE = re.compile(r"\bFROM\s+[A-Z0-9_.\"]+\*", re.IGNORECASE)
# "invalid identifier 'XYZ'" — fired when Snowflake's case-folded lookup misses.
_INVALID_IDENT_RE = re.compile(r"invalid\s+identifier\s+['\"]([A-Z_][A-Z0-9_]*)['\"]", re.IGNORECASE)
# A bare word in the SQL (used to find unquoted mixed-case references).
_WORD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def classify(sql: str, error_msg: str) -> Optional[str]:
    """Return a hint string if the error matches a known dialect mistake.

    Returns None when the error doesn't match — callers should pass the raw
    error text through to the LLM in that case.
    """
    if not error_msg:
        return None
    err = error_msg

    m = _UNKNOWN_FN_RE.search(err)
    if m:
        fn = m.group(1).upper()
        if fn in _HINTS:
            return _HINTS[fn]

    # `invalid identifier 'FULLVISITORID'` 처럼 Snowflake가 대문자화 룩업에 실패한
    # 경우 — SQL에 동일 이름의 mixed-case unquoted 토큰이 있으면 quoting 누락.
    m = _INVALID_IDENT_RE.search(err)
    if m:
        ident = m.group(1).upper()
        for w in _WORD_RE.finditer(sql):
            tok = w.group(1)
            if tok.upper() == ident and tok != tok.upper():
                return (
                    f"컬럼 `{tok}`이 quote 없이 참조됨. Snowflake는 unquoted 식별자를 "
                    f"대문자화하여 `{ident}`로 찾았으나 실제 저장 이름은 `{tok}` (mixed-case). "
                    f"SQL 안의 모든 `{tok}` 참조를 `\"{tok}\"`로 wrap (CTE 별칭, JOIN ON, "
                    f"VARIANT path access 포함)."
                )

    # Wildcard table reference (`FROM ...*`) is a dead giveaway of
    # BigQuery-flavored thinking; Snowflake will reject it as a syntax error.
    if _WILDCARD_FROM_RE.search(sql):
        return (
            "Snowflake에는 BigQuery식 wildcard 테이블이 없음. "
            "단일 날짜 테이블을 직접 지정하거나 UNION ALL로 작성."
        )

    # Spot-check the SQL itself for common BQ functions even when the error
    # message format differs across Snowflake versions.
    sql_upper = sql.upper()
    for token, hint in _HINTS.items():
        if re.search(rf"\b{token}\s*\(", sql_upper):
            return hint

    return None
