"""Regex-based keyword extraction (LLM-free).

Replaces APEX's LLM-based KeywordExtractor. Operates on the natural-language
question + (optional) logical_plan + evidence. The output is a set of uppercase
keyword tokens that the TipMatcher uses for tip retrieval.

Patterns mirror the rules in `Tip Library Details.md` §C.3.
"""
import re
from typing import Iterable, Set


# (regex, keyword) — patterns are case-insensitive; keyword is uppercase.
_PATTERNS = [
    # SQL operations / functions
    (r"\b(rank|dense_rank|row_number)\b",                  "RANK"),
    (r"\bpartition\s+by|\bover\s*\(",                       "WINDOW"),
    (r"\bgroup\s+by\b",                                     "GROUP_BY"),
    (r"\bhaving\b",                                         "HAVING"),
    (r"\bdistinct\b",                                       "DISTINCT"),
    (r"\bcase\s+when\b",                                    "CASE_WHEN"),
    (r"\bwith\s+\w+\s+as\b",                                "CTE"),
    (r"\b(inner\s+join|left\s+join|right\s+join|join)\b",   "JOIN"),
    (r"\b(count|sum|avg|min|max|median)\b",                 "AGGREGATION"),
    (r"\bcoalesce|nvl\b",                                   "NULL"),
    # Question-side intent
    (r"\bhow\s+many\b",                                     "COUNT_QUESTION"),
    (r"\b(highest|lowest|top|bottom|maximum|minimum|best|worst|most|least)\b", "EXTREME"),
    (r"\b(order\s+by|sort)\b",                              "ORDER_BY"),
    (r"\b(percentage|percent|ratio|proportion|rate)\b",     "RATIO"),
    (r"\baverage|\bavg\b",                                  "AVERAGE"),
    (r"\b(null|missing|empty)\b",                           "NULL"),
    (r"\b(more\s+than|less\s+than|greater\s+than|between|at\s+least|at\s+most|above|below|over|under|exceed)\b",
                                                            "COMPARISON"),
    (r"\b(name|title|description)\b",                       "NAME_FIELD"),
    (r"\b(contain|include|starts?\s+with|ends?\s+with|like)\b", "PARTIAL_MATCH"),
    # Multi-step
    (r"\b(subquery|nested|step\s*\d)\b",                    "SUBQUERY"),
    # Filter intent (quoted values usually mean exact match)
    (r"=\s*'[^']+'",                                        "EXACT_VALUE"),
    # Time
    (r"\b(year|month|day|date|datetime|timestamp|epoch)\b", "DATE"),
]


def extract_keywords(*texts: str) -> Set[str]:
    """Concatenate the input texts and extract keywords by regex match."""
    blob = "\n".join(t for t in texts if t)
    if not blob:
        return set()
    out: Set[str] = set()
    for pat, kw in _PATTERNS:
        if re.search(pat, blob, flags=re.IGNORECASE):
            out.add(kw)
    return out


def extract_from(question: str, logical_plan: str = "", evidence: str = "") -> Set[str]:
    """Convenience wrapper for the typical caller."""
    return extract_keywords(question, logical_plan, evidence)
