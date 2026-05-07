"""Normalize a SQL string for structural-equality comparison.

Used by Trial Memory to detect "the same broken pattern again". The goal is
to ignore differences that don't matter (whitespace, alias naming, literal
values) while preserving differences that do (table choice, predicate shape,
projection set).

Strategy:
  - Parse with sqlglot in Snowflake dialect.
  - Replace string/numeric literals with `?` placeholders.
  - Lowercase identifiers (Snowflake's storage representation aside, structural
    identity is what we care about).
  - Re-emit canonical SQL.

Falls back to a whitespace-collapsed comparison if sqlglot can't parse.
"""
import re
from typing import Optional

import sqlglot
from sqlglot import exp


def _sub_literals(tree: exp.Expression) -> exp.Expression:
    for lit in tree.find_all(exp.Literal):
        # Replace with a generic placeholder of the same kind.
        lit.replace(exp.Literal.string("?") if lit.is_string else exp.Literal.number(0))
    return tree


def normalize(sql: str) -> str:
    """Return a canonical form. Empty string for empty input."""
    if not sql or not sql.strip():
        return ""
    try:
        tree = sqlglot.parse_one(sql, dialect="snowflake")
        tree = _sub_literals(tree)
        canon = tree.sql(dialect="snowflake", normalize=True)
        return canon.lower()
    except Exception:
        # Whitespace collapse fallback so we still have something to compare.
        return re.sub(r"\s+", " ", sql.strip().lower())


def is_same_pattern(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return normalize(a) == normalize(b)
