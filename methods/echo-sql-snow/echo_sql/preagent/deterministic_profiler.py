"""1-shot deterministic SQL profiling per logical table (LLM-free).

For each logical table (after partition grouping), emit ONE Snowflake query
that returns row count + per-column NULL ratio + min/max for numeric/date
columns + APPROX_TOP_K for string columns. This is the LLM-free replacement
for APEX's `ParallelProfiler`, which used 306 LLM calls on GA360.

Usage:
    profiler = DeterministicProfiler(runner)
    profiles = profiler.run(logical_tables)   # one dict per logical table
    text = render_profiles(profiles)
"""
import logging
from typing import Any, Dict, List

from echo_sql.exec.snowflake_runner import SnowflakeRunner


logger = logging.getLogger("echo_sql.profiler")

# Numeric/date types as they appear in DDL. The set is intentionally permissive.
_NUMERIC_TYPES = ("NUMBER", "INTEGER", "INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC")
_DATE_TYPES = ("DATE", "TIMESTAMP", "DATETIME")
_STRING_TYPES = ("VARCHAR", "STRING", "TEXT", "CHAR")


def _is_type(col_type: str, candidates) -> bool:
    t = (col_type or "").upper()
    return any(c in t for c in candidates)


def _quote(name: str) -> str:
    """Snowflake identifier quoting; the schema files preserve case."""
    return f'"{name}"'


def _build_profile_sql(table_fullname: str, columns: List[Dict[str, str]]) -> str:
    """Return a single SELECT that returns row_count + per-column stats."""
    parts: List[str] = ["COUNT(*) AS row_count"]
    for c in columns:
        name = c["name"]
        q = _quote(name)
        # Always: null ratio
        parts.append(
            f"SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS null_count_{name}"
        )
        if _is_type(c.get("type", ""), _NUMERIC_TYPES + _DATE_TYPES):
            parts.append(f"MIN({q}) AS min_{name}")
            parts.append(f"MAX({q}) AS max_{name}")
        elif _is_type(c.get("type", ""), _STRING_TYPES):
            parts.append(f"COUNT(DISTINCT {q}) AS distinct_{name}")
            parts.append(f"APPROX_TOP_K({q}, 5) AS top_{name}")
    select_list = ",\n  ".join(parts)
    # SAMPLE keeps cost low for very large tables; Snowflake's SAMPLE supports
    # row-count or percent. 100k rows is enough for stats.
    return f"SELECT\n  {select_list}\nFROM {table_fullname}\nSAMPLE (100000 ROWS)"


class DeterministicProfiler:
    # VARIANT-heavy tables can take 60-120s to aggregate even on a sample.
    DEFAULT_TIMEOUT_S = 180

    def __init__(self, runner: SnowflakeRunner, hard_cap: int = 30, timeout_s: int = DEFAULT_TIMEOUT_S):
        self.runner = runner
        self.hard_cap = hard_cap
        self.timeout_s = timeout_s

    def run(self, logical_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return [{logical_name, ok, row_count, columns: {col: stats}}, ...]."""
        out: List[Dict[str, Any]] = []
        for i, lt in enumerate(logical_tables[: self.hard_cap]):
            target = lt["logical_name"]
            # If the logical name is a partition wildcard, profile the
            # representative member instead.
            target_table = (
                lt["members"][-1] if lt["partition_range"] else target
            )
            sql = _build_profile_sql(target_table, lt["columns"])
            logger.info(
                f"profile [{i+1}/{len(logical_tables)}] {target} (probe={target_table})"
            )
            result = self.runner.execute(sql, with_limit=False, timeout=self.timeout_s)
            out.append(_parse_profile_result(lt, result))
        return out


def _parse_profile_result(lt: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    if not result.get("ok"):
        return {
            "logical_name": lt["logical_name"],
            "ok": False,
            "error": result.get("error"),
            "columns": {},
        }
    df = result.get("rows")
    if df is None or len(df) == 0:
        return {"logical_name": lt["logical_name"], "ok": True, "row_count": 0, "columns": {}}
    row = df.iloc[0].to_dict()
    row_count = int(row.get("ROW_COUNT", 0) or 0)
    cols: Dict[str, Dict[str, Any]] = {}
    for c in lt["columns"]:
        name = c["name"]
        # Snowflake returns column names uppercase by default.
        up = name.upper()
        stats: Dict[str, Any] = {"type": c["type"]}
        nc = row.get(f"NULL_COUNT_{up}")
        if nc is not None and row_count > 0:
            stats["null_pct"] = round(float(nc) / row_count * 100, 1)
        for k_in, k_out in (("MIN_", "min"), ("MAX_", "max"), ("DISTINCT_", "distinct"), ("TOP_", "top")):
            v = row.get(f"{k_in}{up}")
            if v is not None:
                stats[k_out] = v
        cols[name] = stats
    return {
        "logical_name": lt["logical_name"],
        "ok": True,
        "row_count": row_count,
        "columns": cols,
    }


def render_profiles(profiles: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """σ-compress: one block per table with row_count + interesting stats."""
    out: List[str] = []
    for p in profiles:
        if not p.get("ok"):
            out.append(f"## {p['logical_name']}  [profile error: {str(p.get('error'))[:120]}]")
            continue
        rc = p.get("row_count", 0)
        out.append(f"## {p['logical_name']}  ({rc:,} rows in 100k sample)")
        for name, stats in p.get("columns", {}).items():
            bits = [f"type={stats.get('type','')}"]
            if "null_pct" in stats:
                bits.append(f"null%={stats['null_pct']}")
            for k in ("min", "max", "distinct", "top"):
                if k in stats:
                    v = str(stats[k])
                    if len(v) > 80:
                        v = v[:80] + "…"
                    bits.append(f"{k}={v}")
            out.append(f"  - {name}: " + ", ".join(bits))
        out.append("")
    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated]"
    return text
