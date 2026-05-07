"""Group date-partitioned tables into logical tables (LLM-free).

Detects tables with names like `<prefix>_20160801` where the suffix is a date
or year, and groups them when their column sets are nearly identical
(Jaccard >= 0.95). This dramatically reduces the table count seen by
downstream stages — e.g. GA_SESSIONS_20160801..20170320 (306 tables) →
GA_SESSIONS_* (1 logical table).
"""
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple


_DATE_SUFFIX_RE = re.compile(r"^(?P<prefix>.+?)_(?P<suffix>\d{4}(\d{2}(\d{2})?)?)$")


def _split_date_suffix(table_name: str) -> Tuple[str, str]:
    """Return (prefix, date_suffix) or (table_name, '') if no date suffix."""
    m = _DATE_SUFFIX_RE.match(table_name)
    if not m:
        return table_name, ""
    return m.group("prefix"), m.group("suffix")


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def preprocess_schema(
    schema: Dict[str, Any],
    jaccard_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Return a list of logical tables.

    Each logical table is:
        {
            "logical_name": str,           # e.g. "GA_SESSIONS_*" or single-table name
            "schema_name": str,            # parent schema/dataset
            "members": [table_fullname...], # original tables collapsed into this group
            "representative": {...},       # representative table meta (most-recent)
            "oldest": {...} | None,         # also kept for schema-drift visibility
            "columns": [{name, type, comment}],  # union from representative
            "partition_key": str | "",     # e.g. "20160801" range marker
            "partition_range": (lo, hi) | None,
        }
    """
    tables = schema.get("tables", [])
    by_full: Dict[str, Dict[str, Any]] = {}
    for t in tables:
        full = t.get("table_fullname") or t.get("table_name", "")
        if full:
            by_full[full] = t

    # Bucket by (schema_prefix, date_partition_prefix)
    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for full in by_full:
        # full is like "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20160801"
        parts = full.split(".")
        leaf = parts[-1]
        parent = ".".join(parts[:-1])
        prefix, suffix = _split_date_suffix(leaf)
        if suffix:
            buckets[(parent, prefix)].append(full)
        else:
            buckets[(parent, leaf)].append(full)

    logical: List[Dict[str, Any]] = []
    for (parent, prefix), members in buckets.items():
        if len(members) == 1:
            t = by_full[members[0]]
            logical.append(_make_entry(t, members, prefix, parent))
            continue

        # Sort by date suffix (lexicographic on YYYY[MM[DD]] works)
        members_sorted = sorted(
            members,
            key=lambda f: _split_date_suffix(f.split(".")[-1])[1],
        )

        # Verify column-set Jaccard against the representative (newest); split
        # off members that diverge.
        rep_full = members_sorted[-1]
        rep_cols = _col_set(by_full[rep_full])
        kept: List[str] = []
        outliers: List[str] = []
        for full in members_sorted:
            if _jaccard(rep_cols, _col_set(by_full[full])) >= jaccard_threshold:
                kept.append(full)
            else:
                outliers.append(full)

        logical.append(
            _make_entry(
                by_full[rep_full],
                kept,
                prefix,
                parent,
                oldest=by_full[kept[0]] if kept else None,
                is_grouped=True,
            )
        )
        # Outliers each become their own logical table.
        for full in outliers:
            t = by_full[full]
            leaf = full.split(".")[-1]
            logical.append(_make_entry(t, [full], leaf, parent))

    return logical


def _col_set(table: Dict[str, Any]) -> set:
    return set(table.get("column_names", []) or [])


def _make_entry(
    table: Dict[str, Any],
    members: List[str],
    prefix: str,
    parent: str,
    oldest: Dict[str, Any] = None,
    is_grouped: bool = False,
) -> Dict[str, Any]:
    cols = table.get("column_names", []) or []
    types = table.get("column_types", []) or []
    descs = table.get("column_descriptions", []) or table.get("column_comments", []) or []

    member_leaves = [m.split(".")[-1] for m in members]
    suffixes = sorted(_split_date_suffix(leaf)[1] for leaf in member_leaves if _split_date_suffix(leaf)[1])
    partition_range = (suffixes[0], suffixes[-1]) if suffixes else None

    if is_grouped and len(members) > 1:
        logical_name = f"{prefix}_*" if partition_range else prefix
        full_name = f"{parent}.{logical_name}" if parent else logical_name
    else:
        logical_name = members[0].split(".")[-1]
        full_name = members[0]

    return {
        "logical_name": full_name,
        "schema_name": parent,
        "members": members,
        "representative": table,
        "oldest": oldest,
        "columns": [
            {
                "name": cols[i],
                "type": types[i] if i < len(types) else "",
                "comment": descs[i] if i < len(descs) else "",
            }
            for i in range(len(cols))
        ],
        "partition_key": prefix if is_grouped and len(members) > 1 else "",
        "partition_range": partition_range if is_grouped and len(members) > 1 else None,
    }


def render_logical_schema(logical_tables: List[Dict[str, Any]], max_chars: int = 20000) -> str:
    """Format grouped tables for prompt injection."""
    out: List[str] = []
    for t in logical_tables:
        n_members = len(t["members"])
        header = f"## {t['logical_name']}"
        if n_members > 1:
            lo, hi = t["partition_range"] or ("?", "?")
            header += f"  ({n_members} date-partitioned tables, range {lo}..{hi})"
        out.append(header)
        for c in t["columns"]:
            line = f"  - {c['name']}: {c['type']}"
            if c["comment"]:
                line += f"  -- {c['comment']}"
            out.append(line)
        out.append("")
    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated]"
    return text
