import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Set, Tuple

from apex_agent.llm.client import LLMClient
from apex_agent.exec.snowflake_runner import SnowflakeRunner
from apex_agent.exec.profiler import sigma_compress
from apex_agent.schema_linking.prompts import F_SL_SEMANTICS, F_SL_EXP, F_SL_FINAL


_SQL_BLOCK_RE = re.compile(r"```sql\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _parse_json_obj(text: str) -> Dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}


def _render_columns(cols: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"{c['table_fullname']}.{c['column']} — {c.get('type','')} — {c.get('comment','')}".rstrip(" —")
        for c in cols
    )


def _render_columns_for_table(cols: List[Dict[str, str]]) -> str:
    """Render `column_name (column_type): column_description` per line for F_SL_EXP."""
    return "\n".join(
        f"{c['column']} ({c.get('type','')}): {c.get('comment','')}".rstrip(" :()")
        for c in cols
    )


def _group_by_table(cols: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for c in cols:
        out.setdefault(c["table_fullname"], []).append(c)
    return out


def _table_match(name: str, table_fullname: str) -> bool:
    if not name:
        return False
    name_norm = name.strip().strip("`\"' ")
    full_norm = table_fullname.strip()
    if name_norm == full_norm:
        return True
    full_parts = full_norm.split(".")
    name_parts = name_norm.split(".")
    if len(name_parts) > len(full_parts):
        return False
    return full_parts[-len(name_parts):] == name_parts


def _extract_sql_statements(text: str) -> List[str]:
    """Pull all statements from a ```sql ... ``` block, splitting by `;`."""
    blocks = _SQL_BLOCK_RE.findall(text)
    if not blocks:
        return []
    statements: List[str] = []
    for block in blocks:
        for stmt in block.split(";"):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--"):
                continue
            cleaned = "\n".join(
                line for line in stmt.splitlines() if not line.strip().startswith("--")
            ).strip()
            if cleaned:
                statements.append(cleaned)
    return statements


class SemanticLinker:
    """3.3.1 — paper output: {database_structure, query_specific_content_analysis, table_functions}."""

    def __init__(self, llm: LLMClient, temperature: float = 0.2, critical_rules: str = ""):
        self.llm = llm
        self.temperature = temperature
        self.critical_rules = critical_rules

    def run(
        self,
        question: str,
        logical_plan: str,
        d_pruned: List[Dict[str, str]],
        evidence: str = "",
    ) -> Dict[str, Any]:
        prompt = F_SL_SEMANTICS.format(
            CRITICAL_RULES=self.critical_rules,
            question=question,
            logical_plan=logical_plan,
            evidence=evidence if evidence.strip() else "(none)",
            schema=_render_columns(d_pruned),
        )
        resp = self.llm.prompt(system=None, user=prompt, temperature=self.temperature)
        return _parse_json_obj(resp["content"])


class ParallelProfiler:
    """3.3.2 — Per-table exploratory SQL → σ-compressed evidence (single-stage, Round 1)."""

    def __init__(
        self,
        llm: LLMClient,
        runner: SnowflakeRunner,
        temperature: float = 0.4,
        max_workers: int = 4,
        critical_rules: str = "",
        max_sqls_per_table: int = 3,
    ):
        self.llm = llm
        self.runner = runner
        self.temperature = temperature
        self.max_workers = max_workers
        self.critical_rules = critical_rules
        self.max_sqls_per_table = max_sqls_per_table

    def _profile_one(
        self,
        question: str,
        table_fullname: str,
        columns_repr: str,
        semantic_role: str,
        evidence: str,
    ) -> Dict[str, Any]:
        prompt = F_SL_EXP.format(
            CRITICAL_RULES=self.critical_rules,
            table_name=table_fullname,
            columns_repr=columns_repr,
            question=question,
            semantic_role=semantic_role or "(unspecified)",
            evidence=evidence if evidence.strip() else "(none)",
        )
        resp = self.llm.prompt(system=None, user=prompt, temperature=self.temperature)
        text = resp["content"]
        sqls = _extract_sql_statements(text)[: self.max_sqls_per_table]

        observations: List[Dict[str, Any]] = []
        for sql in sqls:
            result = self.runner.execute(sql, with_limit=True, limit=20)
            observations.append(
                {
                    "sql": sql,
                    "observation": sigma_compress(result),
                    "ok": result.get("ok", False),
                }
            )

        return {
            "table": table_fullname,
            "raw_response": text,
            "executions": observations,
        }

    def run(
        self,
        question: str,
        d_pruned: List[Dict[str, str]],
        semantic: Dict[str, Any],
        evidence: str = "",
    ) -> List[Dict[str, Any]]:
        by_table = _group_by_table(d_pruned)
        table_functions: Dict[str, str] = (semantic or {}).get("table_functions", {}) or {}

        tasks = []
        for table_fullname, cols in by_table.items():
            role = ""
            for k, v in table_functions.items():
                if _table_match(k, table_fullname):
                    role = v
                    break
            tasks.append(
                (
                    table_fullname,
                    _render_columns_for_table(cols),
                    role,
                )
            )

        evidence_text = evidence if evidence.strip() else "(none)"
        out: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._profile_one, question, t, c, r, evidence_text
                ): t
                for t, c, r in tasks
            }
            for f in as_completed(futures):
                try:
                    out.append(f.result())
                except Exception as e:
                    out.append({"table": futures[f], "raw_response": "", "executions": [{"sql": "", "observation": f"[exec error: {e}]", "ok": False}]})
        return out


class GlobalSynthesizer:
    """3.3.3 — Single-call synthesis (Round 1 simplification of multi-round flow)."""

    def __init__(self, llm: LLMClient, temperature: float = 0.2):
        self.llm = llm
        self.temperature = temperature

    @staticmethod
    def _render_table_status(
        d_pruned: List[Dict[str, str]],
        full_columns: List[Dict[str, str]],
        evidence_by_table: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        d_pruned_ids = {f"{c['table_fullname']}.{c['column']}" for c in d_pruned}
        by_table_full = _group_by_table(full_columns)

        blocks: List[str] = []
        for table_fullname, cols in by_table_full.items():
            relevant = any(
                f"{c['table_fullname']}.{c['column']}" in d_pruned_ids for c in cols
            )
            label = "[MARKED RELEVANT]" if relevant else "[MARKED IRRELEVANT]"
            cols_lines = "\n".join(
                f"  {c['column']} ({c.get('type','')}): {c.get('comment','')}".rstrip(" :()")
                for c in cols
            )
            obs_lines = []
            for ev in evidence_by_table.get(table_fullname, []):
                for ex in ev.get("executions", []):
                    obs_lines.append(f"  - SQL: {ex.get('sql','')}\n    Obs: {ex.get('observation','')[:600]}")
            obs_text = "\n".join(obs_lines) if obs_lines else "  (none)"
            blocks.append(
                f"Table: {table_fullname} {label}\nColumns:\n{cols_lines}\nObservations:\n{obs_text}"
            )
        return "\n\n".join(blocks)

    def run(
        self,
        question: str,
        d_pruned: List[Dict[str, str]],
        evidence_per_table: List[Dict[str, Any]],
        full_columns: List[Dict[str, str]],
        semantic_summary: Dict[str, Any],
        evidence: str = "",
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        evidence_by_table: Dict[str, List[Dict[str, Any]]] = {}
        for e in evidence_per_table:
            evidence_by_table.setdefault(e.get("table", ""), []).append(e)

        tables_repr = self._render_table_status(d_pruned, full_columns, evidence_by_table)
        db_summary = json.dumps(semantic_summary or {}, ensure_ascii=False, indent=2)

        prompt = F_SL_FINAL.format(
            question=question,
            evidence=evidence if evidence.strip() else "(none)",
            db_summary=db_summary,
            tables_repr=tables_repr,
        )
        resp = self.llm.prompt(system=None, user=prompt, temperature=self.temperature)
        result = _parse_json_obj(resp["content"])

        refined: Dict[str, Any] = result.get("refined_schema", {}) or {}
        rejected: List[Dict[str, Any]] = result.get("rejected_candidates", []) or []
        rejected_ids: Set[str] = set()
        for r in rejected:
            if not isinstance(r, dict):
                continue
            t = r.get("table", "")
            c = r.get("column", "")
            for col in full_columns:
                if _table_match(t, col["table_fullname"]) and col["column"] == c:
                    rejected_ids.add(f"{col['table_fullname']}.{col['column']}")

        wanted: Set[str] = set()
        for table_name, body in refined.items():
            if not isinstance(body, dict):
                continue
            cols_list = body.get("relevant_columns", []) or []
            for entry in cols_list:
                if not isinstance(entry, dict):
                    continue
                col_name = entry.get("column_name") or entry.get("name") or ""
                for col in full_columns:
                    if _table_match(table_name, col["table_fullname"]) and col["column"] == col_name:
                        wanted.add(f"{col['table_fullname']}.{col['column']}")

        wanted -= rejected_ids
        full_lookup = {f"{c['table_fullname']}.{c['column']}": c for c in full_columns}
        d_star = [full_lookup[k] for k in wanted if k in full_lookup]
        if not d_star:
            d_star = list(d_pruned)
        return d_star, result
