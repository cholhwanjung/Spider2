import json
import re
from typing import List, Dict, Any, Set, Tuple

from apex_agent.llm.client import LLMClient
from apex_agent.llm.tokenizer import count_tokens
from apex_agent.schema_linking.prompts import F_SL_DEL, F_SL_SEL


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


def _format_batch(columns: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"{c['table_fullname']}.{c['column']} — {c.get('type','')} — {c.get('comment','')}".rstrip(" —")
        for c in columns
    )


def _column_id(c: Dict[str, str]) -> str:
    return f"{c['table_fullname']}.{c['column']}"


def _table_match(name: str, table_fullname: str) -> bool:
    """Match a table name from the LLM output against a fully-qualified name.

    LLMs often emit short names ('USERS') or partial paths ('SCHEMA.USERS'),
    so we accept any suffix match against the fullname segments.
    """
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


def _resolve_table_columns(
    table_name: str,
    columns: List[Dict[str, str]],
) -> Set[str]:
    return {_column_id(c) for c in columns if _table_match(table_name, c["table_fullname"])}


def _resolve_table_specific_columns(
    table_name: str,
    column_names: List[str],
    columns: List[Dict[str, str]],
) -> Set[str]:
    out: Set[str] = set()
    for c in columns:
        if not _table_match(table_name, c["table_fullname"]):
            continue
        if c["column"] in column_names:
            out.add(_column_id(c))
    return out


def _batch_columns(
    columns: List[Dict[str, str]],
    target_tokens: int,
    model: str,
) -> List[List[Dict[str, str]]]:
    batches: List[List[Dict[str, str]]] = []
    cur: List[Dict[str, str]] = []
    cur_tok = 0
    for c in columns:
        line = f"{c['table_fullname']}.{c['column']} — {c.get('type','')} — {c.get('comment','')}"
        t = count_tokens(line, model=model)
        if cur and cur_tok + t > target_tokens:
            batches.append(cur)
            cur = []
            cur_tok = 0
        cur.append(c)
        cur_tok += t
    if cur:
        batches.append(cur)
    return batches


class DualPruner:
    """② Dual-Pathway Pruning — paper schema:
       Negative pass yields obviously_irrelevant_tables/_columns.
       Positive pass yields relevant_tables/_columns.
       Survivors per batch B_j: (B_j - deleted) ∪ kept.
    """

    def __init__(
        self,
        llm: LLMClient,
        batch_token_budget: int = 10000,
        temperature: float = 0.2,
    ):
        self.llm = llm
        self.batch_token_budget = batch_token_budget
        self.temperature = temperature

    def _parse_table_column_pairs(
        self,
        entries: List[Any],
        columns: List[Dict[str, str]],
    ) -> Set[str]:
        out: Set[str] = set()
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            table_name = entry.get("table") or entry.get("table_name") or ""
            cols = entry.get("columns") or []
            if isinstance(cols, str):
                cols = [cols]
            out |= _resolve_table_specific_columns(table_name, list(cols), columns)
        return out

    def _parse_table_list(
        self,
        names: List[Any],
        columns: List[Dict[str, str]],
    ) -> Set[str]:
        out: Set[str] = set()
        for name in names or []:
            if not isinstance(name, str):
                continue
            out |= _resolve_table_columns(name, columns)
        return out

    def run(
        self,
        question: str,
        logical_plan: str,
        columns: List[Dict[str, str]],
        evidence: str = "",
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        batches = _batch_columns(columns, self.batch_token_budget, self.llm.model)
        d_pruned_ids: Set[str] = set()
        log: List[Dict[str, Any]] = []

        evidence_text = evidence if evidence.strip() else "(none)"

        for j, batch in enumerate(batches):
            batch_text = _format_batch(batch)
            batch_ids = {_column_id(c) for c in batch}

            del_prompt = F_SL_DEL.format(
                question=question,
                logical_plan=logical_plan,
                schema=batch_text,
                evidence=evidence_text,
            )
            sel_prompt = F_SL_SEL.format(
                question=question,
                logical_plan=logical_plan,
                schema=batch_text,
                evidence=evidence_text,
            )

            del_resp = self.llm.prompt(system=None, user=del_prompt, temperature=self.temperature)
            sel_resp = self.llm.prompt(system=None, user=sel_prompt, temperature=self.temperature)

            del_obj = _parse_json_obj(del_resp["content"])
            sel_obj = _parse_json_obj(sel_resp["content"])

            del_table_set = self._parse_table_list(
                del_obj.get("obviously_irrelevant_tables", []), batch
            )
            del_col_set = self._parse_table_column_pairs(
                del_obj.get("obviously_irrelevant_columns", []), batch
            )
            deleted = (del_table_set | del_col_set) & batch_ids

            keep_table_set = self._parse_table_list(
                sel_obj.get("relevant_tables", []), batch
            )
            keep_col_set = self._parse_table_column_pairs(
                sel_obj.get("relevant_columns", []), batch
            )
            kept = (keep_table_set | keep_col_set) & batch_ids

            survivors = (batch_ids - deleted) | kept
            d_pruned_ids |= survivors

            log.append(
                {
                    "batch_index": j,
                    "batch_size": len(batch),
                    "deleted": sorted(deleted),
                    "kept": sorted(kept),
                    "survivors": sorted(survivors),
                    "del_raw": del_obj,
                    "sel_raw": sel_obj,
                }
            )

        d_pruned = [c for c in columns if _column_id(c) in d_pruned_ids]
        return d_pruned, log
