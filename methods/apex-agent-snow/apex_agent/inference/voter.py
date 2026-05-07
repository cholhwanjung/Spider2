import hashlib
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from apex_agent.llm.client import LLMClient
from apex_agent.sql_generation.prompts import F_TIE_BREAKER


_FILENAME_RE = re.compile(r"candidate_(\d+)\.sql", re.IGNORECASE)


def _result_key(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Spider 2.0 relaxed protocol: ignore column order, hash sorted-row representation."""
    if df is None or len(df) == 0:
        return None
    cols = sorted(df.columns)
    df_sorted = df[cols]
    rows = []
    for _, row in df_sorted.iterrows():
        rows.append(tuple(str(x) for x in row))
    rows.sort()
    digest = hashlib.sha256()
    digest.update("|".join(cols).encode())
    for r in rows:
        digest.update(("␟".join(r)).encode())
    return digest.hexdigest()


class ResultMajorityVoter:
    """Result-based majority voting + LLM tie-break (PLAN §4.4)."""

    def __init__(self, llm: LLMClient, tie_break_temperature: float = 0.2):
        self.llm = llm
        self.tie_break_temperature = tie_break_temperature

    def vote(
        self,
        candidates: List[Dict[str, Any]],
        question: str,
        schema_repr: str,
    ) -> Tuple[int, Dict[str, Any]]:
        groups: Dict[Optional[str], List[int]] = defaultdict(list)
        for i, cand in enumerate(candidates):
            res = cand.get("final_result")
            if not res or not res.get("ok"):
                continue
            key = _result_key(res.get("rows"))
            if key is None:
                continue
            groups[key].append(i)

        log: Dict[str, Any] = {
            "groups": {k: v for k, v in groups.items() if k},
            "total_candidates": len(candidates),
            "valid_candidates": sum(len(v) for v in groups.values()),
        }

        if not groups:
            log["chosen"] = 0
            log["reason"] = "no candidate executed successfully — falling back to index 0"
            return 0, log

        sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        top_size = len(sorted_groups[0][1])
        top_groups = [g for g in sorted_groups if len(g[1]) == top_size]

        if len(top_groups) == 1:
            chosen = top_groups[0][1][0]
            log["chosen"] = chosen
            log["reason"] = f"unique majority of size {top_size}"
            return chosen, log

        reps: List[int] = [g[1][0] for g in top_groups]
        rendered = []
        for idx in reps:
            cand = candidates[idx]
            df = (cand.get("final_result") or {}).get("rows")
            preview = df.head(5).to_csv(index=False) if isinstance(df, pd.DataFrame) else "(no rows)"
            strategy = cand.get("strategy") or cand.get("trajectory_summary") or "(none)"
            rendered.append(
                f"### candidate_{idx}.sql\nStrategy:\n{strategy}\n\nFinal SQL:\n{cand.get('final_sql','')}\n\nFirst 5 rows:\n{preview}"
            )

        prompt = F_TIE_BREAKER.format(
            schema=schema_repr,
            question=question,
            candidates_repr="\n\n".join(rendered),
        )
        resp = self.llm.prompt(system=None, user=prompt, temperature=self.tie_break_temperature)
        text = resp["content"]

        chosen = reps[0]
        reason = "tie-break parse failure (fallback to first rep)"
        m = _FILENAME_RE.search(text)
        if m:
            try:
                idx = int(m.group(1))
                if any(idx == r for r in reps):
                    chosen = idx
                    reason = f"tie-break picked candidate_{idx}.sql"
                else:
                    reason = f"LLM picked candidate_{idx}.sql but it isn't a tied rep — fallback"
            except Exception:
                pass

        log["chosen"] = chosen
        log["reason"] = reason
        log["tie_break_response"] = text[:500]
        return chosen, log
