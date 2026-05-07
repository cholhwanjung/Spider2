import re
from typing import List, Set

from apex_agent.llm.client import LLMClient
from apex_agent.sql_generation.prompts import F_SQL_KW


# The paper's output uses lines like `- Keywords: [a, b, c]`. We accept both
# `Keywords:` and `- Keywords:` (with optional leading dash/whitespace).
_KW_LINE_RE = re.compile(r"^\s*-?\s*Keywords:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class KeywordExtractor:
    """④ Deterministic keyword extraction from a SQL realization plan."""

    def __init__(self, llm: LLMClient, temperature: float = 0.3):
        self.llm = llm
        self.temperature = temperature

    def run(
        self,
        question: str,
        logical_plan: str,
        schema: str,
        evidence: str = "",
    ) -> List[str]:
        prompt = F_SQL_KW.format(
            question=question,
            evidence=evidence if evidence.strip() else "(none)",
            schema=schema,
            logical_plan=logical_plan,
        )
        resp = self.llm.prompt(system=None, user=prompt, temperature=self.temperature)
        return self._extract(resp["content"])

    @staticmethod
    def _extract(text: str) -> List[str]:
        kws: Set[str] = set()
        for m in _KW_LINE_RE.finditer(text):
            line = m.group(1).strip().strip("[]")
            for tok in _TOKEN_RE.findall(line):
                kws.add(tok.upper())
        return sorted(kws)
