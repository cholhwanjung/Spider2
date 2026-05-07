from pathlib import Path
from typing import Iterable, List, Set

import yaml


class TipLibrary:
    """Deterministic tip retrieval (PLAN §4.1)."""

    def __init__(self, yaml_path: str):
        path = Path(yaml_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        self.tips: List[dict] = data.get("tips", []) or []

    def match(self, keywords: Iterable[str]) -> List[dict]:
        """Return list of activated tips: all universal + trigger-matched."""
        kw_set: Set[str] = {k.upper() for k in keywords if k}
        out: List[dict] = []
        seen: Set[str] = set()
        for tip in self.tips:
            tip_id = tip.get("id", "")
            if tip_id in seen:
                continue
            if tip.get("universal"):
                out.append(tip)
                seen.add(tip_id)
                continue
            triggers = {t.upper() for t in tip.get("trigger_keywords", []) if t}
            if triggers & kw_set:
                out.append(tip)
                seen.add(tip_id)
        return out

    @staticmethod
    def render(tips: List[dict]) -> str:
        if not tips:
            return "(no tips activated)"
        return "\n".join(f"[{t.get('id','?')}] {t.get('text','').strip()}" for t in tips)
