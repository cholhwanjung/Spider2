"""Universal tips + keyword-matched tips.

Selects ~15 tips (4 universal + ~10 matched) from the 49-tip library.
This intentionally does NOT inject all 49 to avoid attention dilution.
"""
from typing import Iterable, List, Set


def match_tips(tips: List[dict], keywords: Iterable[str]) -> List[dict]:
    """Return universal tips + tips whose trigger_keywords intersect keywords.

    Order: universal first, then matched. Stable within each group.
    """
    kw_set: Set[str] = {k.upper() for k in keywords if k}
    seen: Set[str] = set()
    universal: List[dict] = []
    matched: List[dict] = []
    for tip in tips:
        tip_id = tip.get("id", "")
        if tip_id in seen:
            continue
        if tip.get("universal"):
            universal.append(tip)
            seen.add(tip_id)
            continue
        triggers = {t.upper() for t in tip.get("trigger_keywords", []) if t}
        if triggers & kw_set:
            matched.append(tip)
            seen.add(tip_id)
    return universal + matched


def render_tips(tips: List[dict]) -> str:
    if not tips:
        return "(no tips activated)"
    return "\n".join(f"[{t.get('id','?')}] {t.get('text','').strip()}" for t in tips)
