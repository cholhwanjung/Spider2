from typing import Any, Dict

import pandas as pd


def sigma_compress(result: Dict[str, Any], row_threshold: int = 30, top_k: int = 10) -> str:
    """σ() — compress a SQL execution result for prompt re-injection.

    If the result has > row_threshold rows, return top-k rows + per-column stats
    (count, distinct, min, max, null %, dtype). Otherwise return the full table.
    """
    if not result.get("ok"):
        return f"[ERROR] {result.get('error','')}"

    df: pd.DataFrame = result.get("rows")
    if df is None or len(df) == 0:
        return "[empty result set]"

    if len(df) <= row_threshold:
        return f"[{len(df)} rows]\n{df.to_csv(index=False)}"

    head = df.head(top_k).to_csv(index=False)
    stats_lines = [f"[{len(df)} rows total — showing top {top_k} + stats]", head, "Per-column stats:"]
    for col in df.columns:
        s = df[col]
        try:
            distinct = int(s.nunique(dropna=True))
        except Exception:
            distinct = -1
        try:
            null_pct = float(s.isna().mean()) * 100.0
        except Exception:
            null_pct = -1.0
        sample_min: Any = ""
        sample_max: Any = ""
        try:
            sample_min = s.dropna().min()
            sample_max = s.dropna().max()
        except Exception:
            pass
        stats_lines.append(
            f"  {col}: dtype={s.dtype}, distinct={distinct}, "
            f"null%={null_pct:.1f}, min={sample_min}, max={sample_max}"
        )
    return "\n".join(stats_lines)
