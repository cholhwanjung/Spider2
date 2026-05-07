from typing import Optional

try:
    import tiktoken  # type: ignore
except ImportError:
    tiktoken = None


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Approximate token count. Uses tiktoken if available, else char/4 heuristic."""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except Exception:
                pass
    return max(1, len(text) // 4)
