import json
from pathlib import Path
from typing import List, Dict, Optional


def load_instances(input_file: str, ids_file: Optional[str] = None) -> List[Dict]:
    """Load Spider 2.0-Snow instances from a jsonl file, optionally filtered by an ids file."""
    with open(input_file, "r", encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()]

    if ids_file:
        with open(ids_file, "r", encoding="utf-8") as f:
            allowed = {line.strip() for line in f if line.strip()}
        instances = [i for i in instances if i["instance_id"] in allowed]

    return instances


def load_external_knowledge(documents_path: str, filename: Optional[str]) -> str:
    if not filename:
        return ""
    p = Path(documents_path) / filename
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")
