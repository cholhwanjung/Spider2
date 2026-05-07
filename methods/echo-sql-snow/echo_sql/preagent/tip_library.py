from pathlib import Path
from typing import List

import yaml


class TipLibrary:
    """Loads tips from YAML. See `tip_library.yaml` for the schema."""

    def __init__(self, yaml_path: str):
        path = Path(yaml_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        self.tips: List[dict] = data.get("tips", []) or []
