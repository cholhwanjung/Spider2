import csv
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional


class SchemaCache:
    """Loads and caches per-DB schema metadata from spider2-snow/resource/databases/.

    Each DB folder layout (observed):
        {db_id}/
            {SCHEMA_NAME}/
                DDL.csv                       # table_name, description, DDL
                {TABLE_NAME}.json             # {table_name, table_fullname,
                                              #  column_names, column_types,
                                              #  description?, sample_rows?}
        {db_id}/
            DDL.csv                           # (alt layout: flat)
            {TABLE_NAME}.json
    """

    def __init__(self, databases_path: str):
        self.root = Path(databases_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def load(self, db_id: str) -> Dict[str, Any]:
        with self._lock:
            if db_id in self._cache:
                return self._cache[db_id]

        db_path = self.root / db_id
        if not db_path.exists():
            raise FileNotFoundError(f"Schema folder not found: {db_path}")

        tables: List[Dict[str, Any]] = []
        ddl_rows: List[Dict[str, str]] = []

        for ddl_csv in db_path.rglob("DDL.csv"):
            schema_name = ddl_csv.parent.name if ddl_csv.parent != db_path else ""
            with open(ddl_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["schema_name"] = schema_name
                    ddl_rows.append(row)

        for table_json in db_path.rglob("*.json"):
            try:
                with open(table_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
            if not isinstance(meta, dict):
                continue
            if "column_names" not in meta and "table_name" not in meta:
                continue
            tables.append(meta)

        schema = {
            "db_id": db_id,
            "tables": tables,
            "ddl": ddl_rows,
        }

        with self._lock:
            self._cache[db_id] = schema
        return schema

    @staticmethod
    def flatten_columns(schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return a flat list of {table_fullname, column, type, comment} entries."""
        out: List[Dict[str, str]] = []
        for t in schema.get("tables", []):
            full = t.get("table_fullname") or t.get("table_name", "")
            cols = t.get("column_names", [])
            types = t.get("column_types", [])
            comments = t.get("column_descriptions", []) or t.get("column_comments", [])
            for i, col in enumerate(cols):
                out.append(
                    {
                        "table_fullname": full,
                        "column": col,
                        "type": types[i] if i < len(types) else "",
                        "comment": comments[i] if i < len(comments) else "",
                    }
                )
        return out

    @staticmethod
    def render_columns(columns: List[Dict[str, str]], max_chars: Optional[int] = None) -> str:
        """Format columns for prompt injection: `table.column — type — comment`."""
        lines = []
        for c in columns:
            line = f"{c['table_fullname']}.{c['column']} — {c.get('type','')} — {c.get('comment','')}".rstrip(" —")
            lines.append(line)
        text = "\n".join(lines)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated]"
        return text
