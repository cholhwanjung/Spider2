import json
import logging
import re
import threading
import time
from typing import Dict, Any, Optional

import pandas as pd
import snowflake.connector
from snowflake.connector.errors import ProgrammingError, DatabaseError


logger = logging.getLogger("apex_agent.snowflake")

_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
_AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY|HAVING)\b", re.IGNORECASE)


class SnowflakeRunner:
    """Snowflake SQL executor with optional LIMIT injection and per-instance budget tracking."""

    def __init__(
        self,
        credential_path: str,
        default_timeout: int = 60,
        scan_byte_limit: Optional[int] = None,
    ):
        with open(credential_path, "r", encoding="utf-8") as f:
            self.credentials = json.load(f)
        self.default_timeout = default_timeout
        self.scan_byte_limit = scan_byte_limit
        self._lock = threading.Lock()
        self.scanned_bytes = 0
        self._conn = None

    def _connect(self):
        if self._conn is None:
            self._conn = snowflake.connector.connect(
                **self.credentials,
                login_timeout=self.default_timeout,
                network_timeout=self.default_timeout,
            )
        return self._conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    @staticmethod
    def inject_limit(sql: str, limit: int = 1000) -> str:
        """Add LIMIT to read queries that don't already have one and aren't aggregations only."""
        if _LIMIT_RE.search(sql):
            return sql
        stripped = sql.strip().rstrip(";")
        if not stripped.lower().startswith("select"):
            return sql
        if _AGG_RE.search(stripped):
            return sql
        return f"{stripped}\nLIMIT {limit}"

    def execute(
        self,
        sql: str,
        timeout: Optional[int] = None,
        with_limit: bool = False,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Execute SQL. Returns {ok, rows, columns, error, sql_executed, elapsed_s}."""
        if with_limit:
            sql = self.inject_limit(sql, limit=limit)

        if self.scan_byte_limit is not None and self.scanned_bytes > self.scan_byte_limit:
            return {
                "ok": False,
                "rows": None,
                "columns": [],
                "error": f"scan_byte_limit exceeded ({self.scanned_bytes} > {self.scan_byte_limit})",
                "sql_executed": sql,
                "elapsed_s": 0.0,
            }

        sql_preview = " ".join(sql.split())[:120]
        logger.info(f"sql start: {sql_preview!r}")

        start = time.time()
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql, timeout=timeout or self.default_timeout)

            df = None
            cols: list = []
            if cursor.description:
                cols = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                df = pd.DataFrame(rows, columns=cols)
            row_count = len(df) if df is not None else 0
            logger.info(f"sql ok rows={row_count} cols={len(cols)} elapsed={time.time()-start:.1f}s")

            try:
                stats = cursor.sfqid and conn.cursor().execute(
                    "SELECT BYTES_SCANNED FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION())"
                    f" WHERE QUERY_ID='{cursor.sfqid}'"
                ).fetchone()
                if stats and stats[0]:
                    with self._lock:
                        self.scanned_bytes += int(stats[0])
            except Exception:
                pass

            return {
                "ok": True,
                "rows": df,
                "columns": cols,
                "error": None,
                "sql_executed": sql,
                "elapsed_s": time.time() - start,
            }
        except ProgrammingError as e:
            logger.warning(f"sql fail (ProgrammingError) elapsed={time.time()-start:.1f}s: {str(e)[:200]}")
            return {
                "ok": False,
                "rows": None,
                "columns": [],
                "error": f"SQL Error: {e}",
                "sql_executed": sql,
                "elapsed_s": time.time() - start,
            }
        except DatabaseError as e:
            logger.warning(f"sql fail (DatabaseError) elapsed={time.time()-start:.1f}s: {str(e)[:200]}")
            return {
                "ok": False,
                "rows": None,
                "columns": [],
                "error": f"Database Error: {e}",
                "sql_executed": sql,
                "elapsed_s": time.time() - start,
            }
        except Exception as e:
            logger.warning(f"sql fail (Unexpected) elapsed={time.time()-start:.1f}s: {type(e).__name__}: {str(e)[:200]}")
            return {
                "ok": False,
                "rows": None,
                "columns": [],
                "error": f"Unexpected: {e}",
                "sql_executed": sql,
                "elapsed_s": time.time() - start,
            }
