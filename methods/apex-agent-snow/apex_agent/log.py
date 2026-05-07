"""Lightweight file+stream logging.

Every log record auto-flushes (via the standard StreamHandler.emit() → flush()
chain), so a tail of `{output_folder}/run.log` reflects live progress.
"""
import logging
from pathlib import Path


def setup_logging(output_folder: str, level: int = logging.INFO) -> Path:
    log_path = Path(output_folder) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path
