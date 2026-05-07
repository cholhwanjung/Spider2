"""Per-instance ECHO-SQL pipeline.

Stages:
  0. evidence linking (LLM, only if external knowledge is large)
  1. preagent (LLM-free):
       SchemaPreprocessor → DeterministicProfiler → KeywordExtractor → TipMatcher
  2. agent (LLM, tool calling):
       EchoAgent loop until `submit` or budget
  3. save trajectory + result CSV
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from echo_sql.agent.echo_agent import EchoAgent
from echo_sql.agent.prompts import SYSTEM_VERSION, build_system_prompt, build_user_prompt
from echo_sql.data.loader import load_external_knowledge
from echo_sql.data.schema_cache import SchemaCache
from echo_sql.doc_filter import ExternalDocFilter
from echo_sql.exec.snowflake_runner import SnowflakeRunner
from echo_sql.llm.client import LLMClient
from echo_sql.preagent.deterministic_profiler import (
    DeterministicProfiler,
    render_profiles,
)
from echo_sql.preagent.keyword_extractor import extract_from
from echo_sql.preagent.schema_preprocessor import (
    preprocess_schema,
    render_logical_schema,
)
from echo_sql.preagent.tip_library import TipLibrary
from echo_sql.preagent.tip_matcher import match_tips, render_tips


logger = logging.getLogger("echo_sql.pipeline")


def _save_csv(result: Dict[str, Any], path: Path) -> None:
    df = result.get("rows") if result else None
    if isinstance(df, pd.DataFrame):
        df.to_csv(path, index=False)
    else:
        path.write_text("")


def run_one(
    instance: Dict[str, Any],
    args,
    llm: LLMClient,
    runner: SnowflakeRunner,
    schema_cache: SchemaCache,
    documents_path: str,
    tips_path: str,
) -> Dict[str, Any]:
    inst_id = instance["instance_id"]
    out_dir = Path(args.output_folder) / inst_id
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Evidence linking ──────────────────────────────────────────────
    raw_docs = load_external_knowledge(documents_path, instance.get("external_knowledge"))
    t0 = time.time()
    evidence = ExternalDocFilter(llm).run(
        question=instance["instruction"],
        knowledge_content=raw_docs,
        knowledge_file_name=instance.get("external_knowledge"),
    )
    logger.info(
        f"[{inst_id}] evidence: raw={len(raw_docs)}ch filtered={len(evidence)}ch "
        f"elapsed={time.time()-t0:.1f}s"
    )

    # ── Preagent (LLM-free) ───────────────────────────────────────────
    schema = schema_cache.load(instance["db_id"])
    t0 = time.time()
    logical_tables = preprocess_schema(schema)
    logger.info(
        f"[{inst_id}] schema: raw_tables={len(schema.get('tables', []))} "
        f"logical={len(logical_tables)} elapsed={time.time()-t0:.2f}s"
    )

    t0 = time.time()
    profiles = DeterministicProfiler(runner, hard_cap=args.profile_hard_cap).run(logical_tables)
    ok_profiles = sum(1 for p in profiles if p.get("ok"))
    logger.info(
        f"[{inst_id}] profile: {ok_profiles}/{len(profiles)} ok "
        f"elapsed={time.time()-t0:.1f}s"
    )

    keywords = extract_from(
        question=instance["instruction"],
        evidence=evidence,
    )
    tips = match_tips(TipLibrary(tips_path).tips, keywords)
    logger.info(f"[{inst_id}] keywords={sorted(keywords)} tips={[t.get('id') for t in tips]}")

    # ── Build prompts ─────────────────────────────────────────────────
    system_prompt = build_system_prompt(
        tips=render_tips(tips),
        schema=render_logical_schema(logical_tables),
        profile=render_profiles(profiles),
        evidence=evidence,
    )
    user_prompt = build_user_prompt(question=instance["instruction"])

    (out_dir / "system_prompt.txt").write_text(system_prompt)
    (out_dir / "user_prompt.txt").write_text(user_prompt)

    # ── Agent loop ────────────────────────────────────────────────────
    agent = EchoAgent(
        llm=llm,
        runner=runner,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
    )
    t0 = time.time()
    out = agent.run(system_prompt, user_prompt)
    logger.info(
        f"[{inst_id}] agent: steps={out['steps']} tokens={out['tokens_used']} "
        f"final_sql={'yes' if out['final_sql'] else 'no'} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    final_sql = out["final_sql"]
    final_result = out["final_result"]
    (out_dir / "final.sql").write_text(final_sql or "")
    _save_csv(final_result, out_dir / "result.csv")

    record = {
        "instance_id": inst_id,
        "db_id": instance["db_id"],
        "question": instance["instruction"],
        "evidence": evidence,
        "system_version": SYSTEM_VERSION,
        "logical_tables": [
            {
                "logical_name": t["logical_name"],
                "members": t["members"],
                "columns": [c["name"] for c in t["columns"]],
            }
            for t in logical_tables
        ],
        "profiles": profiles,
        "keywords": sorted(keywords),
        "tips": [t.get("id") for t in tips],
        "trajectory": out["trajectory"],
        "steps": out["steps"],
        "tokens_used_agent": out["tokens_used"],
        "tokens_total": llm.usage_snapshot(),
        "final_sql": final_sql,
        "final_ok": (final_result or {}).get("ok", False),
        "final_error": (final_result or {}).get("error"),
    }
    (out_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    logger.info(
        f"[{inst_id}] DONE elapsed={time.time()-t_start:.1f}s "
        f"final_ok={record['final_ok']} tokens={record['tokens_total']}"
    )
    return record
