import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from apex_agent.data.loader import load_external_knowledge
from apex_agent.data.schema_cache import SchemaCache
from apex_agent.llm.client import LLMClient
from apex_agent.exec.snowflake_runner import SnowflakeRunner
from apex_agent.doc_filter import ExternalDocFilter
from apex_agent.schema_linking.logical_planner import LogicalPlanner
from apex_agent.schema_linking.dual_pruner import DualPruner
from apex_agent.schema_linking.verifier import (
    SemanticLinker,
    ParallelProfiler,
    GlobalSynthesizer,
)
from apex_agent.sql_generation.tip_library import TipLibrary
from apex_agent.sql_generation.guidance import KeywordExtractor
from apex_agent.sql_generation.agent import AgenticSqlAgent
from apex_agent.inference.sampler import CandidateSampler
from apex_agent.inference.voter import ResultMajorityVoter


logger = logging.getLogger("apex_agent.pipeline")


def _save_csv(result: Dict[str, Any], path: Path):
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
) -> Dict[str, Any]:
    instance_id = instance["instance_id"]
    out_dir = Path(args.output_folder) / instance_id
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)

    pipeline_start = time.time()
    schema = schema_cache.load(instance["db_id"])
    full_columns = SchemaCache.flatten_columns(schema)
    logger.info(f"[{instance_id}] db={instance['db_id']} full_columns={len(full_columns)}")

    # ── Evidence Linking (paper) — filter the external knowledge file ──
    raw_docs = load_external_knowledge(documents_path, instance.get("external_knowledge"))
    logger.info(
        f"[{instance_id}] evidence-linking: file={instance.get('external_knowledge')} "
        f"raw_chars={len(raw_docs)}"
    )
    t0 = time.time()
    evidence = ExternalDocFilter(llm).run(
        question=instance["instruction"],
        knowledge_content=raw_docs,
        knowledge_file_name=instance.get("external_knowledge"),
    )
    logger.info(
        f"[{instance_id}] evidence-linking done: filtered_chars={len(evidence)} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    # ── Stage A: Schema Linking ────────────────────────────────
    logger.info(f"[{instance_id}] stage-A.1 logical-planner start")
    t0 = time.time()
    logical_plan = LogicalPlanner(llm).run(instance["instruction"])
    logger.info(
        f"[{instance_id}] stage-A.1 logical-planner done: plan_chars={len(logical_plan)} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    logger.info(f"[{instance_id}] stage-A.2 dual-pruner start")
    t0 = time.time()
    D_pruned, prune_log = DualPruner(llm).run(
        question=instance["instruction"],
        logical_plan=logical_plan,
        columns=full_columns,
        evidence=evidence,
    )
    logger.info(
        f"[{instance_id}] stage-A.2 dual-pruner done: D_pruned={len(D_pruned)}/{len(full_columns)} "
        f"batches={len(prune_log)} elapsed={time.time()-t0:.1f}s"
    )

    logger.info(f"[{instance_id}] stage-A.3 semantic-linker start")
    t0 = time.time()
    semantic = SemanticLinker(llm).run(
        question=instance["instruction"],
        logical_plan=logical_plan,
        d_pruned=D_pruned,
        evidence=evidence,
    )
    logger.info(
        f"[{instance_id}] stage-A.3 semantic-linker done: tables={len(semantic.get('table_functions', {}))} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    logger.info(f"[{instance_id}] stage-A.4 parallel-profiler start")
    t0 = time.time()
    evidence_per_table = ParallelProfiler(llm, runner).run(
        question=instance["instruction"],
        d_pruned=D_pruned,
        semantic=semantic,
        evidence=evidence,
    )
    logger.info(
        f"[{instance_id}] stage-A.4 parallel-profiler done: tables_profiled={len(evidence_per_table)} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    logger.info(f"[{instance_id}] stage-A.5 global-synthesizer start")
    t0 = time.time()
    D_star, synth_log = GlobalSynthesizer(llm).run(
        question=instance["instruction"],
        d_pruned=D_pruned,
        evidence_per_table=evidence_per_table,
        full_columns=full_columns,
        semantic_summary=semantic,
        evidence=evidence,
    )
    logger.info(
        f"[{instance_id}] stage-A.5 global-synthesizer done: D_star={len(D_star)} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    # ── Stage B: SQL Generation ────────────────────────────────
    schema_repr = SchemaCache.render_columns(D_star, max_chars=20000)
    logger.info(f"[{instance_id}] stage-B.1 keyword-extractor start")
    t0 = time.time()
    keywords = KeywordExtractor(llm).run(
        question=instance["instruction"],
        logical_plan=logical_plan,
        schema=schema_repr,
        evidence=evidence,
    )
    guidance_tips = TipLibrary(args.tip_library_path).match(keywords)
    guidance_text = TipLibrary.render(guidance_tips)
    logger.info(
        f"[{instance_id}] stage-B.1 keyword-extractor done: keywords={len(keywords)} "
        f"tips={len(guidance_tips)} elapsed={time.time()-t0:.1f}s"
    )

    sql_agent = AgenticSqlAgent(
        llm=llm,
        runner=runner,
        max_actions=args.max_actions,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    sampler = CandidateSampler(sql_agent, n_samples=args.n_samples)
    logger.info(f"[{instance_id}] stage-B.2 sql-action-loop start n_samples={args.n_samples}")
    t0 = time.time()
    candidates = sampler.run(
        question=instance["instruction"],
        logical_plan=logical_plan,
        schema_repr=schema_repr,
        guidance=guidance_text,
        evidence=evidence,
    )
    ok_count = sum(1 for c in candidates if (c.get("final_result") or {}).get("ok"))
    logger.info(
        f"[{instance_id}] stage-B.2 sql-action-loop done: ok={ok_count}/{len(candidates)} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    for cand in candidates:
        seed = cand.get("seed", 0)
        _save_csv(cand.get("final_result"), candidates_dir / f"cand_{seed}.csv")
        (candidates_dir / f"cand_{seed}.sql").write_text(cand.get("final_sql", ""))

    chosen_idx, vote_log = ResultMajorityVoter(llm).vote(
        candidates, instance["instruction"], schema_repr
    )
    chosen = candidates[chosen_idx]
    final_sql = chosen.get("final_sql", "")
    final_result = chosen.get("final_result")
    logger.info(
        f"[{instance_id}] vote: chosen_seed={chosen.get('seed')} reason={vote_log.get('reason','')}"
    )

    (out_dir / "final.sql").write_text(final_sql)
    _save_csv(final_result, out_dir / "result.csv")

    record = {
        "instance_id": instance_id,
        "db_id": instance["db_id"],
        "question": instance["instruction"],
        "logical_plan": logical_plan,
        "evidence": evidence,
        "d_pruned": [f"{c['table_fullname']}.{c['column']}" for c in D_pruned],
        "d_star": [f"{c['table_fullname']}.{c['column']}" for c in D_star],
        "prune_log": prune_log,
        "semantic": semantic,
        "evidence_per_table": evidence_per_table,
        "synth_log": synth_log,
        "keywords": keywords,
        "guidance_tips": [t.get("id") for t in guidance_tips],
        "candidates": [
            {
                "seed": c.get("seed"),
                "final_sql": c.get("final_sql"),
                "actions_used": c.get("actions_used"),
                "tokens_used": c.get("tokens_used"),
                "trajectory": c.get("trajectory"),
                "ok": (c.get("final_result") or {}).get("ok", False),
                "error": (c.get("final_result") or {}).get("error"),
            }
            for c in candidates
        ],
        "vote": vote_log,
        "chosen_seed": chosen.get("seed"),
        "tokens_total": llm.usage_snapshot(),
    }
    (out_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    logger.info(
        f"[{instance_id}] DONE total_elapsed={time.time()-pipeline_start:.1f}s "
        f"final_sql={'yes' if final_sql else 'no'} final_ok={(final_result or {}).get('ok', False)} "
        f"tokens={record['tokens_total']}"
    )
    return record
