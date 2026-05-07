import argparse
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from apex_agent.log import setup_logging
from apex_agent.data.loader import load_instances
from apex_agent.data.schema_cache import SchemaCache
from apex_agent.llm.client import LLMClient
from apex_agent.exec.snowflake_runner import SnowflakeRunner
from apex_agent.pipeline import run_one


logger = logging.getLogger("apex_agent.main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--databases_path", required=True)
    p.add_argument("--documents_path", required=True)
    p.add_argument("--credential_path", required=True)
    p.add_argument("--output_folder", required=True)

    p.add_argument("--model", default="gpt-5-nano")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--llm_timeout_s", type=float, default=120.0)
    p.add_argument("--llm_default_max_tokens", type=int, default=None,
                   help="Per-call max_tokens (or max_completion_tokens for reasoning models). "
                        "Default: 10k for reasoning, 4k otherwise.")

    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--max_actions", type=int, default=40)
    p.add_argument("--max_tokens", type=int, default=56000,
                   help="Total token budget across the action loop (not per-call).")

    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--ids_file", type=str, default="")

    p.add_argument(
        "--tip_library_path",
        type=str,
        default=str(Path(__file__).parent / "sql_generation" / "tip_library.yaml"),
    )

    p.add_argument("--snowflake_scan_byte_limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(args.output_folder)
    logger.info(f"log file: {log_path}")
    logger.info(f"args: {vars(args)}")

    instances = load_instances(args.input_file, ids_file=args.ids_file or None)
    logger.info(f"loaded {len(instances)} instances. ids_file={args.ids_file or '(none)'}")

    todo = []
    for inst in instances:
        out_dir = Path(args.output_folder) / inst["instance_id"]
        if (out_dir / "result.json").exists():
            logger.info(f"skip {inst['instance_id']} (already processed)")
            continue
        todo.append(inst)
    logger.info(f"to process: {len(todo)} instances")

    schema_cache = SchemaCache(args.databases_path)

    def _make_llm():
        return LLMClient(
            model=args.model,
            default_temperature=args.temperature,
            request_timeout=args.llm_timeout_s,
            default_max_tokens=args.llm_default_max_tokens,
        )

    def _make_runner():
        return SnowflakeRunner(
            credential_path=args.credential_path,
            scan_byte_limit=args.snowflake_scan_byte_limit,
        )

    def _process(inst):
        llm = _make_llm()
        runner = _make_runner()
        try:
            return run_one(inst, args, llm, runner, schema_cache, args.documents_path)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[{inst['instance_id']}] FAILED: {e}\n{tb}")
            err_dir = Path(args.output_folder) / inst["instance_id"]
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(f"{e}\n\n{tb}")
            return {"instance_id": inst["instance_id"], "error": str(e)}
        finally:
            try:
                runner.close()
            except Exception:
                pass

    if args.num_threads <= 1:
        for i, inst in enumerate(todo):
            logger.info(f"START [{i+1}/{len(todo)}] {inst['instance_id']}")
            _process(inst)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
            futures = {ex.submit(_process, inst): inst for inst in todo}
            for n, fut in enumerate(as_completed(futures)):
                inst = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"[{inst['instance_id']}] worker error: {e}")
                else:
                    logger.info(f"FINISHED [{n+1}/{len(todo)}] {inst['instance_id']}")


if __name__ == "__main__":
    main()
