"""Tool-based single-loop SQL agent.

Loop:
  1. system + user message → LLM with tools
  2. if tool_calls present: dispatch each, append `tool` messages, repeat
  3. if model emits final SQL via `submit`, stop
  4. when max_steps or max_tokens budget exhausted, fall back to last
     successful run_sql result (if any)
"""
import json
import logging
from typing import Any, Dict, List, Optional

from echo_sql.agent.tools import TOOL_SPECS, ToolRegistry, parse_arguments
from echo_sql.exec.snowflake_runner import SnowflakeRunner
from echo_sql.llm.client import LLMClient


logger = logging.getLogger("echo_sql.agent")


class EchoAgent:
    def __init__(
        self,
        llm: LLMClient,
        runner: SnowflakeRunner,
        max_steps: int = 12,
        max_tokens: int = 200_000,
    ):
        self.llm = llm
        self.runner = runner
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.tools = ToolRegistry(runner)

    BUDGET_WARNING = (
        "**Budget warning** — you are running out of steps/tokens.\n"
        "Call `submit(sql)` on your NEXT turn with the best COMPLETE SQL that "
        "answers the original question.\n"
        "- Do NOT submit a debugging/sanity query (single COUNT(*), single column, "
        "single date table when the question covers a range, etc.).\n"
        "- The submitted SQL must produce the exact result columns the question asks for.\n"
        "- If unsure, submit your best guess — silence is worse than imperfect."
    )

    def run(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        trajectory: List[Dict[str, Any]] = []
        tokens_used = 0
        last_run_sql: Optional[str] = None
        last_run_ok = False
        final_sql: Optional[str] = None
        budget_warned = False
        last_submit_sql: Optional[str] = None  # for force-accept on duplicate

        for step in range(self.max_steps):
            # Inject a one-shot warning when we're close to either budget so
            # the model exits exploration mode and submits.
            near_budget = (
                step >= self.max_steps - 3
                or tokens_used >= int(self.max_tokens * 0.75)
            )
            if near_budget and not budget_warned:
                logger.info(f"step {step+1}: budget warning injected")
                messages.append({"role": "user", "content": self.BUDGET_WARNING})
                budget_warned = True

            resp = self.llm.chat(
                messages=messages,
                tools=TOOL_SPECS,
                tag=f"agent#{step+1}",
            )
            tokens_used += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)
            content = resp.get("content", "") or ""
            tool_calls = resp.get("tool_calls", []) or []

            step_log: Dict[str, Any] = {
                "step": step + 1,
                "content": content,
                "tool_calls": tool_calls,
                "tool_results": [],
                "tokens_used": tokens_used,
            }

            if not tool_calls:
                # Model produced text instead of calling submit. Treat as
                # last-resort: try to extract a SQL fenced block from content.
                logger.info(f"step {step+1}: no tool calls — terminating")
                trajectory.append(step_log)
                break

            # Append the assistant message (with tool_calls) so subsequent
            # `tool` messages link via tool_call_id.
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls
                    ],
                }
            )

            stop = False
            for tc in tool_calls:
                args = parse_arguments(tc.get("arguments", ""))
                name = tc["name"]
                logger.info(f"step {step+1} → {name}({json.dumps(args)[:200]})")
                result = self.tools.call(name, args)
                step_log["tool_results"].append({"name": name, "args": args, "result": _trim(result)})

                # Track the last successful run_sql for fallback.
                if name == "run_sql" and result.get("ok"):
                    last_run_sql = args.get("sql")
                    last_run_ok = True

                if name == "submit":
                    submitted = args.get("sql", "") or ""
                    if result.get("accepted"):
                        final_sql = submitted or last_run_sql
                        stop = True
                    elif submitted and submitted == last_submit_sql:
                        # Same SQL twice → force-accept (model is sure).
                        logger.info(f"step {step+1}: force-accepting duplicate submit")
                        final_sql = submitted
                        stop = True
                    else:
                        last_submit_sql = submitted
                        # Not accepted: fall through, agent gets the rejection
                        # back as a tool message and can revise on next turn.

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, default=str)[:8000],
                    }
                )

            trajectory.append(step_log)
            if stop:
                break
            if tokens_used >= self.max_tokens:
                logger.info(f"step {step+1}: token budget exhausted ({tokens_used}/{self.max_tokens})")
                break

        # Fallback: if no submit but we have a working run_sql, use it.
        if final_sql is None and last_run_ok and last_run_sql:
            final_sql = last_run_sql

        # Re-execute the chosen SQL to get the final result CSV.
        final_result = None
        if final_sql:
            final_result = self.runner.execute(final_sql, with_limit=False)

        return {
            "final_sql": final_sql or "",
            "final_result": final_result,
            "trajectory": trajectory,
            "tokens_used": tokens_used,
            "steps": len(trajectory),
        }


def _trim(obj: Any, n: int = 1500) -> Any:
    """Truncate observation strings inside tool results so trajectory stays small."""
    if isinstance(obj, dict):
        return {k: _trim(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trim(v, n) for v in obj]
    if isinstance(obj, str) and len(obj) > n:
        return obj[:n] + "…[truncated]"
    return obj
